#=
model.jl — Self-contained inference engine for SymbioSLM

Implements the Symbiogenesis architecture: multi-organelle sequence mixing
with learned per-channel gating, RMSNorm, SwiGLU, and weight-tied output.
No Lux dependency — parameters loaded directly from JLD2. CPU-only inference.

Architecture per block (SymbioBlock):
  SymbioSequenceMixer:
    Organelle 1: CausalDepthwiseConv1d (local n-gram patterns, K=4)
    Organelle 2: MonarchMatrix (global structured mixing, single-head)
    Organelle 3: LongConv (global dense causal filter, K=seq_len)
    OrganelleGate (per-channel softmax fusion over 3 organelles)
  → RMSNorm pre-norm + residual
  SwiGLU FFN → RMSNorm pre-norm + residual

References:
  Symbiogenesis framework (DavinciDreams): Evolutionary NAS via organism fusion
  Monarch Mixer (Dao et al., 2023): Sub-quadratic structured mixing
  Hyena (Poli et al., 2023): Long convolutions for sequence modeling
  Margulis (1967): Endosymbiotic theory of organelle evolution
=#

using NNlib
using NNlib: batched_mul
using Statistics
using Random
using JSON3
using TOML

# ═══════════════════════════════════════════════════════════════════
# Model configuration
# ═══════════════════════════════════════════════════════════════════

struct ModelConfig
    arch::String
    embed_dim::Int
    n_layers::Int
    n_monarch_heads::Int
    conv_kernel_size::Int
    context_length::Int
    vocab_size::Int
    weight_tying::Bool
    bias::Bool
end

function load_config_toml(path::String; vocab_size::Int=0)
    raw = TOML.parsefile(path)
    m = get(raw, "model", Dict())
    return ModelConfig(
        get(m, "arch", "symbiogenesis"),
        get(m, "embed_dim", 256),
        get(m, "n_layers", 8),
        get(m, "n_monarch_heads", 1),
        get(m, "conv_kernel_size", 4),
        get(m, "context_length", 256),
        vocab_size,
        get(m, "weight_tying", true),
        get(m, "bias", false),
    )
end

# ═══════════════════════════════════════════════════════════════════
# Character-level tokenizer
# ═══════════════════════════════════════════════════════════════════

struct CharTokenizer
    char_to_idx::Dict{Char, Int}
    idx_to_char::Vector{Char}
    vocab_size::Int
end

function load_char_vocab_json(path::String)
    raw = JSON3.read(read(path, String))
    chars = Char[only(String(s)) for s in raw]
    char_to_idx = Dict(c => i for (i, c) in enumerate(chars))
    return CharTokenizer(char_to_idx, chars, length(chars))
end

function encode(t::CharTokenizer, text::String)
    indices = Int[]
    sizehint!(indices, length(text))
    for c in text
        idx = get(t.char_to_idx, c, nothing)
        idx !== nothing && push!(indices, idx)
    end
    return indices
end

function decode(t::CharTokenizer, indices::AbstractVector{<:Integer})
    buf = IOBuffer()
    for idx in indices
        if 1 <= idx <= t.vocab_size
            write(buf, t.idx_to_char[idx])
        end
    end
    return String(take!(buf))
end

# ═══════════════════════════════════════════════════════════════════
# BPE Tokenizer (GPT-2 style)
# ═══════════════════════════════════════════════════════════════════

struct BPETokenizer
    encoder::Dict{String, Int}
    decoder::Dict{Int, String}
    merges::Vector{Tuple{String, String}}
    merge_ranks::Dict{Tuple{String, String}, Int}
    byte_to_unicode::Dict{UInt8, Char}
    unicode_to_byte::Dict{Char, UInt8}
    vocab_size::Int
    pat::Regex
end

function load_bpe_tokenizer(vocab_path::String, merges_path::String)
    encoder = JSON3.read(read(vocab_path, String), Dict{String, Int})
    decoder = Dict{Int, String}(v => k for (k, v) in encoder)

    merge_lines = readlines(merges_path)
    start = startswith(first(merge_lines), "#") ? 2 : 1
    merges = Tuple{String, String}[]
    for line in merge_lines[start:end]
        parts = split(strip(line))
        length(parts) == 2 && push!(merges, (String(parts[1]), String(parts[2])))
    end
    merge_ranks = Dict{Tuple{String, String}, Int}(m => i for (i, m) in enumerate(merges))

    b2u = _build_byte_to_unicode()
    u2b = Dict{Char, UInt8}(v => k for (k, v) in b2u)

    pat = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

    return BPETokenizer(encoder, decoder, merges, merge_ranks, b2u, u2b,
                        length(encoder), pat)
end

function encode(t::BPETokenizer, text::String)
    tokens = Int[]
    for m in eachmatch(t.pat, text)
        word = m.match
        encoded_chars = [string(t.byte_to_unicode[b]) for b in Vector{UInt8}(word)]
        bpe_tokens = _bpe_encode_word(encoded_chars, t.merge_ranks)
        for tok in bpe_tokens
            id = get(t.encoder, tok, nothing)
            id !== nothing && push!(tokens, id + 1)
        end
    end
    return tokens
end

function decode(t::BPETokenizer, ids::AbstractVector{<:Integer})
    token_strs = [get(t.decoder, id - 1, "") for id in ids]
    joined = join(token_strs)
    out = UInt8[]
    sizehint!(out, length(joined))
    for c in joined
        b = get(t.unicode_to_byte, c, nothing)
        if b !== nothing
            push!(out, b)
        else
            append!(out, codeunits(string(c)))
        end
    end
    return String(out)
end

function _bpe_encode_word(symbols::Vector{String}, merge_ranks::Dict{Tuple{String, String}, Int})
    while length(symbols) > 1
        best_pair = nothing
        best_rank = typemax(Int)
        for i in 1:length(symbols)-1
            pair = (symbols[i], symbols[i+1])
            rank = get(merge_ranks, pair, typemax(Int))
            if rank < best_rank
                best_rank = rank
                best_pair = pair
            end
        end
        best_rank == typemax(Int) && break

        new_symbols = String[]
        i = 1
        while i <= length(symbols)
            if i < length(symbols) && symbols[i] == best_pair[1] && symbols[i+1] == best_pair[2]
                push!(new_symbols, best_pair[1] * best_pair[2])
                i += 2
            else
                push!(new_symbols, symbols[i])
                i += 1
            end
        end
        symbols = new_symbols
    end
    return symbols
end

function _build_byte_to_unicode()
    bs = UInt8[]
    append!(bs, UInt8('!'):UInt8('~'))
    append!(bs, UInt8('¡'):UInt8('¬'))
    append!(bs, UInt8('®'):UInt8('ÿ'))
    cs = Int[Int(b) for b in bs]
    n = 0
    for b in 0x00:0xff
        if !(b in bs)
            push!(bs, b)
            push!(cs, 256 + n)
            n += 1
        end
    end
    return Dict{UInt8, Char}(b => Char(c) for (b, c) in zip(bs, cs))
end

# ═══════════════════════════════════════════════════════════════════
# Unified tokenizer interface
# ═══════════════════════════════════════════════════════════════════

const Tokenizer = Union{CharTokenizer, BPETokenizer}

tokenizer_vocab_size(t::CharTokenizer) = t.vocab_size
tokenizer_vocab_size(t::BPETokenizer) = t.vocab_size

# ═══════════════════════════════════════════════════════════════════
# Causal mask (multiplicative 0/1 for Monarch/LongConv)
# ═══════════════════════════════════════════════════════════════════

function make_causal_mask(seq_len::Int)
    return Float32[j <= i ? 1.0f0 : 0.0f0 for i in 1:seq_len, j in 1:seq_len]
end

# ═══════════════════════════════════════════════════════════════════
# Layer primitives
# ═══════════════════════════════════════════════════════════════════

function rmsnorm_forward(x, weight; eps=1.0f-6)
    rms = sqrt.(mean(x .^ 2; dims=1) .+ eps)
    return weight .* (x ./ rms)
end

function swiglu_forward(x, ps)
    D = size(x, 1)
    x_flat = reshape(x, D, :)
    gate = ps.w1 * x_flat
    val  = ps.v  * x_flat
    hidden = NNlib.swish.(gate) .* val
    out = ps.w2 * hidden
    return reshape(out, D, size(x)[2:end]...)
end

# ═══════════════════════════════════════════════════════════════════
# Monarch Matrix realization (single-head for symbiogenesis)
# ═══════════════════════════════════════════════════════════════════

"""
    monarch_realize(L1, L2, p) -> Matrix{Float32}

Materialize T×T Monarch matrix: M = Pᵀ · BlockDiag(L1) · P · BlockDiag(L2)
where L1, L2 are (p, p, p) block-diagonal factors and T = p².
"""
function monarch_realize(L1, L2, p::Int)
    T = p * p

    # Start with identity matrix
    I_T = Float32[i == j ? 1.0f0 : 0.0f0 for i in 1:T, j in 1:T]

    # Reshape columns: (T, T) → (p, p, T)
    x = reshape(I_T, p, p, T)

    # Apply L2 block-diagonal: for each block k, multiply L2[:,:,k] @ x[:,k,:]
    x = permutedims(x, (1, 3, 2))         # (p, T, p)
    x = batched_mul(L2, x)                # (p, p, p) × (p, T, p) → (p, T, p)
    x = permutedims(x, (1, 3, 2))         # (p, p, T)

    # Permutation P: transpose the p×p grid
    x = permutedims(x, (2, 1, 3))

    # Apply L1 block-diagonal
    x = permutedims(x, (1, 3, 2))         # (p, T, p)
    x = batched_mul(L1, x)                # (p, T, p)
    x = permutedims(x, (1, 3, 2))         # (p, p, T)

    # Undo permutation
    x = permutedims(x, (2, 1, 3))

    return reshape(x, T, T)
end

# ═══════════════════════════════════════════════════════════════════
# Causal Depthwise Conv1d (short kernel, local n-gram patterns)
# ═══════════════════════════════════════════════════════════════════

"""
    causal_depthwise_conv1d(x, kernel) -> Array

x: (D, T, B), kernel: (K, D)
Causal convolution: pad K-1 zeros on the left, sum over kernel taps.
"""
function causal_depthwise_conv1d(x, kernel)
    D, T, B = size(x)
    K = size(kernel, 1)

    # Causal pad: K-1 zeros on the left
    pad = zeros(Float32, D, K - 1, B)
    x_padded = cat(pad, x; dims=2)  # (D, T+K-1, B)

    # Sum over kernel taps
    out = sum(1:K) do k
        reshape(kernel[k:k, :], D, 1, 1) .* x_padded[:, k:k+T-1, :]
    end

    return out
end

# ═══════════════════════════════════════════════════════════════════
# LongConv — full-length per-channel causal convolution
# ═══════════════════════════════════════════════════════════════════

"""
    long_conv_forward(x, kernel) -> Array

x: (D, T, B), kernel: (K, D) where K = context_length
Per-channel causal convolution with full-length kernel using NNlib.conv
with groups=D for depthwise operation.
"""
function long_conv_forward(x, kernel)
    D, T, B = size(x)
    K = size(kernel, 1)

    # NNlib.conv expects (spatial, channels, batch) for 1D
    # x is (D, T, B) → need (T, D, B)
    x_conv = permutedims(x, (2, 1, 3))  # (T, D, B)

    # Causal padding: K-1 zeros on the left of sequence dimension
    pad_arr = zeros(Float32, K - 1, D, B)
    x_padded = cat(pad_arr, x_conv; dims=1)  # (T + K - 1, D, B)

    # Reshape kernel for NNlib depthwise conv:
    # kernel (K, D) → (K, 1, D): (spatial, in_channels/groups, out_channels)
    w = reshape(kernel, K, 1, D)

    # Match element types
    if eltype(x_padded) != eltype(w)
        x_padded = eltype(w).(x_padded)
    end

    # Depthwise conv1d: groups = D means each channel has its own filter
    y_conv = NNlib.conv(x_padded, w; groups=D)  # (T, D, B)

    # Back to (D, T, B)
    return permutedims(y_conv, (2, 1, 3))
end

# ═══════════════════════════════════════════════════════════════════
# OrganelleGate — per-channel softmax fusion over 3 organelles
# ═══════════════════════════════════════════════════════════════════

"""
    organelle_gate_forward(organelle_outputs, logits) -> Array

organelle_outputs: tuple of N arrays, each (D, T, B)
logits: (N, D) — per-channel gating logits

Combines outputs via per-channel softmax: each channel independently
learns which organelle(s) to rely on.
"""
function organelle_gate_forward(organelle_outputs, logits)
    N, D = size(logits)

    # Per-channel softmax over organelles
    weights = NNlib.softmax(logits; dims=1)  # (N, D)

    # Weighted sum of organelle outputs
    out = sum(1:N) do i
        # weights[i, :] is (D,) — reshape to (D, 1, 1) for broadcast
        reshape(weights[i:i, :], D, 1, 1) .* organelle_outputs[i]
    end

    return out
end

# ═══════════════════════════════════════════════════════════════════
# Symbiogenesis Sequence Mixer — 3 organelles + gate
# ═══════════════════════════════════════════════════════════════════

function symbio_sequence_mixer_forward(x, ps, context_length::Int, mask)
    D, T, B = size(x)
    p = isqrt(context_length)

    # ── Organelle 1: CausalConv (local n-gram patterns) ──
    conv_out = causal_depthwise_conv1d(x, ps.conv.kernel)

    # ── Organelle 2: MonarchMatrix (global structured mixing, single-head) ──
    # Realize full T_max × T_max Monarch matrix
    M = monarch_realize(ps.monarch.L1, ps.monarch.L2, p)

    # Apply causal mask (multiplicative 0/1)
    M_causal = M .* mask

    # Slice to actual sequence length (handles generation where T < context_length)
    M_t = M_causal[1:T, 1:T]

    # Single-head: apply Monarch to ALL channels at once
    # x: (D, T, B) → permute to (T, D, B) → flatten → matmul → reshape back
    x_seq = reshape(permutedims(x, (2, 1, 3)), T, D * B)  # (T, D*B)
    y_monarch = M_t * x_seq                                 # (T, D*B)
    monarch_out = permutedims(reshape(y_monarch, T, D, B), (2, 1, 3))  # (D, T, B)

    # ── Organelle 3: LongConv (global dense causal filter) ──
    longconv_out = long_conv_forward(x, ps.longconv.kernel)

    # ── Gate: per-channel softmax fusion ──
    gated = organelle_gate_forward((conv_out, monarch_out, longconv_out), ps.gate.logits)

    return gated
end

# ═══════════════════════════════════════════════════════════════════
# Full model forward pass
# ═══════════════════════════════════════════════════════════════════

function model_forward(config::ModelConfig, ps, x)
    T = size(x, 1)  # x: (seq_len, batch) of integer token IDs

    # Token embedding: (seq_len, batch) → (embed_dim, seq_len, batch)
    h = ps.tok_emb.weight[:, x]

    # Causal mask (multiplicative 0/1 for symbiogenesis)
    mask = make_causal_mask(config.context_length)

    # Symbiogenesis blocks
    for i in 1:config.n_layers
        name = Symbol("block_$i")
        bp = getproperty(ps.blocks, name)

        # Pre-norm sequence mixing + residual
        normed = rmsnorm_forward(h, bp.ln1.weight)
        mixed = symbio_sequence_mixer_forward(normed, bp.seq_mixer,
                                               config.context_length, mask)
        h = h .+ mixed

        # Pre-norm FFN + residual
        normed2 = rmsnorm_forward(h, bp.ln2.weight)
        ffn_out = swiglu_forward(normed2, bp.ffn)
        h = h .+ ffn_out
    end

    # Final norm
    h = rmsnorm_forward(h, ps.ln_f.weight)

    # Output projection (weight-tied)
    D, T_out, B = size(h)
    h_flat = reshape(h, D, T_out * B)
    if config.weight_tying
        logits = ps.tok_emb.weight' * h_flat
    else
        logits = ps.head.weight * h_flat
    end
    return reshape(logits, :, T_out, B)
end

# ═══════════════════════════════════════════════════════════════════
# Sampling helpers
# ═══════════════════════════════════════════════════════════════════

function top_k_filter(logits::AbstractVector, k::Int)
    k = min(k, length(logits))
    sorted = sort(Array(logits); rev=true)
    threshold = sorted[k]
    return map(l -> l >= threshold ? l : typemin(eltype(logits)), logits)
end

function top_p_filter(logits::AbstractVector, p::Float64)
    sorted_indices = sortperm(Array(logits); rev=true)
    sorted_logits = logits[sorted_indices]
    probs = NNlib.softmax(sorted_logits)
    cumprobs = cumsum(Array(probs))
    cutoff = something(findfirst(>=(p), cumprobs), length(probs))
    result = fill(typemin(eltype(logits)), length(logits))
    for i in 1:cutoff
        result[sorted_indices[i]] = logits[sorted_indices[i]]
    end
    return result
end

function sample_categorical(probs::AbstractVector)
    r = rand(Float32)
    cumulative = 0.0f0
    for i in eachindex(probs)
        cumulative += probs[i]
        r <= cumulative && return i
    end
    return length(probs)
end

# ═══════════════════════════════════════════════════════════════════
# Text generation with streaming callback
# ═══════════════════════════════════════════════════════════════════

function generate_streaming(config::ModelConfig, ps,
                            tokenizer::Tokenizer, prompt::String;
                            max_tokens::Int=200,
                            temperature::Float64=0.8,
                            top_k::Int=0,
                            top_p::Float64=1.0,
                            on_token=nothing)
    tokens = encode(tokenizer, prompt)
    if isempty(tokens)
        tokens = [rand(1:tokenizer_vocab_size(tokenizer))]
    end

    generated = String[]

    for _ in 1:max_tokens
        ctx = if length(tokens) > config.context_length
            tokens[end-config.context_length+1:end]
        else
            copy(tokens)
        end

        x = reshape(ctx, :, 1)
        logits = model_forward(config, ps, x)
        next_logits = Vector{Float32}(logits[:, end, 1])

        if temperature != 1.0
            next_logits ./= Float32(temperature)
        end

        if top_k > 0
            next_logits = top_k_filter(next_logits, top_k)
        end

        if top_p < 1.0
            next_logits = top_p_filter(next_logits, top_p)
        end

        probs = NNlib.softmax(next_logits)
        next_token = sample_categorical(probs)

        push!(tokens, next_token)
        token_str = decode(tokenizer, [next_token])
        push!(generated, token_str)

        on_token !== nothing && on_token(token_str)
    end

    return join(generated)
end
