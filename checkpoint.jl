#=
checkpoint.jl — Load Lux-trained SymbioSLM checkpoint for inference

Loads model parameters from JLD2, config from TOML, and tokenizer from JSON + merges.
Converts Float16 parameters to Float32 for efficient CPU inference.
No RoPE caches needed — symbiogenesis uses learned position mixing via
Monarch matrices and long convolutions.
=#

include("model.jl")
using JLD2

# ═══════════════════════════════════════════════════════════════════
# Float32 conversion for CPU inference
# ═══════════════════════════════════════════════════════════════════

ensure_f32(x::AbstractArray{Float16}) = Float32.(x)
ensure_f32(x::AbstractArray) = x
ensure_f32(x::NamedTuple) = NamedTuple{keys(x)}(map(ensure_f32, values(x)))
ensure_f32(x::Tuple) = map(ensure_f32, x)
ensure_f32(x) = x

# ═══════════════════════════════════════════════════════════════════
# Tokenizer loading — auto-detect BPE vs char based on file format
# ═══════════════════════════════════════════════════════════════════

function load_tokenizer(vocab_path::String, merges_path::String)
    if isfile(merges_path)
        println("Loading BPE tokenizer from $vocab_path + $merges_path ...")
        tok = load_bpe_tokenizer(vocab_path, merges_path)
        println("  BPE vocab_size = $(tok.vocab_size), merges = $(length(tok.merges))")
        return tok
    end

    raw_text = read(vocab_path, String)
    parsed = JSON3.read(raw_text)
    if parsed isa AbstractDict
        println("Loading BPE tokenizer from $vocab_path (no merges file) ...")
        tok = load_bpe_tokenizer_no_merges(vocab_path)
        println("  BPE vocab_size = $(tok.vocab_size) (no merges)")
        return tok
    end

    println("Loading character tokenizer from $vocab_path ...")
    tok = load_char_vocab_json(vocab_path)
    println("  char vocab_size = $(tok.vocab_size)")
    return tok
end

function load_bpe_tokenizer_no_merges(vocab_path::String)
    encoder = JSON3.read(read(vocab_path, String), Dict{String, Int})
    decoder = Dict{Int, String}(v => k for (k, v) in encoder)
    b2u = _build_byte_to_unicode()
    u2b = Dict{Char, UInt8}(v => k for (k, v) in b2u)
    pat = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    return BPETokenizer(encoder, decoder, Tuple{String,String}[],
                        Dict{Tuple{String,String},Int}(), b2u, u2b,
                        length(encoder), pat)
end

# ═══════════════════════════════════════════════════════════════════
# Load everything needed for inference
# ═══════════════════════════════════════════════════════════════════

function load_inference_model(ckpt_path::String, config_path::String,
                              vocab_path::String, merges_path::String)
    # Tokenizer (determines vocab_size)
    tokenizer = load_tokenizer(vocab_path, merges_path)
    vs = tokenizer_vocab_size(tokenizer)

    # Config (with dynamically-set vocab_size from tokenizer)
    println("Loading config from $config_path ...")
    config = load_config_toml(config_path; vocab_size=vs)
    println("  arch=$(config.arch), embed_dim=$(config.embed_dim), layers=$(config.n_layers)")
    println("  monarch_heads=$(config.n_monarch_heads), conv_kernel=$(config.conv_kernel_size)")
    println("  context_length=$(config.context_length), weight_tying=$(config.weight_tying)")

    # Parameters
    println("Loading parameters from $ckpt_path ...")
    ps = ensure_f32(JLD2.load(ckpt_path, "parameters"))

    step = try JLD2.load(ckpt_path, "step") catch; 0 end
    val_loss = try JLD2.load(ckpt_path, "best_val_loss") catch; Inf end
    println("  step=$step, best_val_loss=$(round(val_loss; digits=4))")

    # Verify embedding dimensions match
    emb_shape = size(ps.tok_emb.weight)
    println("  embedding weight: $(emb_shape) (expect $(config.embed_dim) x $(config.vocab_size))")
    if emb_shape[2] != config.vocab_size
        @warn "Vocab size mismatch!" config_vocab=config.vocab_size embedding_vocab=emb_shape[2]
        config = ModelConfig(config.arch, config.embed_dim, config.n_layers,
                             config.n_monarch_heads, config.conv_kernel_size,
                             config.context_length, emb_shape[2],
                             config.weight_tying, config.bias)
        println("  Adjusted vocab_size to $(config.vocab_size) from embedding weight")
    end

    return (; config, ps, tokenizer, step, val_loss)
end
