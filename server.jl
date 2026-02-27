#=
server.jl — OpenAI-compatible inference server for SymbioSLM

Serves a Lux.jl trained Symbiogenesis model (multi-organelle sequence mixing:
CausalConv + MonarchMatrix + LongConv, fused via OrganelleGate, with RMSNorm,
SwiGLU, weight-tied). Downloads artifacts from HuggingFace on first run.

Endpoints:
    GET  /                       -> health check / API info
    GET  /v1/models              -> list available models
    POST /v1/chat/completions    -> generate text (OpenAI format, streaming supported)
=#

include("checkpoint.jl")
using HTTP
using UUIDs
using Downloads

# ═══════════════════════════════════════════════════════════════════
# Download artifacts from HuggingFace
# ═══════════════════════════════════════════════════════════════════

const CKPT_DIR = "checkpoints"
const CKPT_PATH = joinpath(CKPT_DIR, "final.jld2")
const CONFIG_PATH = joinpath(CKPT_DIR, "config.toml")
const VOCAB_PATH = joinpath(CKPT_DIR, "vocab.json")
const MERGES_PATH = joinpath(CKPT_DIR, "merges.txt")
const HF_REPO = get(ENV, "HF_REPO", "LisaMegaWatts/SymbioSLM")
const PORT = parse(Int, get(ENV, "PORT", "7860"))

function download_from_hf(repo::String, filename::String, local_path::String)
    url = "https://huggingface.co/$repo/resolve/main/$filename"
    println("Downloading $url ...")
    mkpath(dirname(local_path))
    Downloads.download(url, local_path)
    sz = round(filesize(local_path) / 1024^2, digits=1)
    println("  -> $local_path ($sz MB)")
end

function ensure_artifacts()
    for (localpath, remote) in [(CKPT_PATH, "final.jld2"),
                                (CONFIG_PATH, "config.toml"),
                                (VOCAB_PATH, "vocab.json")]
        if !isfile(localpath)
            println("No local $remote found, downloading from $HF_REPO ...")
            try
                download_from_hf(HF_REPO, remote, localpath)
            catch e
                println("Download failed for $remote: $e")
                println("Place $remote at $localpath manually.")
                exit(1)
            end
        end
    end
    if !isfile(MERGES_PATH)
        println("Attempting to download merges.txt (optional, for BPE) ...")
        try
            download_from_hf(HF_REPO, "merges.txt", MERGES_PATH)
        catch e
            println("  merges.txt not found (will use char tokenizer if vocab is array format)")
        end
    end
end

# ═══════════════════════════════════════════════════════════════════
# Download and load model
# ═══════════════════════════════════════════════════════════════════

ensure_artifacts()

println("\nLoading model...")
const INF_MODEL = load_inference_model(CKPT_PATH, CONFIG_PATH, VOCAB_PATH, MERGES_PATH)
const CONFIG = INF_MODEL.config
const PS = INF_MODEL.ps
const TOKENIZER = INF_MODEL.tokenizer
const MODEL_CREATED_AT = Int(floor(time()))

println("\nModel ready: arch=$(CONFIG.arch), vocab=$(CONFIG.vocab_size), embd=$(CONFIG.embed_dim), " *
        "layers=$(CONFIG.n_layers), monarch_heads=$(CONFIG.n_monarch_heads), ctx=$(CONFIG.context_length)")

# ═══════════════════════════════════════════════════════════════════
# HTTP helpers
# ═══════════════════════════════════════════════════════════════════

const CORS_HEADERS = [
    "Access-Control-Allow-Origin" => "*",
    "Access-Control-Allow-Methods" => "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers" => "Content-Type, Authorization",
]

function json_response(status::Int, body; extra_headers=[])
    json_bytes = JSON3.write(body)
    headers = [
        "Content-Type" => "application/json",
        CORS_HEADERS...,
        extra_headers...
    ]
    return HTTP.Response(status, headers, json_bytes)
end

function cors_preflight()
    return HTTP.Response(204, CORS_HEADERS)
end

# ═══════════════════════════════════════════════════════════════════
# Extract prompt from OpenAI chat messages
# ═══════════════════════════════════════════════════════════════════

function extract_prompt(messages)
    if isempty(messages)
        return ""
    end
    for i in length(messages):-1:1
        role = string(get(messages[i], :role, ""))
        if role == "user"
            return string(get(messages[i], :content, ""))
        end
    end
    return string(get(messages[end], :content, ""))
end

# ═══════════════════════════════════════════════════════════════════
# SSE helpers
# ═══════════════════════════════════════════════════════════════════

function sse_line(data)
    return "data: $(JSON3.write(data))\n\n"
end

# ═══════════════════════════════════════════════════════════════════
# Request handler
# ═══════════════════════════════════════════════════════════════════

function handle_request(request::HTTP.Request)
    method = request.method
    target = request.target

    if method == "OPTIONS"
        return cors_preflight()
    end

    # GET / — health check
    if method == "GET" && target == "/"
        return json_response(200, Dict(
            "name" => "SymbioSLM",
            "version" => "1.0.0",
            "description" => "A Symbiogenesis model trained on classical philosophy texts — " *
                             "multi-organelle sequence mixing inspired by biological endosymbiosis",
            "architecture" => "Decoder-only (CausalConv + MonarchMatrix + LongConv + OrganelleGate, " *
                              "RMSNorm, SwiGLU, weight-tied)",
            "model" => Dict(
                "arch" => CONFIG.arch,
                "vocab_size" => CONFIG.vocab_size,
                "embed_dim" => CONFIG.embed_dim,
                "n_layers" => CONFIG.n_layers,
                "n_monarch_heads" => CONFIG.n_monarch_heads,
                "conv_kernel_size" => CONFIG.conv_kernel_size,
                "context_length" => CONFIG.context_length
            ),
            "organelles" => [
                "CausalConv (local n-gram patterns)",
                "MonarchMatrix (global structured mixing)",
                "LongConv (global dense causal filter)"
            ],
            "endpoints" => ["/v1/models", "/v1/chat/completions"],
            "features" => ["streaming", "OpenAI-compatible", "top-k", "top-p"],
            "compatible_with" => ["OpenAI API", "OpenRouter"]
        ))
    end

    # GET /v1/models
    if method == "GET" && target == "/v1/models"
        return json_response(200, Dict(
            "object" => "list",
            "data" => [Dict(
                "id" => "symbioslm-philosophy",
                "object" => "model",
                "created" => MODEL_CREATED_AT,
                "owned_by" => "symbioslm"
            )]
        ))
    end

    # POST /v1/chat/completions
    if method == "POST" && target == "/v1/chat/completions"
        local body
        try
            body = JSON3.read(String(request.body))
        catch e
            return json_response(400, Dict("error" => Dict(
                "message" => "Invalid JSON in request body",
                "type" => "invalid_request_error",
                "code" => "invalid_json")))
        end

        temperature = Float64(clamp(get(body, :temperature, 0.8), 0.01, 2.0))
        max_tokens = Int(clamp(get(body, :max_tokens, 200), 1, CONFIG.context_length))
        top_k_val = Int(clamp(get(body, :top_k, 40), 0, CONFIG.vocab_size))
        top_p_val = Float64(clamp(get(body, :top_p, 1.0), 0.0, 1.0))
        stream = Bool(get(body, :stream, false))

        messages = get(body, :messages, [])
        prompt_text = extract_prompt(messages)

        if stream
            completion_id = "chatcmpl-" * string(uuid4())
            created = Int(floor(time()))

            buf = IOBuffer()

            initial_chunk = Dict(
                "id" => completion_id,
                "object" => "chat.completion.chunk",
                "created" => created,
                "model" => "symbioslm-philosophy",
                "choices" => [Dict(
                    "index" => 0,
                    "delta" => Dict("role" => "assistant", "content" => ""),
                    "finish_reason" => nothing
                )]
            )
            write(buf, sse_line(initial_chunk))

            token_count = Ref(0)

            generate_streaming(CONFIG, PS, TOKENIZER, prompt_text;
                               max_tokens, temperature, top_k=top_k_val, top_p=top_p_val,
                               on_token = function(token_str)
                                   token_count[] += 1
                                   chunk = Dict(
                                       "id" => completion_id,
                                       "object" => "chat.completion.chunk",
                                       "created" => created,
                                       "model" => "symbioslm-philosophy",
                                       "choices" => [Dict(
                                           "index" => 0,
                                           "delta" => Dict("content" => token_str),
                                           "finish_reason" => nothing
                                       )]
                                   )
                                   write(buf, sse_line(chunk))
                               end)

            prompt_tokens = length(encode(TOKENIZER, prompt_text))
            finish_chunk = Dict(
                "id" => completion_id,
                "object" => "chat.completion.chunk",
                "created" => created,
                "model" => "symbioslm-philosophy",
                "choices" => [Dict(
                    "index" => 0,
                    "delta" => Dict(),
                    "finish_reason" => token_count[] >= max_tokens ? "length" : "stop"
                )],
                "usage" => Dict(
                    "prompt_tokens" => prompt_tokens,
                    "completion_tokens" => token_count[],
                    "total_tokens" => prompt_tokens + token_count[]
                )
            )
            write(buf, sse_line(finish_chunk))
            write(buf, "data: [DONE]\n\n")

            sse_body = take!(buf)
            headers = [
                "Content-Type" => "text/event-stream",
                "Cache-Control" => "no-cache",
                "X-Accel-Buffering" => "no",
                CORS_HEADERS...
            ]
            return HTTP.Response(200, headers, sse_body)

        else
            n_completions = Int(clamp(get(body, :n, 1), 1, 4))

            choices = []
            total_completion_tokens = 0
            for i in 1:n_completions
                text = generate_streaming(CONFIG, PS, TOKENIZER, prompt_text;
                                          max_tokens, temperature, top_k=top_k_val, top_p=top_p_val)
                finish_reason = length(text) >= max_tokens ? "length" : "stop"
                push!(choices, Dict(
                    "index" => i - 1,
                    "message" => Dict("role" => "assistant", "content" => text),
                    "finish_reason" => finish_reason))
                total_completion_tokens += length(text)
            end

            prompt_tokens = length(encode(TOKENIZER, prompt_text))
            return json_response(200, Dict(
                "id" => "chatcmpl-" * string(uuid4()),
                "object" => "chat.completion",
                "created" => Int(floor(time())),
                "model" => "symbioslm-philosophy",
                "choices" => choices,
                "usage" => Dict(
                    "prompt_tokens" => prompt_tokens,
                    "completion_tokens" => total_completion_tokens,
                    "total_tokens" => prompt_tokens + total_completion_tokens),
                "system_fingerprint" => "symbioslm-v1"))
        end
    end

    return json_response(404, Dict("error" => Dict(
        "message" => "Not found: $method $target",
        "type" => "invalid_request_error",
        "code" => "not_found")))
end

# ═══════════════════════════════════════════════════════════════════
# Start server
# ═══════════════════════════════════════════════════════════════════

println("\nSymbioSLM server starting on 0.0.0.0:$PORT ...")
println("  GET  http://localhost:$PORT/")
println("  GET  http://localhost:$PORT/v1/models")
println("  POST http://localhost:$PORT/v1/chat/completions")
println("  POST http://localhost:$PORT/v1/chat/completions  (stream=true)")
println()

HTTP.serve(handle_request, "0.0.0.0", PORT)
