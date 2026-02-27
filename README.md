---
title: SymbioSLM
emoji: "\U0001F9EC"
colorFrom: green
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - julia
  - lux
  - slm
  - philosophy
  - openai-compatible
  - bpe
  - symbiogenesis
  - monarch-mixer
  - long-convolution
  - rmsnorm
  - swiglu
---

# SymbioSLM

A Symbiogenesis decoder-only model trained on classical philosophy texts, implemented in Julia with Lux.jl. Features multi-organelle sequence mixing inspired by biological endosymbiosis (Margulis, 1967): three complementary "organelles" — CausalConv, MonarchMatrix, and LongConv — are fused via a learned per-channel OrganelleGate. Serves an OpenAI-compatible API with streaming support.

## Endpoints

- `GET /` — Health check and model info
- `GET /v1/models` — List available models
- `POST /v1/chat/completions` — Generate text (supports streaming, top-k, top-p)

## Usage

```bash
# Non-streaming
curl -X POST https://your-space.hf.space/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "the nature of"}], "max_tokens": 200}'

# Streaming
curl -X POST https://your-space.hf.space/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "the nature of"}], "stream": true, "temperature": 0.7, "top_k": 40}'
```

## Architecture

- **Model**: ~5M params, 256d embed, 8 layers
- **Sequence mixing**: 3 organelles per block fused via OrganelleGate:
  - CausalConv (local n-gram patterns, K=4)
  - MonarchMatrix (global structured mixing, single-head)
  - LongConv (global dense causal filter, K=256)
- **Gating**: Per-channel softmax over organelles (learned specialization)
- **Tokenizer**: BPE (2000 tokens)
- **Framework**: Lux.jl (explicit parameter/state management)
- **Normalization**: RMSNorm (pre-norm)
- **Feed-forward**: SwiGLU activation
- **Weight tying**: Shared embedding/output projection
- **Inference**: CPU-only, no Lux dependency at runtime (pure NNlib)

## Environment Variables

- `HF_REPO` — HuggingFace model repo (default: `LisaMegaWatts/SymbioSLM`)
- `PORT` — Server port (default: `7860`)
