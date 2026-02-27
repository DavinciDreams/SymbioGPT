---
language:
  - en
license: mit
library_name: lux
tags:
  - julia
  - lux
  - slm
  - philosophy
  - symbiogenesis
  - monarch-mixer
  - long-convolution
  - causal-conv
  - rmsnorm
  - swiglu
  - bpe
  - text-generation
pipeline_tag: text-generation
model-index:
  - name: SymbioSLM
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          type: LisaMegaWatts/philosophy-corpus
          name: philosophy-corpus
        metrics:
          - type: perplexity
            value: 79.9
            name: Val PPL (step 1000)
---

# SymbioSLM

A ~5M parameter decoder-only language model using the **Symbiogenesis** architecture — a novel multi-organelle sequence mixing design inspired by biological endosymbiosis (Margulis, 1967). Implemented entirely in Julia using Lux.jl and trained on classical philosophy texts.

## Architecture

Symbiogenesis replaces softmax attention with three complementary "organelles" per block, fused via a learned per-channel gate:

```
SymbioBlock (x6)
+-- RMSNorm
+-- SymbioSequenceMixer
|   +-- Organelle 1: CausalDepthwiseConv1d   (local n-gram patterns, K=4)
|   +-- Organelle 2: Multi-head MonarchMatrix (global sub-quadratic mixing)
|   +-- Organelle 3: LongConv                (global dense causal filter)
|   +-- OrganelleGate                        (per-channel softmax fusion)
+-- RMSNorm
+-- SwiGLU FFN
```

### How It Works

1. **CausalConv** captures local bigram/trigram/4-gram patterns via depthwise convolution (1 kernel per channel, length 4).

2. **Monarch matrices** provide global sequence mixing through factored M = P^T * BlockDiag(L1) * P * BlockDiag(L2), achieving 87.5% parameter reduction vs dense mixing (8,192 vs 65,536 params per head at T=256).

3. **LongConv** learns a full-length (T=256) causal filter per channel, enabling arbitrary position-dependent mixing.

4. **OrganelleGate** fuses all three via per-channel softmax: each of the 256 embedding channels independently learns which organelle to rely on.

No positional encoding (RoPE) is needed — the Monarch matrices and LongConv kernels implicitly learn position-dependent patterns.

## Model Details

| Parameter | Value |
|---|---|
| Architecture | Symbiogenesis (3 organelles + gate) |
| Parameters | ~4.1M |
| Embed dim | 256 |
| Layers | 6 |
| Monarch heads | 4 |
| Context length | 256 tokens |
| Vocabulary | 2,000 (ByteLevel BPE) |
| FFN | SwiGLU (hidden=640) |
| Normalization | RMSNorm (pre-norm) |
| Weight tying | Yes (shared input/output embeddings) |
| Precision | Float32 (F16 slower for Monarch block sizes) |

### Parameter Breakdown

| Component | Params | % |
|---|---|---|
| Token embedding (tied) | 512K | 12.6% |
| CausalConv (x6) | 6.1K | 0.2% |
| Monarch heads (x6, 4 heads each) | 197K | 4.8% |
| LongConv (x6) | 393K | 9.7% |
| OrganelleGate (x6) | 4.6K | 0.1% |
| SwiGLU FFN (x6) | 2.95M | 72.6% |
| RMSNorm (x13) | 3.3K | <0.1% |
| **Total** | **~4.1M** | |

### Sequence Mixing Efficiency

| | Transformer | Monarch | Symbiogenesis |
|---|---|---|---|
| Seq mixer params/block | 262K | 67K | 100K |
| Reduction vs Transformer | - | 74% | **62%** |
| Position encoding | RoPE (separate) | None | None |

## Architecture Evolution

The Symbiogenesis architecture went through two major revisions:

| | v1 | v2 (current) |
|---|---|---|
| Organelle 2 | CausalSelfAttention (4 heads) | Multi-head MonarchMatrix (4 heads) |
| Params | 5.44M | **4.07M** (25% reduction) |
| Throughput (F32) | 4,800 tok/s | **19,169 tok/s** (4x improvement) |
| Position encoding | None (attention is input-dependent) | None (Monarch learns fixed mixing) |

**Why the change:** Attention was the dominant cost center and made the model slower than the pure Transformer baseline. Replacing it with Monarch matrices gave 4x throughput improvement while reducing parameters by 25%.

## Training

| | Value |
|---|---|
| Dataset | [philosophy-corpus](https://huggingface.co/datasets/LisaMegaWatts/philosophy-corpus) |
| Corpus | 981 classical texts (Aristotle, Plato, Euclid, Descartes, Kant, Nietzsche, ...) |
| Train tokens | ~100M (Chinchilla-optimal: 20 tok/param) |
| Optimizer | AdamW (lr=1e-3, min_lr=1e-4, cosine decay) |
| Warmup | 500 steps (linear) |
| Max steps | 12,305 |
| Batch size | 32 |
| Gradient clipping | 1.0 (global norm) |
| Precision | Float32 (see AMP findings below) |
| Hardware | NVIDIA RTX 3060 12GB |
| Throughput | ~19K tok/s (Float32) |
| Projected training time | ~88 min (F32 at 19K tok/s) |
| Framework | Julia + Lux.jl + Zygote.jl + CUDA.jl |

### Training Curves (partial — v1 architecture)

Early training was conducted on the v1 (attention-based) architecture before the Monarch rewrite. These curves demonstrate the overall learning dynamics of the 3-organelle fusion:

| Step | Train Loss | Val Loss | Val PPL | Gate Entropy | Grad Norm | LR |
|---|---|---|---|---|---|---|
| 1 | 17.10 | 17.03 | 24.9M | 1.099 | 14.26 | 2e-6 |
| 50 | 9.24 | — | — | — | — | 1e-4 |
| 100 | 7.03 | — | — | — | — | 2e-4 |
| 200 | 5.90 | — | — | — | — | 4e-4 |
| 300 | 5.48 | — | — | — | — | 6e-4 |
| 500 | 6.50 | 4.92 | 137.5 | 1.098 | 0.79 | 1e-3 |
| 700 | 4.48 | — | — | — | — | 1e-3 |
| 1,000 | 4.43 | 4.38 | 79.9 | 1.094 | 0.54 | 1e-3 |
| 1,200 | 4.16 | — | — | — | — | 9.9e-4 |
| 1,400 | 4.12 | — | — | — | — | 9.9e-4 |

**Loss trajectory:** The model drops from random (PPL 24.9M) to PPL 137 within 500 steps, then continues improving to PPL 79.9 at step 1000. Train loss was still decreasing at step 1400 (4.12) when the run was stopped for architectural revision.

**Gate entropy:** Started at 1.099 (≈ ln(3) = 1.099, uniform across 3 organelles) and showed minimal specialization (1.094 at step 1000) during this early phase. Full organelle differentiation is expected to emerge later in training.

### Gelation Monitoring

Training includes phase transition detection inspired by polymer physics:

- **CUSUM on loss curvature**: Detects sudden changes in 2nd derivative of loss curve, signaling phase transitions where the model rapidly improves
- **Gate entropy**: Tracks organelle specialization (1.099 = uniform distribution across 3 organelles, 0 = fully specialized to one organelle)
- **Kuramoto order parameter**: Measures synchronization of block dynamics across layers (R > 0.9 = gelation — all blocks entering coherent oscillation)

Configuration: CUSUM threshold = 5.0, window size = 10 steps.

## Benchmarking

### Throughput Comparison

All benchmarks on NVIDIA RTX 3060 12GB, batch size 32, context length 256:

| Architecture | Params | F32 (tok/s) | F16 AMP (tok/s) | F16 Speedup |
|---|---|---|---|---|
| Transformer (JuliaSLM) | 5.04M | 26,781 | 30,110 | **1.12x** |
| **Symbiogenesis v2 (Monarch)** | **4.07M** | **19,169** | **16,007** | **0.84x** |
| Symbiogenesis v2 (Attention) | 5.44M | 13,713 | 14,543 | 1.06x |
| Symbiogenesis v1 (Attention) | 5.44M | 4,809 | 4,871 | 1.01x |

### Throughput Analysis

- **v1 → v2 (Attention):** 4.8K → 13.7K tok/s (2.9x) from fixing AMP type promotion bugs that caused silent Float32 upcasts
- **v2 (Attention) → v2 (Monarch):** 13.7K → 19.2K tok/s (1.4x) from replacing attention with Monarch matrices
- **Overall v1 → v2 (Monarch):** 4.8K → 19.2K tok/s (4x) total improvement
- **vs Transformer:** 72% of Transformer throughput at 81% of the parameters

### AMP (Float16) Findings

A key finding: **Float16 AMP is slower than Float32 for Monarch-based Symbiogenesis** (0.84x). Root cause:

1. **Monarch matrix realization** involves `permutedims` and `NNlib.batched_mul` on small (16×16×16) tensors — these are memory-bound, not compute-bound, so F16's compute advantage doesn't help
2. **Julia/CUDA.jl type promotion**: Zygote's AD can silently promote F16 intermediates back to F32 during the backward pass, eliminating any F16 savings while adding cast overhead
3. **Transformer benefits from F16** because attention's `batched_mul` on (T×T×H×B) matrices is compute-bound — exactly where F16 tensor cores excel

**Decision:** Train SymbioSLM in Float32 (19K tok/s) rather than Float16 (16K tok/s). The Transformer baseline is trained in Float16 AMP (30K tok/s) where the 1.12x speedup is worthwhile.

## Comparison with Other Julia SLM Variants

| | [JuliaSLM](https://huggingface.co/LisaMegaWatts/JuliaSLM) | [MonarchSLM](https://huggingface.co/LisaMegaWatts/MonarchSLM) | **SymbioSLM** |
|---|---|---|---|
| Architecture | Transformer | Monarch Mixer | Symbiogenesis |
| Sequence mixing | 4-head attention | 8-head Monarch + conv | 3 organelles + gate |
| Parameters | 5.04M | 4.98M | ~4.1M |
| Layers | 6 | 8 | 6 |
| Val PPL | **34.5** | 38.4 | TBD (in training) |
| Throughput (F32) | 26.8K tok/s | 19K tok/s | 19.2K tok/s |
| Throughput (F16) | **30.1K tok/s** | — | 16.0K tok/s |
| Position encoding | RoPE | None | None |
| Weight tying | Yes | Yes | Yes |

### Why SymbioSLM Has Fewer Parameters

SymbioSLM's 3-organelle mixer uses ~100K params/block vs 262K for Transformer attention. The savings come from:
- CausalConv: ~1K params (depthwise, kernel=4)
- Monarch heads (4×): ~33K params (factored matrices)
- LongConv: ~65K params (T=256 filter per channel)
- Gate: ~0.8K params (3 weights per channel)

This is a 62% reduction in sequence mixing parameters, though not enough to justify extra layers (unlike MonarchSLM's 74% reduction which enables 8 layers).

## Usage

### Generate with Julia

```julia
using Pkg; Pkg.activate("julia-slm")
include("src/JuliaGPT.jl")
using .JuliaGPT
using .JuliaGPT: Lux, CUDA

tok = BPETokenizer("vocab.json", "merges.txt")
device = Lux.gpu_device()
ps, st, _, step, val_loss = load_checkpoint("final.jld2"; device)

model = create_model(ModelConfig(;
    arch="symbiogenesis", vocab_size=vocab_size(tok),
    embed_dim=256, n_layers=6, n_heads=4, head_dim=64,
    n_monarch_heads=4, conv_kernel_size=4,
    ffn_mult=4, context_length=256, weight_tying=true,
))

text = generate(model, ps, st, tok, "the nature of ";
    max_new_tokens=200, temperature=0.8, top_k=40)
println(text)
```

### OpenAI-Compatible API

The model is served via [SymbioSLM Space](https://huggingface.co/spaces/LisaMegaWatts/SymbioSLM):

```bash
curl -X POST https://lisamegawatts-symbioslm.hf.space/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "the nature of"}],
    "max_tokens": 200,
    "temperature": 0.8,
    "top_k": 40
  }'
```

Streaming supported with `"stream": true`.

## Files

| File | Description |
|---|---|
| `final.jld2` | Trained model parameters (JLD2 format) |
| `config.toml` | Model architecture configuration |
| `vocab.json` | BPE vocabulary (2000 tokens) |
| `merges.txt` | BPE merge rules |

## Biological Inspiration

The architecture is named after Lynn Margulis' theory of **symbiogenesis** (1967): the proposal that eukaryotic cells originated through the endosymbiotic fusion of distinct prokaryotic organisms. Mitochondria and chloroplasts retain their own DNA, demonstrating their origin as once-independent organisms that became specialized organelles within a larger cell.

Similarly, each SymbioBlock contains three "organelles" with different mathematical properties (local convolution, global structured mixing, global dense filtering) that are fused into a single functional unit through the learned OrganelleGate. The gate entropy tracks how strongly the network differentiates between organelles — analogous to the degree of specialization achieved through evolutionary integration.

## Citation

```bibtex
@misc{symbioslm2026,
  title={Symbiogenesis: Multi-Organelle Sequence Mixing for Small Language Models},
  author={LisaMegaWatts},
  year={2026},
  url={https://huggingface.co/LisaMegaWatts/SymbioSLM}
}
```

## References

- Margulis, L. (1967). On the origin of mitosing cells. *J. Theoretical Biology*, 14(3), 225-274.
- Dao, T., et al. (2023). Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture. *NeurIPS 2023*.
- Poli, M., et al. (2023). Hyena Hierarchy: Towards Larger Convolutional Language Models. *ICML 2023*.
- Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.

## License

MIT
