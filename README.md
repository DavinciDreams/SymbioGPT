---
title: SymbioGPT
emoji: "\U0001F9EC"
colorFrom: green
colorTo: purple
license: mit
tags:
  - pytorch
  - slm
  - philosophy
  - bpe
  - symbiogenesis
  - monarch-mixer
  - long-convolution
  - causal-attention
  - rmsnorm
  - swiglu
  - organelle-gate
---

# SymbioGPT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DavinciDreams/SymbioGPT/blob/main/train_colab.ipynb)

A multi-organelle language model that learns to blend four sequence-mixing primitives via per-channel gating. Inspired by biological endosymbiosis (Margulis, 1967): just as mitochondria and chloroplasts were once independent organisms absorbed into eukaryotic cells, SymbioGPT fuses four complementary "organelles" and lets a learned gate decide which each channel needs.

Trained on curated classical philosophy texts with Chinchilla-optimal scaling.

## Architecture

```
Input → Embedding → SymbioBlock × 8 → RMSNorm → Output (weight-tied)

Each SymbioBlock:
  RMSNorm → SymbioSequenceMixer → SkipGate → +residual
              ├─ CausalDepthwiseConv1d  (local, O(n))
              ├─ MonarchMatrix           (global structured, O(n√n))
              ├─ LongConv                (global dense, O(n))
              ├─ CausalSelfAttention     (global, O(n²), RoPE)
              └─ OrganelleGate           (4-way per-channel softmax)
  RMSNorm → SwiGLU → SkipGate → +residual
```

### 10M Config

| Parameter | Value |
|-----------|-------|
| d_model | 320 |
| n_layers | 8 |
| n_heads | 5 (attention organelle) |
| head_dim | 64 |
| ffn_mult | 4 |
| context_length | 256 |
| vocab_size | 2000 (BPE) |
| conv_kernel_size | 4 |
| weight_tying | true |
| free_energy_beta | 0.001 |
| **Total params** | **~11M** |

## Training

Chinchilla-optimal: ~220M tokens for 11M params (20x ratio), single pass over curated data.

```bash
# Local (RTX 3060)
python train_teacher.py --steps 27000 --batch_size 32 --precision fp16 --compile \
  --train_path data/train_curated.txt

# Colab (A100) — use the notebook
```

**Data curation**: `curate_data.py` scores 912K chunks from the corpus on vocabulary diversity, sentence structure, repetition, and content quality, then selects the top 40% (266M tokens).

### Training features

- AMP (FP16/BF16) mixed precision
- `torch.compile` for fused kernels
- Gradient accumulation
- Cosine LR schedule with warmup
- Free energy regularization (complexity penalty on weight magnitudes)
- W&B logging + HuggingFace checkpoint upload

## Files

| File | Description |
|------|-------------|
| `symbio_model.py` | All organelle modules + SymbioGPT model |
| `train_teacher.py` | Standalone training script |
| `train_colab.ipynb` | Colab notebook (A100, self-contained) |
| `curate_data.py` | Data quality scoring + curation pipeline |
| `test_symbio.py` | Verification tests (shapes, causality, gradients) |
| `model.jl` | Original Julia SymbioSLM (3 organelles, inference) |
| `server.jl` | Julia OpenAI-compatible API server |

## Links

- **GitHub**: [DavinciDreams/SymbioGPT](https://github.com/DavinciDreams/SymbioGPT)
- **HuggingFace**: [LisaMegaWatts/SymbioGPT-10M](https://huggingface.co/LisaMegaWatts/SymbioGPT-10M)
- **W&B**: Project `symbiogenesis`, run `symbio-teacher-10m-chinchilla`
- **Symbiogenesis framework**: [DavinciDreams/symbiogenesis](https://github.com/DavinciDreams/symbiogenesis)
- **Original Julia model**: [LisaMegaWatts/SymbioSLM](https://huggingface.co/LisaMegaWatts/SymbioSLM)
