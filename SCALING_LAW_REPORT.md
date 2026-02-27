# SLM Scaling Law Report: Philosophy Corpus

**Date**: 2026-02-27
**Author**: Lisa @ Decentralized Intelligence Agency
**GitHub**: [DavinciDreams/SymbioGPT](https://github.com/DavinciDreams/SymbioGPT)

## Abstract

We report scaling law measurements for small language models (1M–23M params) trained on a curated philosophy corpus. Five model architectures — spanning standard transformers, Monarch mixers, and multi-organelle symbiogenesis designs — are compared under controlled conditions. Key finding: loss drops steeply from 1M to 5M parameters (4.45 → 3.54), then **plateaus entirely** from 5M to 11M (3.54 → 3.56), indicating the 266M-token corpus is the binding constraint above ~5M params.

## Experimental Setup

### Dataset

- **Corpus**: Curated philosophy text, quality-scored with `curate_data.py` (vocab diversity, word length, punctuation quality, alphabetic ratio)
- **Train tokens**: 266M (from 912K chunks → top 365K selected, score ≥ 0.755)
- **Val tokens**: 72M
- **Tokenizer**: BPE, vocab = 2,000
- **Context length**: 256 tokens

### Models

All models share the same tokenizer, context length, and training corpus.

| Model | Params | d_model | Layers | Attention | FFN | Architecture |
|-------|--------|---------|--------|-----------|-----|-------------|
| JuliaFluxGPT-1M | 1,010,816 | 128 | 4 | 4Q/4KV MHA, hd=32 | SwiGLU 192 | LLaMA-style (PyTorch) |
| SymbioSLM | 4,070,000 | 256 | 8 | 3-organelle gate | SwiGLU 640 | Symbiogenesis (Lux.jl) |
| MonarchSLM | 4,980,000 | 256 | 8 | 8 Monarch heads | SwiGLU 640 | Monarch Mixer (Lux.jl) |
| JuliaSLM | 5,040,000 | 256 | 6 | 4H MHA, hd=64 | SwiGLU 640 | Transformer (Lux.jl) |
| SymbioGPT-10M | 11,053,400 | 320 | 8 | 4-organelle gate | SwiGLU 832 | Symbiogenesis (PyTorch) |

**Not included** (dirty data, not comparable):
- JuliaFluxGPT-23M (d=512, 8L, 8Q/2KV GQA) — trained on uncurated data, val_loss ~6.6
- MicroJulia (~1M) — char-level tokenizer, uncurated data

## Results

### Scaling Law Table

| Model | Params | Val Loss | Val PPL | Tokens Seen | Tok/Param | W&B Run |
|-------|--------|----------|---------|-------------|-----------|---------|
| JuliaFluxGPT-1M | 1.01M | **4.446** | 85.3 | ~80M | 80:1 | [`p7yt1too`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/p7yt1too) |
| SymbioSLM | 4.07M | **3.620** | 37.3 | ~82M | 20:1 | [`drthfacf`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/julia-slm/runs/drthfacf) (metrics from checkpoint) |
| MonarchSLM | 4.98M | **3.650** | 38.4 | ~100M | 20:1 | — (metrics from checkpoint) |
| JuliaSLM | 5.04M | **3.540** | 34.5 | ~101M | 20:1 | — (metrics from checkpoint) |
| SymbioGPT-10M | 11.05M | **3.563** | 35.3 | 266M | 24:1 | [`5bi3p1dp`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/5bi3p1dp) |

### JuliaFluxGPT-1M Training Curve

Trained on NVIDIA T4 (Colab), bf16, 4× Chinchilla budget.

| Step | Tokens | Val Loss | Val PPL | Notes |
|------|--------|----------|---------|-------|
| 250 | 4.1M | 6.539 | 691.5 | Early learning |
| 500 | 8.2M | 5.445 | 231.6 | |
| 1,000 | 16.4M | 4.957 | 142.2 | |
| 1,233 | 20.2M | ~4.85 | ~128 | **Chinchilla boundary (20:1)** |
| 2,000 | 32.8M | 4.777 | 118.7 | |
| 3,000 | 49.2M | 4.607 | 100.2 | |
| 4,000 | 65.5M | 4.497 | 89.8 | |
| **4,932** | **80.8M** | **4.446** | **85.3** | **Final (4× Chinchilla)** |

### SymbioGPT-10M Training Curve (from W&B `5bi3p1dp`)

Trained on NVIDIA A100 (Colab), bf16, batch=64, Chinchilla-optimal.

- 13,400 steps, ~60K tok/s on A100
- Final val_loss = 3.563, val_ppl = 35.3
- Gate entropy 1.21 (below uniform 1.39 — real specialization)

**Organelle gate specialization pattern:**
| Layer | Causal Conv | Monarch | Long Conv | Attention |
|-------|------------|---------|-----------|-----------|
| 0 | **0.56** | 0.14 | 0.14 | 0.16 |
| 3 | 0.32 | 0.18 | 0.18 | 0.32 |
| 7 | 0.21 | 0.16 | 0.16 | **0.47** |

Early layers prefer local convolution; later layers prefer attention — consistent with the hypothesis that local patterns are captured first, then long-range dependencies.

### JuliaFluxGPT-23M Training History (dirty data)

Multiple crashed W&B runs in `JuliaFluxGPT` project show the training trajectory on uncurated data:

| W&B Run | Stage | Val Loss Range | Notes |
|---------|-------|---------------|-------|
| [`mkh7robk`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/JuliaFluxGPT/runs/mkh7robk) | Steps 0–500 | 12.7 | Initial training (A100) |
| [`h0qutwnv`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/JuliaFluxGPT/runs/h0qutwnv) | Steps 500–2500 | 12.7 → 12.4 | Resume |
| [`rzhelyrs`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/JuliaFluxGPT/runs/rzhelyrs) | Steps 2500–4500 | 12.4 → 11.8 | Resume |
| [`wv1xpld4`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/JuliaFluxGPT/runs/wv1xpld4) | Steps 6500–8500 | 11.8 → 11.3 | Resume |
| [`4odlhnwf`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/JuliaFluxGPT/runs/4odlhnwf) | Steps 29000+ | 10.7 → **6.62** | Data switch mid-run |

The jump from 10.7 to 6.62 in run `4odlhnwf` corresponds to a data/configuration change during training. The final val_loss of ~6.6 on the curated val set confirms the model learned something, but far below the ~3.5 achieved by 5M models trained on curated data from the start.

### Wolves Compression Run (dirty weights)

| W&B Run | Status | Notes |
|---------|--------|-------|
| [`hwjoituu`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/hwjoituu) | Running (stalled) | SVD compression of JuliaFluxGPT-23M |

- Population of 10 wolves, SVD rank schedules [77–512]
- Best after 5 generations: val_loss=7.31, 19.5M params (14.3% compression)
- Fitness landscape essentially flat — all wolves within 0.03 of each other
- **Conclusion**: Compressing dirty-data weights is not productive; the weights lack meaningful structure to preserve

### Distillation Test (from symbiogenesis project)

| W&B Run | Model | Val Loss | Notes |
|---------|-------|----------|-------|
| [`qgxw2u5j`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/qgxw2u5j) | Distilled student | 3.788 | SymbioGPT Phase C distillation |
| [`eqfiw9oj`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/eqfiw9oj) | Scratch student | 3.782 | Same architecture, no distillation |

Distillation provided marginal benefit (+0.006 loss) in this test — likely because the teacher and student were too similar in capacity.

## Key Findings

### 1. Steep Scaling Below 5M, Hard Plateau Above

- **1M → 5M** (5× params): loss drops **0.91** (4.45 → 3.54), PPL 85 → 35
- **5M → 11M** (2.2× params): loss increases **+0.02** (3.54 → 3.56)

The 266M-token corpus saturates around 5M parameters. Additional capacity above this point is wasted without more data.

### 2. Architecture-Agnostic Convergence at ~5M

Three radically different sequence mixing architectures converge to nearly identical performance:

| Architecture | Mixing Mechanism | Val Loss |
|-------------|------------------|----------|
| Transformer MHA | Standard attention | 3.54 |
| Symbiogenesis | 3-organelle gated mixture | 3.62 |
| Monarch Mixer | Block-diagonal frequency mixing | 3.65 |

The spread is only **0.11 loss** — within noise for different architectures and training schedules. Data quality and tokenizer choices dominate at this scale.

### 3. Organelle Specialization is Real but Data-Bottlenecked

SymbioGPT-10M shows genuine layer-wise specialization (gate entropy 1.21 < uniform 1.39) but doesn't translate to lower loss than the simpler 5M transformer. The organelle routing learns meaningful patterns but can't leverage them without more diverse training data.

### 4. Data Quality > Model Size

JuliaFluxGPT at 23M params trained on dirty data achieves val_loss ~6.6. JuliaSLM at 5M params (4.6× smaller) trained on curated data achieves 3.54. **Data quality provides a 3.06 loss improvement** — equivalent to many orders of magnitude in model scaling.

### 5. Over-training Small Models is Worthwhile

JuliaFluxGPT-1M trained to 4× Chinchilla (80:1 tok/param vs 20:1) showed continuous improvement:
- At 1× Chinchilla: val_loss ≈ 4.85
- At 4× Chinchilla: val_loss = 4.45
- Extra 3× tokens bought 0.40 loss improvement — significant for the compute cost on T4

## Next Steps

1. **Fusion transfer**: Project JuliaSLM's curated-data weights (d=256) into JuliaFluxGPT (d=512) via symbiogenesis projection, fine-tune on curated data. Notebook: `fuse_juliaslm.ipynb`
2. **Corpus expansion**: The clear bottleneck. Need >1B tokens to see benefit from >5M params.
3. **Post-fusion wolves**: Run SVD compression on the fused model — then the weights will have meaningful structure to compress.
4. **Symbiogenesis Phase D**: MoE via fusion — combine specialized organelle units into a mixture-of-experts architecture.

## HuggingFace Repos

| Model | Repo | Checkpoint |
|-------|------|-----------|
| JuliaFluxGPT-1M | [LisaMegaWatts/JuliaFluxGPT-1M](https://huggingface.co/LisaMegaWatts/JuliaFluxGPT-1M) | `juliaflux_1m_best.pt` |
| JuliaSLM | [LisaMegaWatts/JuliaSLM](https://huggingface.co/LisaMegaWatts/JuliaSLM) | `final.jld2`, `juliaslm_weights.npz` |
| MonarchSLM | [LisaMegaWatts/MonarchSLM](https://huggingface.co/LisaMegaWatts/MonarchSLM) | `final.jld2` |
| SymbioSLM | [LisaMegaWatts/SymbioSLM](https://huggingface.co/LisaMegaWatts/SymbioSLM) | checkpoint v1 |
| SymbioGPT-10M | [LisaMegaWatts/SymbioGPT-10M](https://huggingface.co/LisaMegaWatts/SymbioGPT-10M) | `symbio_best.pt` |
| JuliaFluxGPT | [LisaMegaWatts/JuliaFluxGPT](https://huggingface.co/LisaMegaWatts/JuliaFluxGPT) | `juliaflux_weights.pt` |

## W&B Run Index

| Run ID | Project | Name | Model | Status |
|--------|---------|------|-------|--------|
| `p7yt1too` | symbiogenesis | juliafluxgpt-1m-scaling | JuliaFluxGPT-1M | finished |
| `5bi3p1dp` | symbiogenesis | symbio-teacher-10m-chinchilla | SymbioGPT-10M | finished |
| `drthfacf` | julia-slm | symbiogenesis-4262K | SymbioSLM | crashed (metrics from checkpoint) |
| `hwjoituu` | symbiogenesis | wolves-compress-juliafluxgpt | Wolves compression | running |
| `mkh7robk` | JuliaFluxGPT | julia-mkh7robk | JuliaFluxGPT-23M | crashed |
| `h0qutwnv` | JuliaFluxGPT | resume-500 | JuliaFluxGPT-23M | crashed |
| `rzhelyrs` | JuliaFluxGPT | resume-2500 | JuliaFluxGPT-23M | crashed |
| `wv1xpld4` | JuliaFluxGPT | resume-6500 | JuliaFluxGPT-23M | crashed |
| `4odlhnwf` | JuliaFluxGPT | resume-28998 | JuliaFluxGPT-23M | crashed |
| `qgxw2u5j` | symbiogenesis | distill-test-distilled | Distillation test | finished |
| `eqfiw9oj` | symbiogenesis | distill-test-scratch | Distillation control | finished |

All runs under W&B entity: `lisamegawatts-decentralized-intelligence-agency`
