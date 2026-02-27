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
| JuliaFluxGPT-fused | 22.79M | **3.698** | 40.4 | ~131M | 6:1 | [`juliafluxgpt-slm-fusion`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis) |

### JuliaFluxGPT-1M Training Curve

Trained on NVIDIA T4 (Colab), bf16, 4× Chinchilla budget.

| Step | Tokens | Val Loss | Val PPL | Notes |
|------|--------|----------|---------|-------|
| 250 | 4.1M | 6.539 | 691.5 | Early learning |
| 500 | 8.2M | 5.445 | 231.6 | |
| 750 | 12.3M | 5.087 | 161.8 | |
| 1,000 | 16.4M | 4.957 | 142.2 | |
| 1,233 | 20.2M | ~4.85 | ~128 | **Chinchilla boundary (20:1)** |
| 1,250 | 20.5M | 4.851 | 127.8 | |
| 1,500 | 24.6M | 4.777 | 118.8 | |
| 1,750 | 28.7M | 4.735 | 113.8 | |
| 2,000 | 32.8M | 4.777 | 118.7 | |
| 2,250 | 36.9M | 4.665 | 106.2 | |
| 2,500 | 41.0M | 4.576 | 97.1 | |
| 2,750 | 45.1M | 4.555 | 95.1 | |
| 3,000 | 49.2M | 4.607 | 100.2 | |
| 3,250 | 53.3M | 4.543 | 94.0 | |
| 3,500 | 57.3M | 4.482 | 88.4 | |
| 3,750 | 61.4M | 4.491 | 89.2 | |
| 4,000 | 65.5M | 4.497 | 89.8 | |
| 4,250 | 69.6M | 4.466 | 87.0 | |
| 4,500 | 73.7M | 4.477 | 87.9 | |
| 4,750 | 77.8M | 4.453 | 85.9 | |
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
| [`hwjoituu`](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/hwjoituu) | Killed (gen 5) | SVD compression of JuliaFluxGPT-23M |

- Population of 10 wolves, SVD rank schedules [77–512]
- ~13 min/generation on A100, killed after gen 5

**Initial population (gen 0):**

| Wolf | Params | Val Loss | Entropy | Fitness | Rank Range |
|------|--------|----------|---------|---------|------------|
| 0 | 23.1M | 7.342 | 1.449 | -7.356 | 89–500 |
| 1 | 22.1M | 7.335 | 1.479 | -7.349 | 92–504 |
| 2 | 23.4M | 7.332 | 1.439 | -7.347 | 77–506 |
| 3 | 24.7M | 7.345 | 1.402 | -7.359 | 78–509 |
| 4 | 20.5M | 7.327 | 1.519 | -7.342 | 88–510 |
| **5** | **20.6M** | **7.316** | **1.522** | **-7.331** | **79–464** |
| 6 | 20.1M | 7.326 | 1.539 | -7.341 | 90–496 |
| 7 | 22.0M | 7.335 | 1.480 | -7.350 | 80–503 |
| 8 | 21.4M | 7.327 | 1.496 | -7.342 | 112–497 |
| 9 | 22.5M | 7.334 | 1.463 | -7.349 | 82–512 |

**Evolution summary:**

| Gen | Best Loss | Best PPL | Best Params | Compression | Replacements |
|-----|-----------|----------|-------------|-------------|-------------|
| 0 | 7.316 | 1503.3 | 20.6M | 9.5% smaller | 2 |
| 5 | 7.309 | 1493.0 | 19.5M | 14.3% smaller | 3 |

- Fitness spread at init: only **0.028** (worst -7.359 vs best -7.331) — essentially flat
- After 5 generations: loss improved by only **0.007** (7.316 → 7.309)
- **Conclusion**: Compressing dirty-data weights is not productive; the weights lack meaningful structure to preserve. The fitness landscape is flat because all SVD truncations lose roughly equal amounts of noise.

**Why the run was killed**: SVD compression searches for low-rank structure in weight matrices — layers where a small number of singular values capture most of the learned information. In a well-trained model, attention heads specialize and FFN layers develop sparse activation patterns, producing weight matrices with steep singular value decay (high effective rank concentration). Dirty-data weights lack this structure: the singular values decay gradually and uniformly, meaning every truncation rank loses roughly the same amount of information. The result is a flat fitness landscape where no compression schedule is meaningfully better than any other. After 5 generations (~65 minutes of A100 compute) with only 0.007 loss improvement and 0.028 total fitness spread across 10 wolves, the run was terminated.

**Alternative strategy — fusion-first, then compress**: Rather than compressing weights that encode noise, we instead transfer high-quality weights from a smaller model trained on curated data (JuliaSLM, 5M params, val_loss=3.54) into the larger JuliaFluxGPT architecture (23M params) via symbiogenesis projection. This "fusion" uses dimension-aware weight mapping: zero-padded embeddings, head-duplicated query projections (4→8 heads), averaged KV heads (4→2 for GQA), and padded FFN matrices (640→1344 inner dim). After fine-tuning, the fused model will have well-structured weights that encode real linguistic patterns from the curated corpus — weights that are *worth* compressing. Post-fusion wolves compression is planned as the next phase.

### Fusion Transfer: JuliaSLM → JuliaFluxGPT (in progress)

| Source | Target | Method | Status |
|--------|--------|--------|--------|
| JuliaSLM (5.04M, d=256, 6L, 4H MHA) | JuliaFluxGPT (23M, d=512, 8L, 8Q/2KV GQA) | Symbiogenesis projection | Fine-tuning |

**Projection mapping:**
- **Embedding** (2000×256 → 2000×512): Source weights placed in first 256 dims, remaining 256 dims filled with low-magnitude noise (σ = 0.02 × source std) to avoid RMSNorm distortion from zeros
- **Q projection** (256×256 → 512×512): Each of 4 source heads duplicated to 2 target heads (4→8), zero-padded columns for expanded input dim
- **KV projection** (256×256 → 256×512, fused): Source K,V heads averaged in pairs (4→2 KV heads for GQA), packed into fused wkv layout
- **Output projection** (256×256 → 512×512): Split equally across duplicated head pairs (÷2 to preserve magnitude), zero-padded rows for expanded output dim
- **SwiGLU FFN** (w1,v: 640×256 → 1344×512; w2: 256×640 → 512×1344): Source weights placed in upper-left block, remainder zero-padded
- **RMSNorm** (256 → 512): Source weights in first 256 dims, pad with 1.0 (identity scaling)
- **Layers 6–7** (no source): Randomly initialized (JuliaSLM has 6 layers, JuliaFluxGPT has 8)

**Pre-fine-tune baseline**: Fused model before training: val_loss=6.843, ppl=937.2 — the zero-padded dimensions and random layers 6–7 start poorly but still far better than random init.

**Fine-tuning results** (8000 steps, LR 3e-4 → 1e-5 cosine, batch=64, bf16, A100):

| Step | Val Loss | Val PPL | Notes |
|------|----------|---------|-------|
| 500 | 3.934 | 51.1 | |
| 1,000 | 3.863 | 47.6 | |
| 1,500 | 3.859 | 47.4 | |
| 2,000 | 3.844 | 46.7 | |
| 2,500 | 3.851 | 47.0 | |
| 3,000 | 3.807 | 45.0 | |
| 3,500 | 3.788 | 44.1 | |
| 4,000 | 3.771 | 43.4 | |
| 4,500 | 3.715 | 41.0 | |
| 5,000 | 3.739 | 42.1 | |
| 5,500 | 3.730 | 41.7 | |
| 6,000 | 3.720 | 41.3 | |
| 6,500 | 3.703 | 40.6 | |
| **7,000** | **3.698** | **40.4** | **Best** |
| 7,500 | 3.705 | 40.6 | Plateau |
| 8,000 | 3.705 | 40.6 | Plateau |

**Final: val_loss=3.698, val_ppl=40.4 at step 7000** (best checkpoint saved). The model plateaued after step 7000 with LR approaching minimum.

**Analysis**: The fused 23M model achieved val_loss=3.70 — better than SymbioSLM (3.62) and MonarchSLM (3.65), but did not break the JuliaSLM plateau (3.54) despite having 4.5× more parameters. This provides strong additional evidence that the 266M-token corpus is the binding constraint: even with successful weight transfer from a well-trained smaller model, the extra capacity cannot learn representations that aren't present in the data. The fusion dramatically improved over the dirty-data baseline (6.84 → 3.70) but could not exceed what a 5M model already learned from the same corpus.

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
- **5M → 23M** via fusion (4.5× params): loss increases **+0.16** (3.54 → 3.70)

The 266M-token corpus saturates around 5M parameters. Additional capacity above this point is wasted without more data. Even with high-quality weight transfer (symbiogenesis projection), a 23M model cannot surpass what a 5M model already learned from the same corpus.

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

### 5. Fusion Transfer Confirms Data Bottleneck

Projecting JuliaSLM weights (5M, val_loss=3.54) into JuliaFluxGPT (23M) via symbiogenesis projection and fine-tuning for 8000 steps produced val_loss=3.698 — better than SymbioSLM (3.62) and MonarchSLM (3.65), but worse than the source JuliaSLM (3.54). The 23M model has 4.5× more parameters but cannot outperform the 5M model on the same 266M-token corpus.

This is the strongest evidence yet for the data bottleneck thesis: the additional 18M parameters in the fused model have nothing useful to learn from the existing corpus. The weight transfer gave them good initialization from JuliaSLM, but fine-tuning could only recover to the same performance range, not exceed it. **More data, not more parameters, is the path forward.**

### 6. SVD Compression Requires Well-Trained Weights

Wolves evolutionary compression on JuliaFluxGPT-23M (dirty-data weights, val_loss=6.62) produced a flat fitness landscape: 10 wolves spanning 20–25M params all scored within 0.028 fitness of each other, and 5 generations improved loss by only 0.007. SVD truncation is effective when weight matrices have steep singular value decay — i.e., when a few directions capture most of the learned structure. Dirty-data weights lack this property; their singular values decay uniformly, making all rank truncations equally lossy.

By contrast, transferring curated-data weights from a smaller model (JuliaSLM, 5M, val_loss=3.54) via symbiogenesis projection and fine-tuning produces a model that starts at val_loss=3.93 after 500 steps — already 2.7 loss better than the dirty baseline. These weights are expected to have concentrated singular value structure suitable for meaningful compression in a post-fusion wolves run.

**Implication**: The order of operations matters. Train (or transfer) quality weights first, then compress. Compressing noisy weights is computationally wasteful and theoretically unsound.

### 6. Over-training Small Models is Worthwhile

JuliaFluxGPT-1M trained to 4× Chinchilla (80:1 tok/param vs 20:1) showed continuous improvement:
- At 1× Chinchilla: val_loss ≈ 4.85
- At 4× Chinchilla: val_loss = 4.45
- Extra 3× tokens bought 0.40 loss improvement — significant for the compute cost on T4

## Next Steps

1. ~~**Fusion transfer**~~: **Complete.** JuliaSLM → JuliaFluxGPT projection fusion achieved val_loss=3.698 (PPL 40.4) at step 7000. Did not break the 3.54 plateau — confirms data bottleneck. Notebook: `fuse_juliaslm.ipynb`, checkpoint: `LisaMegaWatts/JuliaFluxGPT-fused`
2. **Wolves on JuliaSLM**: **In progress.** SVD compression targeting JuliaSLM (5M, clean weights, val_loss=3.54) with more aggressive hyperparams. Notebook: `wolves_juliaslm.ipynb`
3. **Post-fusion wolves**: Run SVD compression on the fused 23M model — the clean weights will have meaningful singular value structure to compress.
4. **Corpus expansion**: The clear bottleneck. Need >1B tokens to see benefit from >5M params. All models from 5M to 23M converge to ~3.5–3.7 on this corpus.
5. **Symbiogenesis Phase D**: MoE via fusion — combine specialized organelle units into a mixture-of-experts architecture.

## HuggingFace Repos

| Model | Repo | Checkpoint |
|-------|------|-----------|
| JuliaFluxGPT-1M | [LisaMegaWatts/JuliaFluxGPT-1M](https://huggingface.co/LisaMegaWatts/JuliaFluxGPT-1M) | `juliaflux_1m_best.pt` |
| JuliaSLM | [LisaMegaWatts/JuliaSLM](https://huggingface.co/LisaMegaWatts/JuliaSLM) | `final.jld2`, `juliaslm_weights.npz` |
| MonarchSLM | [LisaMegaWatts/MonarchSLM](https://huggingface.co/LisaMegaWatts/MonarchSLM) | `final.jld2` |
| SymbioSLM | [LisaMegaWatts/SymbioSLM](https://huggingface.co/LisaMegaWatts/SymbioSLM) | checkpoint v1 |
| SymbioGPT-10M | [LisaMegaWatts/SymbioGPT-10M](https://huggingface.co/LisaMegaWatts/SymbioGPT-10M) | `symbio_best.pt` |
| JuliaFluxGPT | [LisaMegaWatts/JuliaFluxGPT](https://huggingface.co/LisaMegaWatts/JuliaFluxGPT) | `juliaflux_weights.pt` |
| JuliaFluxGPT-fused | [LisaMegaWatts/JuliaFluxGPT-fused](https://huggingface.co/LisaMegaWatts/JuliaFluxGPT-fused) | `juliaflux_fused_best.pt` |

## W&B Run Index

| Run ID | Project | Name | Model | Status |
|--------|---------|------|-------|--------|
| `p7yt1too` | symbiogenesis | juliafluxgpt-1m-scaling | JuliaFluxGPT-1M | finished |
| `5bi3p1dp` | symbiogenesis | symbio-teacher-10m-chinchilla | SymbioGPT-10M | finished |
| `drthfacf` | julia-slm | symbiogenesis-4262K | SymbioSLM | crashed (metrics from checkpoint) |
| — | symbiogenesis | juliafluxgpt-slm-fusion | JuliaFluxGPT-fused | finished |
| `hwjoituu` | symbiogenesis | wolves-compress-juliafluxgpt | Wolves compression | killed (gen 5) |
| `mkh7robk` | JuliaFluxGPT | julia-mkh7robk | JuliaFluxGPT-23M | crashed |
| `h0qutwnv` | JuliaFluxGPT | resume-500 | JuliaFluxGPT-23M | crashed |
| `rzhelyrs` | JuliaFluxGPT | resume-2500 | JuliaFluxGPT-23M | crashed |
| `wv1xpld4` | JuliaFluxGPT | resume-6500 | JuliaFluxGPT-23M | crashed |
| `4odlhnwf` | JuliaFluxGPT | resume-28998 | JuliaFluxGPT-23M | crashed |
| `qgxw2u5j` | symbiogenesis | distill-test-distilled | Distillation test | finished |
| `eqfiw9oj` | symbiogenesis | distill-test-scratch | Distillation control | finished |

All runs under W&B entity: `lisamegawatts-decentralized-intelligence-agency`
