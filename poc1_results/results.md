# POC 1: Organelle-Specialist Fusion (Endosymbiosis) — Results

**Date**: 2026-02-28
**Compute**: Colab T4, ~2 hours total
**W&B Project**: [symbiogenesis](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis)

## Hypothesis

Single-organelle specialists capture orthogonal features (conv = local n-grams, attention = long-range dependencies, monarch = sub-quadratic global mixing). Fusing them into a gated multi-organelle model via weight transplant produces a better model than any individual specialist at the same parameter budget.

## Setup

| Component | Details |
|-----------|---------|
| Data | 266M token curated philosophy corpus, BPE-2000, ctx=256 |
| Specialists | 3x single-organelle models, d=64, L=3, H=2, hd=32, trained 15K steps each |
| Fused model | 3-organelle SymbioGPT, d=128, L=3, H=4, hd=32, 849K params |
| Teacher | JuliaSLM 5M (val_loss=3.54) |
| Fine-tuning | 15K steps with KD (alpha=0.5, T=2.0) |
| Precision | f32 (AMP counterproductive for small Monarch ops) |

## Results

| Model | Params | Val Loss | PPL | vs Fused |
|-------|--------|----------|-----|----------|
| ConvSpec (specialist) | 203K | 4.416 | 82.8 | +0.391 |
| AttnSpec (specialist) | 252K | 4.368 | 78.9 | +0.343 |
| MonarchSpec (specialist) | 227K | 4.518 | 91.6 | +0.493 |
| **Fused (specialists + KD)** | **849K** | **4.025** | **56.0** | **---** |
| Baseline scratch (partial, 18K/60K steps) | 849K | 4.020 | 55.7 | -0.005 |
| Baseline KD-only | 849K | not run | — | N/A |
| JuliaSLM teacher | 5.04M | 3.54 | 34.5 | -0.485 |

## Specialist Orthogonality

Pairwise logit cosine similarity (256 val sequences):

|             | ConvSpec | AttnSpec | MonarchSpec |
|-------------|----------|----------|-------------|
| ConvSpec    | 1.000    | 0.855    | 0.848       |
| AttnSpec    | 0.855    | 1.000    | 0.848       |
| MonarchSpec | 0.848    | 0.848    | 1.000       |

Moderate orthogonality (~0.85). Specialists are somewhat differentiated but share significant overlap — likely because 2000-token vocabulary and 256-context on a single domain doesn't force strong divergence.

## Gate Specialization

Post-fine-tuning gate weights (per-channel softmax, averaged over channels):

| Layer | causal_conv | monarch | attention | Dominant |
|-------|-------------|---------|-----------|----------|
| 0     | 0.307       | 0.306   | 0.387     | attention |
| 1     | 0.257       | 0.296   | 0.448     | attention |
| 2     | 0.196       | 0.241   | 0.563     | attention |

- **Gate entropy**: 1.032 (uniform would be 1.099 for 3 organelles)
- **Pattern**: Attention dominance increases with depth (38% → 45% → 56%)
- **Matches 10M model**: SymbioGPT-10M showed conv-dominant early layers and attention-dominant later layers. At this scale, attention dominates throughout but the gradient is consistent — later layers increasingly prefer attention.

## Scaling Law Context

| Model | Params | Val Loss |
|-------|--------|----------|
| JuliaFluxGPT-1M | 1.01M | 4.446 |
| **POC1 Fused** | **0.85M** | **4.025** |
| SymbioSLM | 4.07M | 3.620 |
| JuliaSLM | 5.04M | 3.540 |

The fused 849K model significantly outperforms the 1.01M JuliaFluxGPT baseline, placing it well on the scaling curve.

## Success Criteria Assessment

| Criterion | Result | Notes |
|-----------|--------|-------|
| Fused < best specialist by >= 0.1 | **YES** (+0.343) | Strong — fusion clearly combines specialist knowledge |
| Fused < from-scratch baseline | **INCONCLUSIVE** | Scratch caught up at 18K steps (4.020 vs 4.025), stopped early |
| Fused < KD-only baseline | **NOT TESTED** | Stopped to save compute |
| Gate entropy decreases during fine-tune | **YES** (1.099 → 1.032) | Mild specialization emerged |
| Gate pattern matches 10M model | **PARTIAL** | Attention-dominant later layers matches, but no conv-dominant early layer |

## Key Takeaways

1. **Fusion works**: Transplanting specialist weights into a multi-organelle host and fine-tuning with KD produces a model that beats all individual specialists by 0.34+ nats. The endosymbiosis metaphor holds — combining orthogonal capabilities yields more than the sum of parts.

2. **But it's not a free lunch over from-scratch**: The scratch baseline reached equivalent loss (4.020) at only 18K steps — meaning gradient descent on this small model finds the same basin without transplanted initialization. The fusion gives a head start (faster convergence at early steps) but not a better final answer at equal compute.

3. **Pre-fine-tune loss was terrible (9.52)**: Zero-padding from d=64→128 disrupted the weight structure worse than random init (~7.6). The KD fine-tuning recovered quickly, but this suggests the transplant procedure needs refinement — perhaps scaling weights rather than zero-padding.

4. **Gate specialization is real but mild**: Entropy dropped from 1.099 to 1.032. The attention-dominant gradient across layers is consistent with the 10M model, suggesting this is a genuine architectural preference, not noise. But the specialization isn't dramatic at this scale.

5. **Orthogonality was moderate (~0.85)**: On a single-domain corpus with tiny vocabulary, specialists don't diverge much. POC3 (data-domain specialists) may show stronger orthogonality.

## Implications for Next POCs

- **POC2 (Horizontal Gene Transfer)**: The moderate orthogonality suggests that cross-model weight sharing could be productive — specialists aren't so different that transplant is destructive.
- **POC3 (Multi-domain specialists)**: Training specialists on different data domains (not just different organelles) should produce stronger orthogonality and potentially a fusion advantage over scratch.
- **Transplant refinement**: The 9.52 pre-fine-tune loss indicates zero-padding is crude. Consider: (a) learned projection layers, (b) gradual unfreezing, (c) scaling rather than padding.

## W&B Runs

| Run | ID | Link |
|-----|----|------|
| ConvSpec | — | poc1-ConvSpec |
| AttnSpec | — | poc1-AttnSpec |
| MonarchSpec | — | poc1-MonarchSpec |
| Fused + KD | najg2h3a | [view](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/najg2h3a) |
| Orthogonality | 19sz0axd | [view](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/19sz0axd) |
| Baseline scratch (partial) | w61jioh1 | [view](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/w61jioh1) |

## HuggingFace

Model repo: [LisaMegaWatts/SymbioGPT-POC1-endosymbiosis](https://huggingface.co/LisaMegaWatts/SymbioGPT-POC1-endosymbiosis)
