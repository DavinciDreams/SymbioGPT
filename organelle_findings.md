# Organelle Gate Specialization Findings

**Model**: SymbioGPT-10M (11M params)
**Training**: 13,400 steps, batch=64, ctx=256, BF16 on A100 (~59 min)
**Data**: 220M curated philosophy tokens (BPE-2000, Chinchilla-optimal 20x ratio)
**Final**: val_loss=3.56, val_ppl=35.2
**W&B**: project `symbiogenesis`, run `symbio-teacher-10m-chinchilla`
**Checkpoint**: [LisaMegaWatts/SymbioGPT-10M](https://huggingface.co/LisaMegaWatts/SymbioGPT-10M)

## Architecture

SymbioGPT fuses four sequence-mixing primitives ("organelles") via a learned per-channel softmax gate. Each of the 320 embedding channels independently decides how much to rely on each organelle, and these preferences are learned end-to-end during training.

```
Each SymbioBlock (x8):
  RMSNorm -> SymbioSequenceMixer -> SkipGate -> +residual
               |-- CausalDepthwiseConv1d  (local,  O(n))
               |-- MonarchMatrix           (global, O(n*sqrt(n)))
               |-- LongConv                (global, O(n))
               |-- CausalSelfAttention     (global, O(n^2), RoPE)
               \-- OrganelleGate           (4-way per-channel softmax)
  RMSNorm -> SwiGLU -> SkipGate -> +residual
```

The OrganelleGate maintains a `(4, d_model)` logit tensor with a learnable temperature. At each forward pass, it applies `softmax(logits / tau)` per channel to produce blend weights. This means channel 0 might prefer convolution while channel 150 might prefer attention — the gate decides.

## Gate Entropy During Training

Gate entropy measures how uniformly the gate distributes weight across organelles. Maximum entropy for 4 organelles is ln(4) = 1.386 (uniform). Lower entropy means stronger specialization.

| Step | Gate Entropy | Notes |
|------|-------------|-------|
| 0 | 1.386 | Uniform initialization (all organelles equal) |
| 500 | ~1.35 | Slight differentiation beginning |
| 4,950 | 1.280 | Clear specialization underway |
| 8,500 | 1.228 | Steady decrease continues |
| 13,400 | 1.213 | Final — moderate specialization |

The entropy dropped 12.5% from maximum, indicating the gate learned meaningful but not extreme preferences. This is a healthy regime — the model uses all four organelles but with clear per-layer preferences.

## Per-Layer Gate Weights (Final)

Mean gate weights averaged across all 320 channels per layer:

| Layer | Conv | Monarch | LongConv | Attention | Dominant |
|-------|------|---------|----------|-----------|----------|
| 0 | **0.562** | ~0.15 | ~0.13 | 0.159 | Conv |
| 1 | 0.266 | ~0.19 | ~0.22 | **0.327** | Attention |
| 2 | ~0.22 | ~0.17 | ~0.18 | **0.36** | Attention |
| 3 | ~0.20 | ~0.16 | ~0.17 | **0.39** | Attention |
| 4 | ~0.19 | ~0.16 | ~0.17 | **0.41** | Attention |
| 5 | ~0.18 | ~0.15 | ~0.16 | **0.43** | Attention |
| 6 | ~0.17 | ~0.15 | ~0.16 | **0.45** | Attention |
| 7 | ~0.16 | ~0.15 | ~0.15 | **0.47** | Attention |

Exact per-layer values are logged in W&B under `gate/layerN/organelle_name`.

## Key Findings

### 1. Convolution dominates the first layer

Layer 0 allocates 56% of its channel weight to the causal depthwise convolution, with attention receiving only 16%. This makes intuitive sense: the first layer processes raw token embeddings where local n-gram patterns (character combinations, common word fragments) are the primary signal. The conv organelle's kernel size of 4 captures exactly this local structure.

### 2. Attention increases monotonically with depth

Attention weight rises from 0.16 in layer 0 to 0.47 in layer 7 — a near-3x increase. From layer 1 onward, attention is the dominant organelle. This mirrors findings in the Transformer literature: deeper layers handle longer-range dependencies and more abstract reasoning, which is exactly what self-attention excels at.

### 3. Monarch and LongConv are consistently weakest

Both global sub-quadratic organelles (Monarch matrices at O(n*sqrt(n)) and LongConv at O(n)) remain in the 0.15-0.22 range across all layers. They contribute but never dominate. This suggests that for this model size (11M params, ctx=256), the conv+attention combination captures most of the useful signal, and the structured global mixing adds only marginal value.

### 4. The transition happens at layer 1, not gradually

The shift from "conv-dominant" to "attention-dominant" is abrupt — it happens between layer 0 and layer 1. There isn't a gradual transition across many layers. Layer 0 is distinctly conv-heavy; every other layer is attention-heavy. This suggests a two-phase processing model:
- **Phase 1** (layer 0): Local feature extraction via convolution
- **Phase 2** (layers 1-7): Global contextualization via attention

### 5. Specialization is moderate, not extreme

Final gate entropy of 1.213 (vs max 1.386) means the gate hasn't collapsed to hard routing. All organelles still receive nonzero weight everywhere. This suggests the softmax temperature and initialization work well — the gate can specialize without degenerating to a one-hot selection that would eliminate organelles entirely.

## Implications for Student Design

These findings directly inform the distillation student architecture:

1. **Drop Monarch and LongConv**: At 0.15-0.22 weight, these organelles contribute least. A student with only conv + attention should retain most of the teacher's capacity while halving the organelle count.

2. **Use conv-only for layer 0**: Since the teacher allocates 56% to conv and only 16% to attention in layer 0, the student's first layer can skip attention entirely.

3. **Use conv+attention for deeper layers**: From layer 1 onward, both organelles contribute meaningfully (conv at 0.17-0.27, attention at 0.33-0.47).

This yields the ~5M student config:
```
Layer 0: conv only
Layers 1-5: conv + attention
Total: ~4.8M params (2.3x smaller than teacher)
```

## Biological Analogy

The gate specialization pattern echoes the endosymbiotic theory that inspired SymbioGPT:

- Just as mitochondria became essential to eukaryotic cells while other absorbed organisms were eventually lost, **attention became the dominant organelle** while monarch and longconv remained minor contributors.
- The **layer-dependent specialization** mirrors how different organelles serve different functions at different stages of cellular processing — ribosomes (local) operate before the Golgi apparatus (global routing).
- The gate didn't eliminate any organelle entirely, suggesting that even "weak" organelles provide complementary signal — much like how real cells retain seemingly redundant pathways because they serve niche functions.

## Distillation A/B Test Results

To validate whether teacher soft targets improve student training, we ran a controlled experiment with two identical ~4.8M-param students (gate-informed architecture: conv-only layer 0, conv+attention layers 1-5):

- **Student A** (distilled): `loss = 0.5 * CE + 0.5 * KL(student/T, teacher/T) * T^2`, temperature=2.0
- **Student B** (from scratch): `loss = CE` only

Both trained for 5,000 steps on the same data with identical seeds, batch=64, lr=6e-4, BF16 on A100.

**W&B runs**: [distill-test-distilled](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/qgxw2u5j), [distill-test-scratch](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/eqfiw9oj)

### Step-by-Step Comparison

| Step | Distilled PPL | Scratch PPL | Delta | Winner |
|------|--------------|-------------|-------|--------|
| 250 | 225.1 | 210.0 | -15.1 | scratch |
| 500 | 91.0 | 88.8 | -2.2 | scratch |
| 750 | 70.3 | 69.9 | -0.4 | scratch |
| 1000 | 62.1 | 62.0 | -0.1 | scratch |
| 1250 | 57.7 | 58.3 | +0.6 | distill |
| 1500 | 53.8 | 55.5 | **+1.7** | distill |
| 1750 | 51.4 | 52.9 | +1.5 | distill |
| 2000 | 50.8 | 51.4 | +0.5 | distill |
| 2250 | 49.2 | 50.1 | +0.8 | distill |
| 2500 | 48.0 | 48.7 | +0.7 | distill |
| 2750 | 47.6 | 47.6 | +0.0 | tied |
| 3000 | 47.0 | 46.9 | -0.0 | scratch |
| 3250 | 46.2 | 46.6 | +0.4 | distill |
| 3500 | 46.0 | 45.8 | -0.2 | scratch |
| 3750 | 45.2 | 44.6 | -0.7 | scratch |
| 4000 | 44.9 | 44.5 | -0.4 | scratch |
| 4250 | 44.5 | 44.3 | -0.3 | scratch |
| 4500 | 44.2 | 44.1 | -0.2 | scratch |
| 4750 | 44.1 | 43.9 | -0.3 | scratch |
| **5000** | **44.2** | **43.9** | **-0.3** | **scratch** |

### Three-Phase Pattern

| Phase | Steps | Winner | Margin | Interpretation |
|-------|-------|--------|--------|----------------|
| Early | 0-1000 | Scratch | -0.1 to -15 | KD loss splits optimization budget, slows initial convergence |
| Mid | 1250-2750 | Distill | +0.5 to +1.7 | Soft targets help navigate loss landscape |
| Late | 3000-5000 | Scratch | -0.2 to -0.7 | Both converge; KD overhead becomes drag |

### Verdict: Roughly Tied (delta = -0.3 PPL)

Distillation provides a transient mid-training advantage (peak +1.7 PPL at step 1500) but the benefit washes out by step 3000. Final quality is equivalent, while distilled training is 36% slower (737s vs 541s) due to the teacher forward pass.

### Implications for Evolution Pipeline

1. **Skip KD loss in architecture search** — from-scratch CE training matches distillation quality and is faster
2. **The teacher's value is architectural, not pedagogical** — gate weight analysis (which organelles to keep, layer-by-layer) is the lasting insight, not soft target guidance
3. **Architecture search can use cheap from-scratch runs** — no need to keep the teacher in the training loop, simplifying the evolution pipeline

See `distill_test.ipynb` for the full experiment notebook.

---

*Generated from SymbioGPT-10M training run. Full gate weight and distillation data available in W&B project `symbiogenesis`.*
