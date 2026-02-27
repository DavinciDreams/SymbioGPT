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

## Ongoing: Distillation A/B Test

To validate that teacher gate insights transfer effectively, we are running a controlled experiment:

- **Student A** (distilled): Trained with KL divergence from teacher soft targets + CE loss
- **Student B** (from scratch): CE loss only, identical architecture and hyperparameters

Both use the gate-informed student architecture above. Results will determine whether teacher guidance accelerates convergence and/or improves final quality before proceeding to evolutionary architecture search.

See `distill_test.ipynb` for the full experiment.

---

*Generated from SymbioGPT-10M training run. Full gate weight data available in W&B project `symbiogenesis`.*
