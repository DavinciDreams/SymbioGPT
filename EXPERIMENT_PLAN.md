# Symbiogenesis Experiment Plan

**Last updated**: 2026-02-28
**Status**: Post-POC1, planning next experiments
**Compute constraint**: Colab T4/A100 (free tier)

---

## What We Know (Established Results)

| Finding | Source | Implication |
|---------|--------|-------------|
| Fusion beats all specialists by +0.34 nats | POC1 | Combining orthogonal organelles works |
| Scratch baseline catches fused model at 18K steps | POC1 | Fusion = faster convergence, not better final loss (at 849K scale) |
| Zero-padding transplant gives worse-than-random init (9.52 vs 7.6) | POC1 | Transplant method is the bottleneck, not the concept |
| Specialist orthogonality is moderate (~0.85 cosine) | POC1 | Same-domain training limits specialist diversity |
| Gate specialization emerges: attention increases with depth | POC1 + 10M teacher | Universal pattern — conv early, attention later |
| KD benefit is transient (scratch catches up by 5K steps) | distill_test.ipynb | Teacher value is architectural, not pedagogical |
| All models 5M-23M plateau at loss 3.5-3.7 on 266M tokens | SCALING_LAW_REPORT | Data bottleneck is binding — more params don't help |
| Monarch/LongConv gates stay weak (~15-22%) in 10M teacher | organelle_findings | 2 of 4 organelles may be deadweight |

---

## Experiments Ranked by Information Value

### Tier 1: High value, low compute

These directly test whether the core findings are real or artifacts of POC1's design.

#### POC2: Transplant Refinement — Scaling vs Zero-Padding

**Question**: Does the 9.52 pre-finetune loss (worse than random) mean transplant is fundamentally broken, or just that zero-padding is crude?

**Design**: Same 3 specialists from POC1, same fused target (849K). Compare 3 transplant methods:
- **A) Zero-pad** (POC1 baseline): pad (64→128) with zeros
- **B) Scale**: multiply specialist weights by √(d_tgt/d_src) to preserve variance, pad remainder with scaled noise
- **C) Learned projection**: insert a frozen 64→128 linear layer per transplanted module, train projections for 1K steps, then unfreeze everything

**Compute**: ~1 hour on T4 (reuse specialist checkpoints, only fine-tune 3 variants × 15K steps)

**Success**: Method B or C gives pre-finetune loss < 7.6 (better than random init) AND final loss < scratch baseline at matched compute.

**Why this matters**: If we can transplant without destroying the weight structure, fusion becomes a genuine initialization advantage — not just a convergence accelerator.

---

#### POC3: Organelle Pruning — Do We Need All Three?

**Question**: The 10M teacher showed Monarch/LongConv gates at 15-22%. Are they load-bearing or dead weight?

**Design**: Take the trained fused model from POC1 (849K, loss 4.025). Ablate:
- **A) Full 3-organelle** (POC1 result): conv + monarch + attention
- **B) Drop monarch**: mask monarch gate to zero, measure loss degradation
- **C) Drop conv**: mask conv gate to zero
- **D) Attention-only**: mask both conv and monarch
- **E) 2-organelle from scratch**: train conv+attention only (no monarch) from scratch, same compute

**Compute**: <30 min (A-D are just eval with masking, E is one training run)

**Success**: If dropping monarch costs < 0.05 nats, remove it from future experiments (fewer params, faster training, cleaner signal).

**Why this matters**: Every unnecessary organelle dilutes the gate signal and wastes params. The 10M teacher already hints monarch is marginal. Confirming this at small scale simplifies everything downstream.

---

### Tier 2: Medium value, medium compute

#### POC4: Domain-Specialist Fusion

**Question**: Does training specialists on different *data* (not just different organelles) produce stronger orthogonality and a fusion advantage over scratch?

**Design**:
- Split the 266M token corpus into 3 domains by topic (philosophy, science, literature — or temporal/style splits)
- Train 3 identical multi-organelle models (each 300K, same architecture), one per domain
- Fuse into 900K target using best transplant method from POC2
- Compare fused vs scratch baseline

**Compute**: ~2 hours on T4

**Hypothesis**: Domain specialists should have lower logit cosine similarity (< 0.80 vs POC1's 0.85) because they literally saw different data. This should produce a fusion advantage that *doesn't* get caught by scratch.

**Why this matters**: If orthogonality is the key variable, and data diversity drives orthogonality more than architecture diversity, then the scaling story becomes: train many small domain-expert models in parallel, fuse them. This is embarrassingly parallel and fundamentally different from training one big model.

---

#### POC5: Evolutionary Organelle Selection

**Question**: Can we let a population of models with different organelle combinations compete, and have selection pressure find the optimal combination?

**Design**:
- Population of 8-12 models, each ~300K params
- Each model gets a random subset of organelles (1-3 from {conv, monarch, attention})
- Train all for 5K steps (short — just enough to differentiate)
- Rank by val loss
- Top 4 survive, bottom 4 die
- "Reproduce": copy survivors' organelle configs, mutate (add/remove one organelle with p=0.3)
- Repeat for 3-5 generations

**Compute**: ~3 hours on T4 (5K steps × 8 models × 5 generations = 200K steps total)

**Success**: Population converges to a consistent organelle combination. Compare to hand-picked combination.

**Why this matters**: This is the symbiogenesis thesis in miniature — can evolution discover architecture? If the population consistently converges to conv+attention (dropping monarch), that's independent confirmation of POC3's ablation.

---

### Tier 3: High value, high compute (need A100 or corpus expansion)

#### POC6: Scale Test — Does Fusion Advantage Grow with Model Size?

**Question**: At 849K, scratch catches fused in 18K steps. Does this gap widen or narrow at 5M? At 10M?

**Design**:
- Repeat POC1 protocol at 3 scales: 849K (done), 3M, 10M
- Use best transplant method from POC2
- Full baseline comparison at each scale

**Compute**: ~8 hours on A100 (3M and 10M take real compute)

**Hypothesis**: If fusion advantage grows with scale, symbiogenesis is a scaling method. If it shrinks, it's only useful for small models (warm-start, not architecture).

**Why this matters**: This is the make-or-break experiment for the whole thesis. Everything else is groundwork for this.

---

#### POC7: Corpus Expansion

**Question**: Is 266M tokens the real bottleneck, or is the architecture also limiting?

**Design**:
- Expand corpus to 1B+ tokens (add more philosophy texts, adjacent domains)
- Retrain JuliaSLM-5M teacher → expect loss < 3.2
- Retrain SymbioGPT-10M → check if organelle specialization changes with more data

**Compute**: 4-6 hours on A100 per model

**Why this matters**: Every experiment above is bounded by the 266M token ceiling. Breaking through it is prerequisite for meaningful scaling claims.

---

## Experiment Dependency Graph

```
POC1 (DONE)
  │
  ├── POC2 (transplant refinement) ← NEXT, reuses POC1 checkpoints
  │     │
  │     └── POC4 (domain specialists) ← uses best transplant method
  │           │
  │           └── POC6 (scale test) ← uses best transplant + domain approach
  │
  ├── POC3 (organelle pruning) ← NEXT, just eval + 1 training run
  │     │
  │     └── POC5 (evolutionary selection) ← validates pruning findings
  │
  └── POC7 (corpus expansion) ← independent, unlocks all scale experiments
```

## Recommended Execution Order

1. **POC3** (organelle pruning) — 30 min, answers "do we need monarch?" immediately
2. **POC2** (transplant refinement) — 1 hour, fixes the biggest POC1 weakness
3. **POC4** (domain specialists) — 2 hours, tests whether data orthogonality > architecture orthogonality
4. **POC5** (evolutionary selection) — 3 hours, tests the symbiogenesis thesis directly
5. **POC7** (corpus expansion) — prerequisite for any claims beyond toy scale
6. **POC6** (scale test) — the thesis experiment, needs POC2+POC7 first

---

## Open Questions

1. **Is per-channel gating the right granularity?** Per-channel means each embedding dimension picks its own organelle. Would per-token or per-head gating produce cleaner specialization?

2. **Should specialists share embeddings?** POC1 transplanted embeddings from the best specialist. What if all specialists share a frozen embedding layer and only diverge in sequence mixing?

3. **Is the gate entropy metric sufficient?** Entropy dropping from 1.099 to 1.032 looks mild. What about mutual information between gate weights and token properties (e.g., position, frequency)?

4. **Monarch's role**: At ctx=256, Monarch's O(n√n) advantage over attention's O(n²) is negligible (256² = 65K vs 256×16 = 4K — both trivial). Monarch may only matter at ctx >= 4096. Should we test at longer context?

---

## Resources

| Resource | Location |
|----------|----------|
| POC1 notebook | `poc1_endosymbiosis.ipynb` |
| POC1 results | `poc1_results/results.md` |
| SymbioGPT model | `symbio_model.py` |
| Scaling analysis | `SCALING_LAW_REPORT.md` |
| 10M gate findings | `organelle_findings.md` |
| Distillation A/B test | `distill_test.ipynb` |
| Training script | `train_teacher.py` |
| W&B project | [symbiogenesis](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis) |
| Notebook templates | `/home/ubuntu/Dev/notebook-templates/` |
