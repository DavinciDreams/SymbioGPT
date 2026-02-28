#!/usr/bin/env python3
"""Symbiogenesis LoRA Evolution on Gemma-3-270M (local run).

Optimized for RTX 3060 12GB:
- No deepcopy: uses PEFT adapter add/delete on single base model
- Batch size 2 + grad accum 4 (effective batch 8)
- 262k vocab = huge logits, so small batches are critical
"""

import os, math, time, random, json, gc
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import Counter

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import wandb

# ── Config ──────────────────────────────────────────────────────────────────
DATA_REPO = 'LisaMegaWatts/SymbioGPT-10M'
MODEL_NAME = 'LisaMegaWatts/Ouroboros-1MContext-Gemma-270m'
CTX = 512

POP_SIZE = 10
GENERATIONS = 50
CHILDREN_PER_GEN = 2
TOURNAMENT_K = 3
TRAIN_STEPS_PER_UNIT = 300
UNIT_LR = 2e-4
UNIT_BATCH = 2        # small for 12GB GPU (262k vocab = huge logits)
GRAD_ACCUM = 4        # effective batch = 8
BETA = 0.01
GELATION_PATIENCE = 10

EXTENDED_STEPS = 2000
EXTENDED_LR = 2e-4
EXTENDED_BATCH = 2
EXTENDED_GRAD_ACCUM = 4
EXTENDED_WARMUP = 100
EVAL_EVERY = 250

EVAL_BATCH = 4        # eval doesn't need gradients, can be a bit larger

# ── GPU + dtype ─────────────────────────────────────────────────────────────
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f"GPU: {gpu_name} ({mem / 1e9:.1f} GB)")
    DTYPE = torch.bfloat16 if any(x in gpu_name.lower() for x in ['a100', 'h100', 'a10', 'l4', 'l40', '3060', '3070', '3080', '3090', '4060', '4070', '4080', '4090']) else torch.float16
else:
    DTYPE = torch.float32
print(f"Using dtype: {DTYPE}")

# ── W&B + HF login ─────────────────────────────────────────────────────────
wandb.login()
from huggingface_hub import login as hf_login
hf_login(token=os.environ.get("HF_TOKEN"), add_to_git_credential=False)

# ── Data pipeline ───────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)

print('Downloading raw text...')
hf_hub_download(repo_id=DATA_REPO, filename='data/train_curated_sample.txt', local_dir='.')
hf_hub_download(repo_id=DATA_REPO, filename='data/val_sample.txt', local_dir='.')

print(f'Loading tokenizer: {MODEL_NAME}')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = len(tokenizer)
print(f'Vocab size: {VOCAB_SIZE:,}')

print('Tokenizing training data...')
with open('data/train_curated_sample.txt', 'r') as f:
    train_text = f.read()
train_ids = tokenizer.encode(train_text, add_special_tokens=False)
print(f'Train tokens: {len(train_ids):,}')

print('Tokenizing validation data...')
with open('data/val_sample.txt', 'r') as f:
    val_text = f.read()
val_ids = tokenizer.encode(val_text, add_special_tokens=False)
print(f'Val tokens: {len(val_ids):,}')


def chunk_tokens(token_ids, seq_len):
    n = len(token_ids) // (seq_len + 1)
    token_ids = token_ids[:n * (seq_len + 1)]
    data = torch.tensor(token_ids, dtype=torch.long).reshape(n, seq_len + 1)
    return data[:, :-1], data[:, 1:]


train_inputs, train_labels = chunk_tokens(train_ids, CTX)
val_inputs, val_labels = chunk_tokens(val_ids, CTX)
print(f'Train: {len(train_inputs):,} seqs ({len(train_inputs)*CTX:,} tokens)')
print(f'Val: {len(val_inputs):,} seqs')
del train_text, val_text, train_ids, val_ids

# ── Load frozen base model (single copy, on CUDA) ──────────────────────────
print(f'\nLoading {MODEL_NAME} (frozen base, dtype={DTYPE})...')
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
).to(device)
base_model.eval()
for p in base_model.parameters():
    p.requires_grad = False

n_base_params = sum(p.numel() for p in base_model.parameters())
print(f'Base model: {n_base_params:,} params ({n_base_params/1e6:.0f}M)')
print(f'Device: {next(base_model.parameters()).device}')
print(f'GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB / '
      f'{torch.cuda.max_memory_allocated()/1e9:.2f} GB peak')

linear_names = set()
for name, module in base_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_names.add(name.split('.')[-1])
print(f'Available Linear modules: {sorted(linear_names)}')

# Baseline eval
print('\nBaseline eval (no adapter)...')
total_loss = 0.0
total_tokens = 0
with torch.no_grad():
    for i in range(0, min(len(val_inputs), 64), EVAL_BATCH):
        batch_in = val_inputs[i:i+EVAL_BATCH].to(device)
        batch_tgt = val_labels[i:i+EVAL_BATCH].to(device)
        out = base_model(batch_in)
        logits = out.logits.float()
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B*T, V), batch_tgt.reshape(B*T), reduction='sum')
        total_loss += loss.item()
        total_tokens += B * T
        del out, logits, loss
base_loss = total_loss / total_tokens
base_ppl = math.exp(min(base_loss, 20.0))
print(f'Baseline: val_loss={base_loss:.4f} ppl={base_ppl:.1f}')
print(f'GPU mem after baseline: {torch.cuda.memory_allocated()/1e9:.2f} GB')

# ── LoRA unit ───────────────────────────────────────────────────────────────
ATTN_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
MLP_TARGETS = ('gate_proj', 'up_proj', 'down_proj')
ALL_TARGETS = ATTN_TARGETS + MLP_TARGETS

TARGET_POOLS = [
    ('q_proj', 'v_proj'),
    ('q_proj', 'k_proj', 'v_proj'),
    ('q_proj', 'k_proj', 'v_proj', 'o_proj'),
    ('q_proj', 'v_proj', 'gate_proj', 'up_proj'),
    ALL_TARGETS,
]

RANK_POOL = [4, 8, 16, 32]
MAX_RANK = 64
MAX_TARGETS = len(ALL_TARGETS)


@dataclass
class LoRAUnit:
    rank: int
    target_modules: Tuple[str, ...]
    alpha: float = 16.0
    dropout: float = 0.0
    val_loss: float = float('inf')
    n_trainable: int = 0
    fitness: float = float('-inf')
    generation: int = 0

    @property
    def depth(self):
        return len(self.target_modules)

    @property
    def width(self):
        return self.rank

    def arch_key(self):
        return (self.rank, tuple(sorted(self.target_modules)))

    def summary(self):
        targets = ','.join(t.replace('_proj', '') for t in self.target_modules)
        return (f'r={self.rank} [{targets}] loss={self.val_loss:.4f} '
                f'fit={self.fitness:.4f} params={self.n_trainable:,}')


def random_unit(gen=0):
    rank = random.choice(RANK_POOL)
    targets = random.choice(TARGET_POOLS)
    dropout = random.choice([0.0, 0.0, 0.05])
    return LoRAUnit(
        rank=rank, target_modules=targets, alpha=2.0 * rank,
        dropout=dropout, generation=gen,
    )


def make_lora_config(unit: LoRAUnit) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=unit.rank,
        lora_alpha=int(unit.alpha), target_modules=list(unit.target_modules),
        lora_dropout=unit.dropout, bias='none',
    )


# ── Fusion ──────────────────────────────────────────────────────────────────
def fuse_sequential(a: LoRAUnit, b: LoRAUnit, gen: int) -> Optional[LoRAUnit]:
    seen = set()
    new_targets = []
    for t in list(a.target_modules) + list(b.target_modules):
        if t not in seen:
            seen.add(t)
            new_targets.append(t)
    if len(new_targets) > MAX_TARGETS:
        return None
    new_rank = max(4, min((a.rank + b.rank) // 2, MAX_RANK))
    return LoRAUnit(rank=new_rank, target_modules=tuple(new_targets),
                    alpha=2.0 * new_rank,
                    dropout=random.choice([a.dropout, b.dropout]), generation=gen)


def fuse_parallel(a: LoRAUnit, b: LoRAUnit, gen: int) -> Optional[LoRAUnit]:
    seen = set()
    new_targets = []
    for t in list(a.target_modules) + list(b.target_modules):
        if t not in seen:
            seen.add(t)
            new_targets.append(t)
    if len(new_targets) > MAX_TARGETS:
        return None
    new_rank = a.rank + b.rank
    if new_rank > MAX_RANK:
        return None
    return LoRAUnit(rank=new_rank, target_modules=tuple(new_targets),
                    alpha=2.0 * new_rank,
                    dropout=random.choice([a.dropout, b.dropout]), generation=gen)


def fuse_hybrid(a: LoRAUnit, b: LoRAUnit, gen: int) -> Optional[LoRAUnit]:
    return fuse_sequential(a, b, gen) if random.random() < 0.5 else fuse_parallel(a, b, gen)


def mutate(unit: LoRAUnit, mutation_rate=0.3) -> LoRAUnit:
    rank = unit.rank
    targets = list(unit.target_modules)
    if random.random() < mutation_rate:
        rank = max(4, min(rank + random.choice([-4, -2, 2, 4]), MAX_RANK))
    if random.random() < mutation_rate:
        available = [t for t in ALL_TARGETS if t not in targets]
        if available and len(targets) < MAX_TARGETS:
            targets.append(random.choice(available))
        elif len(targets) > 1:
            targets.remove(random.choice(targets))
    return LoRAUnit(rank=rank, target_modules=tuple(targets), alpha=2.0 * rank,
                    dropout=unit.dropout, generation=unit.generation)


def tournament_select(population: List[LoRAUnit], k=3) -> LoRAUnit:
    contestants = random.sample(population, min(k, len(population)))
    return max(contestants, key=lambda u: u.fitness)


# ── Train + eval (no deepcopy — adapter add/delete) ────────────────────────
def evaluate_lm(model, val_inputs, val_labels, batch_size=EVAL_BATCH, max_batches=32):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = min(len(val_inputs) // batch_size, max_batches)
    with torch.no_grad():
        for i in range(n_batches):
            batch_in = val_inputs[i*batch_size:(i+1)*batch_size].to(device)
            batch_tgt = val_labels[i*batch_size:(i+1)*batch_size].to(device)
            out = model(batch_in)
            logits = out.logits.float()
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), batch_tgt.reshape(B*T), reduction='sum')
            total_loss += loss.item()
            total_tokens += B * T
            del out, logits, loss
    return total_loss / max(total_tokens, 1), math.exp(min(total_loss / max(total_tokens, 1), 20.0))


# Global adapter counter to ensure unique names
_adapter_counter = 0


def train_and_eval_unit(unit, base_model, train_inputs, train_labels,
                        val_inputs, val_labels, n_steps=300, lr=2e-4,
                        batch_size=UNIT_BATCH, grad_accum=GRAD_ACCUM, warmup=50):
    """Train a LoRA unit without copying the base model.

    Uses PEFT adapter naming: adds adapter, trains, evals, deletes adapter.
    """
    global _adapter_counter
    adapter_name = f'unit_{_adapter_counter}'
    _adapter_counter += 1

    lora_config = make_lora_config(unit)

    # Wrap base model with PEFT (first time) or add adapter
    if not hasattr(base_model, 'peft_config'):
        model = get_peft_model(base_model, lora_config, adapter_name=adapter_name)
    else:
        model = base_model
        model.add_adapter(adapter_name, lora_config)
        model.set_adapter(adapter_name)

    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    unit.n_trainable = sum(p.numel() for p in trainable_params)

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(n_steps - warmup, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_train = len(train_inputs)
    step = 0
    accum_step = 0
    while step < n_steps:
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            if step >= n_steps:
                break
            idx = perm[i:i+batch_size]
            if len(idx) < batch_size:
                continue
            batch_in = train_inputs[idx].to(device)
            batch_tgt = train_labels[idx].to(device)

            with torch.amp.autocast('cuda', dtype=DTYPE):
                out = model(batch_in)
                logits = out.logits
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.reshape(B*T, V), batch_tgt.reshape(B*T))
                loss = loss / grad_accum  # scale for accumulation

            loss.backward()
            del out, logits, loss
            accum_step += 1

            if accum_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

    # Evaluate
    val_loss, val_ppl = evaluate_lm(model, val_inputs, val_labels)
    unit.val_loss = val_loss
    unit.fitness = -(val_loss + BETA * math.log(max(unit.n_trainable, 1)))

    # Cleanup: delete adapter, free optimizer
    del optimizer, scheduler, trainable_params
    model.delete_adapter(adapter_name)
    gc.collect()
    torch.cuda.empty_cache()

    return unit, model  # return model so we can reuse the PeftModel wrapper


# ── CUSUM gelation ──────────────────────────────────────────────────────────
class GelationMonitor:
    def __init__(self, baseline_window=5, sensitivity=4.0):
        self.baseline_window = baseline_window
        self.sensitivity = sensitivity
        self.depth_history = []
        self.width_history = []
        self.fitness_history = []
        self.cusum_depth = 0.0
        self.cusum_width = 0.0
        self.baseline_depth_mean = None
        self.baseline_depth_std = None
        self.baseline_width_mean = None
        self.baseline_width_std = None
        self.gelation_detected = False
        self.gelation_step = None

    def record(self, avg_depth, avg_width, avg_fitness, best_fitness):
        self.depth_history.append(avg_depth)
        self.width_history.append(avg_width)
        self.fitness_history.append(best_fitness)
        step = len(self.depth_history)

        if step == self.baseline_window:
            d, w = self.depth_history, self.width_history
            self.baseline_depth_mean = sum(d) / len(d)
            self.baseline_depth_std = max((sum((x - self.baseline_depth_mean)**2 for x in d) / len(d))**0.5, 0.1)
            self.baseline_width_mean = sum(w) / len(w)
            self.baseline_width_std = max((sum((x - self.baseline_width_mean)**2 for x in w) / len(w))**0.5, 0.1)
        elif step > self.baseline_window and not self.gelation_detected:
            depth_dev = (avg_depth - self.baseline_depth_mean) / self.baseline_depth_std
            width_dev = (avg_width - self.baseline_width_mean) / self.baseline_width_std
            self.cusum_depth = max(0.0, self.cusum_depth + depth_dev)
            self.cusum_width = max(0.0, self.cusum_width + width_dev)
            if self.cusum_depth > self.sensitivity or self.cusum_width > self.sensitivity:
                self.gelation_detected = True
                self.gelation_step = step


# ── Main evolution ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    monitor = GelationMonitor(baseline_window=5, sensitivity=4.0)

    run = wandb.init(
        project='symbiogenesis',
        name='symbiogenesis-gemma-270m-lora-rtx3060',
        config={
            'method': 'symbiogenesis_lora_evolution',
            'base_model': MODEL_NAME,
            'base_params': n_base_params,
            'base_loss': base_loss,
            'base_ppl': base_ppl,
            'task': 'causal_lm',
            'corpus': 'philosophy_curated',
            'context_length': CTX,
            'population_size': POP_SIZE,
            'generations': GENERATIONS,
            'children_per_gen': CHILDREN_PER_GEN,
            'train_steps_per_unit': TRAIN_STEPS_PER_UNIT,
            'unit_lr': UNIT_LR,
            'unit_batch': UNIT_BATCH,
            'grad_accum': GRAD_ACCUM,
            'effective_batch': UNIT_BATCH * GRAD_ACCUM,
            'beta': BETA,
            'fusion_strategy': 'hybrid',
            'rank_pool': RANK_POOL,
            'max_rank': MAX_RANK,
            'gpu': gpu_name if torch.cuda.is_available() else 'cpu',
        },
        tags=['symbiogenesis', 'lora', 'gemma-270m', 'causal-lm', 'philosophy', 'rtx3060'],
        reinit='finish_previous',
    )

    # Initialize population
    print(f'\nInitializing population of {POP_SIZE} units...')
    population = []
    t_start = time.time()
    peft_model = base_model  # will become PeftModel after first unit

    for i in range(POP_SIZE):
        unit = random_unit(gen=0)
        unit, peft_model = train_and_eval_unit(
            unit, peft_model, train_inputs, train_labels,
            val_inputs, val_labels,
            n_steps=TRAIN_STEPS_PER_UNIT, lr=UNIT_LR,
        )
        population.append(unit)
        elapsed = time.time() - t_start
        mem = torch.cuda.memory_allocated() / 1e9
        print(f'  Unit {i}: {unit.summary()} ({elapsed:.0f}s, {mem:.1f}GB)')

    population.sort(key=lambda u: u.fitness, reverse=True)
    print(f'\nBest initial: {population[0].summary()}')
    print(f'Worst initial: {population[-1].summary()}')

    avg_depth = sum(u.depth for u in population) / len(population)
    avg_width = sum(u.width for u in population) / len(population)
    avg_fitness = sum(u.fitness for u in population) / len(population)
    monitor.record(avg_depth, avg_width, avg_fitness, population[0].fitness)

    # Evolution loop
    print(f'\nStarting evolution: {GENERATIONS} generations...')
    gen = 0

    for gen in range(1, GENERATIONS + 1):
        gen_start = time.time()
        replacements = 0

        for _ in range(CHILDREN_PER_GEN):
            parent_a = tournament_select(population, TOURNAMENT_K)
            parent_b = tournament_select(population, TOURNAMENT_K)

            child = fuse_hybrid(parent_a, parent_b, gen)
            if child is None:
                continue

            child = mutate(child, mutation_rate=0.3)
            child, peft_model = train_and_eval_unit(
                child, peft_model, train_inputs, train_labels,
                val_inputs, val_labels,
                n_steps=TRAIN_STEPS_PER_UNIT, lr=UNIT_LR,
            )

            tournament_idx = random.sample(range(len(population)),
                                           min(TOURNAMENT_K, len(population)))
            loser_idx = min(tournament_idx, key=lambda i: population[i].fitness)
            if child.fitness > population[loser_idx].fitness:
                population[loser_idx] = child
                replacements += 1

        population.sort(key=lambda u: u.fitness, reverse=True)
        best = population[0]

        avg_depth = sum(u.depth for u in population) / len(population)
        avg_width = sum(u.width for u in population) / len(population)
        avg_fitness = sum(u.fitness for u in population) / len(population)
        unique_archs = len(set(u.arch_key() for u in population))
        diversity = unique_archs / len(population)

        monitor.record(avg_depth, avg_width, avg_fitness, best.fitness)

        gen_time = time.time() - gen_start
        wandb.log({
            'evo/best_fitness': best.fitness,
            'evo/best_loss': best.val_loss,
            'evo/best_rank': best.rank,
            'evo/best_depth': best.depth,
            'evo/best_params': best.n_trainable,
            'evo/avg_fitness': avg_fitness,
            'evo/avg_depth': avg_depth,
            'evo/avg_width': avg_width,
            'evo/diversity': diversity,
            'evo/replacements': replacements,
            'evo/cusum_depth': monitor.cusum_depth,
            'evo/cusum_width': monitor.cusum_width,
            'evo/gelation': 1 if monitor.gelation_detected else 0,
            'evo/gen_time_s': gen_time,
        }, step=gen)

        if gen % 5 == 0 or gen == 1 or monitor.gelation_detected:
            ppl = math.exp(min(best.val_loss, 20.0))
            gel_str = f' ** GELATION at gen {monitor.gelation_step} **' if monitor.gelation_detected else ''
            elapsed = time.time() - t_start
            mem = torch.cuda.memory_allocated() / 1e9
            print(f'Gen {gen:3d} | best: loss={best.val_loss:.4f} ppl={ppl:.1f} '
                  f'r={best.rank} d={best.depth} | '
                  f'div={diversity:.2f} repl={replacements} | '
                  f'{gen_time:.0f}s (total {elapsed:.0f}s) {mem:.1f}GB{gel_str}')

        if (monitor.gelation_detected and
            monitor.gelation_step is not None and
            gen >= monitor.gelation_step + GELATION_PATIENCE):
            print(f'\nEarly stopping: {GELATION_PATIENCE} gens after gelation.')
            break

    elapsed = time.time() - t_start
    print(f'\nEvolution complete in {elapsed:.0f}s ({elapsed/60:.1f} min)')
    print(f'Best unit: {population[0].summary()}')
    if monitor.gelation_detected:
        print(f'Gelation detected at generation {monitor.gelation_step}')

    wandb.finish()

    # ── Results ─────────────────────────────────────────────────────────────
    best = population[0]
    best_ppl = math.exp(min(best.val_loss, 20.0))

    print('\n' + '=' * 70)
    print('SYMBIOGENESIS RESULTS — Gemma-3-270M LoRA Evolution')
    print('=' * 70)
    print(f'\nBase model: {MODEL_NAME} ({n_base_params/1e6:.0f}M params, frozen)')
    print(f'Base loss: {base_loss:.4f} (ppl={base_ppl:.1f})')
    print(f'\nBest adapter:')
    print(f'  Rank: {best.rank}')
    print(f'  Targets: {list(best.target_modules)}')
    print(f'  Trainable params: {best.n_trainable:,}')
    print(f'  Val loss: {best.val_loss:.4f}')
    print(f'  Val PPL: {best_ppl:.1f}')
    print(f'  Fitness: {best.fitness:.4f}')
    print(f'  Generation: {best.generation}')

    print(f'\nGelation detected: {monitor.gelation_detected}')
    if monitor.gelation_detected:
        print(f'  At generation: {monitor.gelation_step}')

    print(f'\nFinal population:')
    print(f'{"Unit":<5} {"Rank":>5} {"Depth":>6} {"Targets":<40} {"Loss":>8} {"PPL":>8} {"Params":>10}')
    print('-' * 85)
    for i, u in enumerate(population):
        ppl = math.exp(min(u.val_loss, 20.0))
        targets = ','.join(t.replace('_proj', '') for t in u.target_modules)
        print(f'{i:<5} {u.rank:>5} {u.depth:>6} {targets:<40} '
              f'{u.val_loss:>8.4f} {ppl:>8.1f} {u.n_trainable:>10,}')

    rank_dist = Counter(u.rank for u in population)
    print(f'\nRank distribution: {dict(sorted(rank_dist.items()))}')
    target_counts = Counter()
    for u in population:
        for t in u.target_modules:
            target_counts[t] += 1
    print(f'Target frequency:')
    for t, c in sorted(target_counts.items(), key=lambda x: -x[1]):
        print(f'  {t}: {c}/{len(population)} ({c/len(population)*100:.0f}%)')

    unique = len(set(u.arch_key() for u in population))
    print(f'\nArchitecture diversity: {unique}/{len(population)} unique configs')

    # ── Extended fine-tune ──────────────────────────────────────────────────
    # Need a clean model for extended training
    # Unwrap back to base model
    if hasattr(peft_model, 'base_model'):
        # Get the unwrapped base model
        unwrapped = peft_model.get_base_model()
    else:
        unwrapped = peft_model

    print(f'\n{"=" * 70}')
    print(f'Extended fine-tuning: r={best.rank}, targets={list(best.target_modules)}')
    print(f'Steps: {EXTENDED_STEPS}, LR: {EXTENDED_LR}')

    best_lora = make_lora_config(best)
    # Create fresh PeftModel for extended training
    ext_model = get_peft_model(unwrapped, best_lora, adapter_name='best_ext')
    ext_model.train()

    trainable_params = [p for p in ext_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f'Trainable params: {n_trainable:,}')

    ext_run = wandb.init(
        project='symbiogenesis',
        name=f'symbiogenesis-gemma-270m-best-r{best.rank}-rtx3060',
        config={
            'method': 'extended_finetune',
            'base_model': MODEL_NAME,
            'lora_rank': best.rank,
            'lora_targets': list(best.target_modules),
            'n_trainable': n_trainable,
            'evo_val_loss': best.val_loss,
            'evo_generation': best.generation,
            'gelation_step': monitor.gelation_step,
            'n_steps': EXTENDED_STEPS,
            'lr': EXTENDED_LR,
            'batch_size': EXTENDED_BATCH,
            'grad_accum': EXTENDED_GRAD_ACCUM,
        },
        tags=['symbiogenesis', 'lora', 'gemma-270m', 'extended-finetune', 'philosophy'],
        reinit='finish_previous',
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=EXTENDED_LR, weight_decay=0.01)

    def lr_lambda_ext(step):
        if step < EXTENDED_WARMUP:
            return (step + 1) / max(EXTENDED_WARMUP, 1)
        progress = (step - EXTENDED_WARMUP) / max(EXTENDED_STEPS - EXTENDED_WARMUP, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_ext)

    n_train = len(train_inputs)
    best_ext_loss = float('inf')
    step = 0
    accum_step = 0
    t_start = time.time()

    while step < EXTENDED_STEPS:
        perm = torch.randperm(n_train)
        for i in range(0, n_train, EXTENDED_BATCH):
            if step >= EXTENDED_STEPS:
                break
            idx = perm[i:i+EXTENDED_BATCH]
            if len(idx) < EXTENDED_BATCH:
                continue
            batch_in = train_inputs[idx].to(device)
            batch_tgt = train_labels[idx].to(device)

            with torch.amp.autocast('cuda', dtype=DTYPE):
                out = ext_model(batch_in)
                logits = out.logits
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.reshape(B*T, V), batch_tgt.reshape(B*T))
                loss = loss / EXTENDED_GRAD_ACCUM

            loss.backward()
            del out, logits, loss
            accum_step += 1

            if accum_step % EXTENDED_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if step % 50 == 0:
                    wandb.log({'train/lr': scheduler.get_last_lr()[0]}, step=step)

                if step > 0 and step % EVAL_EVERY == 0:
                    val_loss, val_ppl = evaluate_lm(ext_model, val_inputs, val_labels)
                    wandb.log({'val/loss': val_loss, 'val/ppl': val_ppl}, step=step)
                    marker = ' ** BEST **' if val_loss < best_ext_loss else ''
                    if val_loss < best_ext_loss:
                        best_ext_loss = val_loss
                        ext_model.save_pretrained('best_lora_adapter')
                    elapsed = time.time() - t_start
                    print(f'  [step {step:5d}] val={val_loss:.4f} ppl={val_ppl:.1f} '
                          f'lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.0f}s){marker}')
                    ext_model.train()

                step += 1

    final_loss, final_ppl = evaluate_lm(ext_model, val_inputs, val_labels)
    if final_loss < best_ext_loss:
        best_ext_loss = final_loss
        ext_model.save_pretrained('best_lora_adapter')

    elapsed = time.time() - t_start
    wandb.log({'val/final_loss': best_ext_loss,
               'val/final_ppl': math.exp(min(best_ext_loss, 20.0))})
    print(f'\nExtended fine-tune done ({elapsed:.0f}s)')
    print(f'Best: val_loss={best_ext_loss:.4f} ppl={math.exp(min(best_ext_loss, 20.0)):.1f}')
    print(f'Baseline: val_loss={base_loss:.4f} ppl={base_ppl:.1f}')
    print(f'Improvement: {base_loss - best_ext_loss:.4f} loss, '
          f'{base_ppl - math.exp(min(best_ext_loss, 20.0)):.1f} PPL')
    wandb.finish()

    # ── Upload to HF ───────────────────────────────────────────────────────
    run_tag = datetime.now().strftime('%Y%m%d')
    HF_REPO = f'LisaMegaWatts/SymbioSLM-ouroboros-lora-{run_tag}'
    print(f'\nUploading to NEW repo: {HF_REPO}')

    os.makedirs('upload', exist_ok=True)
    meta = {
        'method': 'symbiogenesis_lora_evolution',
        'base_model': MODEL_NAME,
        'base_params': n_base_params,
        'base_loss': base_loss,
        'base_ppl': base_ppl,
        'best_adapter': {
            'rank': best.rank,
            'target_modules': list(best.target_modules),
            'alpha': best.alpha,
            'dropout': best.dropout,
            'n_trainable': n_trainable,
            'evo_val_loss': best.val_loss,
            'evo_generation': best.generation,
            'extended_val_loss': best_ext_loss,
            'extended_ppl': math.exp(min(best_ext_loss, 20.0)),
        },
        'evolution': {
            'population_size': POP_SIZE,
            'generations': gen,
            'gelation_detected': monitor.gelation_detected,
            'gelation_step': monitor.gelation_step,
            'final_diversity': unique / len(population),
        },
        'corpus': 'curated_philosophy',
        'context_length': CTX,
        'github': 'https://github.com/DavinciDreams/SymbioGPT',
    }
    with open('upload/symbiogenesis_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    hf_api = HfApi()
    try:
        create_repo(HF_REPO, exist_ok=False)
        if os.path.exists('best_lora_adapter'):
            hf_api.upload_folder(
                folder_path='best_lora_adapter', repo_id=HF_REPO,
                commit_message=f'Best LoRA adapter (r={best.rank}, '
                              f'loss={best_ext_loss:.4f}, '
                              f'gelation@gen{monitor.gelation_step})',
            )
        hf_api.upload_file(
            path_or_fileobj='upload/symbiogenesis_metadata.json',
            path_in_repo='symbiogenesis_metadata.json', repo_id=HF_REPO,
            commit_message='Symbiogenesis evolution metadata',
        )
        print(f'\nUploaded to: https://huggingface.co/{HF_REPO}')
    except Exception as e:
        print(f'HF upload issue: {e}')

    print('\nDone!')
