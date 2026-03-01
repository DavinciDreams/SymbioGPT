#!/usr/bin/env python3
"""Extended fine-tune of the best adapter from symbiogenesis evolution.

Winner: r=44, all 7 targets, val_loss=4.1451, PPL=63.1
"""

import os, math, time, json, gc
from datetime import datetime

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import wandb

# ── Config ──────────────────────────────────────────────────────────────────
DATA_REPO = 'LisaMegaWatts/SymbioGPT-10M'
MODEL_NAME = 'LisaMegaWatts/Ouroboros-1MContext-Gemma-270m'
CTX = 512

# Best adapter from evolution (gelation at gen 7)
BEST_RANK = 44
BEST_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
BEST_ALPHA = 88  # 2 * rank
EVO_VAL_LOSS = 4.1451
GELATION_STEP = 7

EXTENDED_STEPS = 2000
EXTENDED_LR = 2e-4
EXTENDED_BATCH = 2
EXTENDED_GRAD_ACCUM = 4
EXTENDED_WARMUP = 100
EVAL_EVERY = 250
EVAL_BATCH = 4

# ── GPU + dtype ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if 'rtx' in gpu_name.lower() or 'a100' in gpu_name.lower() else torch.float16
print(f"GPU: {gpu_name}, dtype: {DTYPE}")

# ── Login ───────────────────────────────────────────────────────────────────
wandb.login()
from huggingface_hub import login as hf_login
hf_login(token=os.environ.get("HF_TOKEN"), add_to_git_credential=False)

# ── Data ────────────────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
hf_hub_download(repo_id=DATA_REPO, filename='data/train_curated_sample.txt', local_dir='.')
hf_hub_download(repo_id=DATA_REPO, filename='data/val_sample.txt', local_dir='.')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open('data/train_curated_sample.txt', 'r') as f:
    train_ids = tokenizer.encode(f.read(), add_special_tokens=False)
with open('data/val_sample.txt', 'r') as f:
    val_ids = tokenizer.encode(f.read(), add_special_tokens=False)

def chunk_tokens(token_ids, seq_len):
    n = len(token_ids) // (seq_len + 1)
    token_ids = token_ids[:n * (seq_len + 1)]
    data = torch.tensor(token_ids, dtype=torch.long).reshape(n, seq_len + 1)
    return data[:, :-1], data[:, 1:]

train_inputs, train_labels = chunk_tokens(train_ids, CTX)
val_inputs, val_labels = chunk_tokens(val_ids, CTX)
print(f'Train: {len(train_inputs):,} seqs, Val: {len(val_inputs):,} seqs')
del train_ids, val_ids

# ── Load model + apply best LoRA ────────────────────────────────────────────
print(f'Loading {MODEL_NAME}...')
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(device)
base_model.eval()
for p in base_model.parameters():
    p.requires_grad = False

n_base_params = sum(p.numel() for p in base_model.parameters())

# Baseline eval
print('Baseline eval...')
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
print(f'Baseline: loss={base_loss:.4f} ppl={base_ppl:.1f}')

# Apply best LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=BEST_RANK,
    lora_alpha=BEST_ALPHA, target_modules=BEST_TARGETS,
    lora_dropout=0.0, bias='none',
)
model = get_peft_model(base_model, lora_config)
model.train()

trainable_params = [p for p in model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable_params)
print(f'LoRA trainable: {n_trainable:,} / {n_base_params:,} ({n_trainable/n_base_params*100:.2f}%)')

# ── Evaluate helper ─────────────────────────────────────────────────────────
def evaluate_lm(model):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = min(len(val_inputs) // EVAL_BATCH, 32)
    with torch.no_grad():
        for i in range(n_batches):
            batch_in = val_inputs[i*EVAL_BATCH:(i+1)*EVAL_BATCH].to(device)
            batch_tgt = val_labels[i*EVAL_BATCH:(i+1)*EVAL_BATCH].to(device)
            out = model(batch_in)
            logits = out.logits.float()
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), batch_tgt.reshape(B*T), reduction='sum')
            total_loss += loss.item()
            total_tokens += B * T
            del out, logits, loss
    avg = total_loss / max(total_tokens, 1)
    return avg, math.exp(min(avg, 20.0))

# ── Train ───────────────────────────────────────────────────────────────────
run = wandb.init(
    project='symbiogenesis',
    name=f'symbiogenesis-gemma-270m-best-r{BEST_RANK}-extended',
    config={
        'method': 'extended_finetune',
        'base_model': MODEL_NAME,
        'lora_rank': BEST_RANK,
        'lora_targets': BEST_TARGETS,
        'n_trainable': n_trainable,
        'evo_val_loss': EVO_VAL_LOSS,
        'gelation_step': GELATION_STEP,
        'n_steps': EXTENDED_STEPS,
        'lr': EXTENDED_LR,
        'batch_size': EXTENDED_BATCH,
        'grad_accum': EXTENDED_GRAD_ACCUM,
    },
    tags=['symbiogenesis', 'lora', 'gemma-270m', 'extended-finetune', 'philosophy'],
    reinit='finish_previous',
)

optimizer = torch.optim.AdamW(trainable_params, lr=EXTENDED_LR, weight_decay=0.01)

def lr_lambda(step):
    if step < EXTENDED_WARMUP:
        return (step + 1) / max(EXTENDED_WARMUP, 1)
    progress = (step - EXTENDED_WARMUP) / max(EXTENDED_STEPS - EXTENDED_WARMUP, 1)
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

best_ext_loss = float('inf')
step = 0
accum_step = 0
n_train = len(train_inputs)
t_start = time.time()

print(f'\nExtended fine-tuning: {EXTENDED_STEPS} steps...')

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
            out = model(batch_in)
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
                val_loss, val_ppl = evaluate_lm(model)
                wandb.log({'val/loss': val_loss, 'val/ppl': val_ppl}, step=step)
                marker = ' ** BEST **' if val_loss < best_ext_loss else ''
                if val_loss < best_ext_loss:
                    best_ext_loss = val_loss
                    model.save_pretrained('best_lora_adapter')
                elapsed = time.time() - t_start
                print(f'  [step {step:5d}] val={val_loss:.4f} ppl={val_ppl:.1f} '
                      f'lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.0f}s){marker}')
                model.train()

            step += 1

# Final eval
final_loss, final_ppl = evaluate_lm(model)
if final_loss < best_ext_loss:
    best_ext_loss = final_loss
    model.save_pretrained('best_lora_adapter')

elapsed = time.time() - t_start
wandb.log({'val/final_loss': best_ext_loss,
           'val/final_ppl': math.exp(min(best_ext_loss, 20.0))})
print(f'\nExtended fine-tune done ({elapsed:.0f}s)')
print(f'Best: val_loss={best_ext_loss:.4f} ppl={math.exp(min(best_ext_loss, 20.0)):.1f}')
print(f'Baseline: val_loss={base_loss:.4f} ppl={base_ppl:.1f}')
print(f'Improvement: {base_loss - best_ext_loss:.4f} loss, '
      f'{base_ppl - math.exp(min(best_ext_loss, 20.0)):.1f} PPL')
wandb.finish()

# ── Upload to HF ───────────────────────────────────────────────────────────
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
        'rank': BEST_RANK,
        'target_modules': BEST_TARGETS,
        'alpha': BEST_ALPHA,
        'n_trainable': n_trainable,
        'evo_val_loss': EVO_VAL_LOSS,
        'extended_val_loss': best_ext_loss,
        'extended_ppl': math.exp(min(best_ext_loss, 20.0)),
    },
    'evolution': {
        'gelation_step': GELATION_STEP,
        'population_converged_to': 'all 7 targets, rank 26-58',
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
            commit_message=f'Best LoRA adapter (r={BEST_RANK}, loss={best_ext_loss:.4f}, gelation@gen{GELATION_STEP})',
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
