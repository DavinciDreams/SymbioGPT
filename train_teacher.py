"""Train SymbioGPT-10M teacher model for symbiotic distillation.

Trains a multi-organelle SymbioGPT model on the philosophy corpus.
This model later serves as the teacher for evolutionary architecture search.

Usage:
    python train_teacher.py [--steps N] [--batch_size N] [--lr F] [--beta F]

Requirements (CLAUDE.md mandated):
    - W&B logging active (WANDB_API_KEY set)
    - HF repo exists (LisaMegaWatts/SymbioGPT-10M)
    - Checkpoint directory set
"""
import argparse
import logging
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ubuntu/Dev/symbiogenesis")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from symbio_model import (
    SymbioConfig,
    SymbioGPT,
    complexity_penalty,
    compute_gate_entropy,
    compute_symbio_params,
)
from symbiogenesis.transformer_tokenizer import BPETokenizer

logger = logging.getLogger(__name__)

# ── Paths ──
DATA_DIR = "/home/ubuntu/Dev/buildwithbooks/text-pipeline/output"
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
VAL_PATH = os.path.join(DATA_DIR, "val.txt")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
MERGES_PATH = os.path.join(DATA_DIR, "merges.txt")
CHECKPOINT_DIR = "/home/ubuntu/Dev/juliaGPT/SymbioSLM/checkpoints/teacher"
HF_REPO = "LisaMegaWatts/SymbioGPT-10M"

# ── 10M Teacher Config ──
TEACHER_CONFIG = SymbioConfig(
    d_model=320,
    n_layers=8,
    n_heads=5,
    head_dim=64,
    ffn_mult=4,
    context_length=256,
    vocab_size=2000,
    weight_tying=True,
    organelles=("causal_conv", "monarch", "long_conv", "attention"),
    conv_kernel_size=4,
    n_monarch_heads=1,
    gate_temperature_init=1.0,
    free_energy_beta=0.001,
)


def load_data(tokenizer: BPETokenizer, context_length: int, max_tokens: int = 0):
    """Load and tokenize training/validation data.

    Caches tokenized tensors as .pt files next to the source text so
    subsequent runs skip the slow BPE tokenization step.

    Returns (train_inputs, train_labels, val_inputs, val_labels)
    as tensors of shape (N, context_length).
    """
    cache_dir = "/home/ubuntu/Dev/juliaGPT/SymbioSLM/data"
    os.makedirs(cache_dir, exist_ok=True)

    def _load_file(path, max_tok=0):
        # Check for cached tokens first (stored in SymbioSLM/data/)
        basename = os.path.basename(path)
        cache_path = os.path.join(cache_dir, basename + ".tokens.pt")
        if os.path.exists(cache_path):
            mtime_txt = os.path.getmtime(path)
            mtime_cache = os.path.getmtime(cache_path)
            if mtime_cache > mtime_txt:
                logger.info("Loading cached tokens: %s", cache_path)
                tokens = torch.load(cache_path, weights_only=True).tolist()
                if max_tok > 0:
                    tokens = tokens[:max_tok]
                logger.info("Loaded %d tokens from cache", len(tokens))
                return tokens

        logger.info("Tokenizing %s (this may take a while)...", path)
        if max_tok > 0:
            char_limit = max_tok * 4
            with open(path, "r", encoding="utf-8") as f:
                text = f.read(char_limit)
            logger.info("Read %d chars (limited to ~%d tokens)", len(text), max_tok)
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        tokens = tokenizer.encode(text)
        logger.info("Tokenized %s: %d tokens", path, len(tokens))

        # Cache for next time
        logger.info("Caching tokens to %s...", cache_path)
        torch.save(torch.tensor(tokens, dtype=torch.int32), cache_path)
        return tokens

    train_tokens = _load_file(TRAIN_PATH, max_tokens)
    val_tokens = _load_file(VAL_PATH, max_tokens // 10 if max_tokens > 0 else 0)

    def _chunk(tokens, seq_len):
        n = len(tokens) // (seq_len + 1)
        tokens = tokens[: n * (seq_len + 1)]
        data = torch.tensor(tokens, dtype=torch.long).reshape(n, seq_len + 1)
        return data[:, :-1], data[:, 1:]

    train_inputs, train_labels = _chunk(train_tokens, context_length)
    val_inputs, val_labels = _chunk(val_tokens, context_length)

    total_tokens = len(train_inputs) * context_length
    logger.info(
        "Data: %d train seqs (%s tokens), %d val seqs (context=%d)",
        len(train_inputs), f"{total_tokens:,}", len(val_inputs), context_length,
    )
    return train_inputs, train_labels, val_inputs, val_labels


def evaluate(model, val_inputs, val_labels, batch_size, device, amp_dtype=None):
    """Compute validation loss and perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(val_inputs), batch_size):
            batch_in = val_inputs[i : i + batch_size].to(device)
            batch_tgt = val_labels[i : i + batch_size].to(device)
            with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
                logits = model(batch_in)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.float().reshape(B * T, V),
                batch_tgt.reshape(B * T),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += B * T

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def preflight_check(config: SymbioConfig):
    """Mandatory pre-training checks per CLAUDE.md."""
    import wandb

    # 1. W&B
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY not set. W&B logging is mandatory.")

    # 2. Checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logger.info("Checkpoint dir: %s", CHECKPOINT_DIR)

    # 3. HF repo
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.repo_info(HF_REPO)
        logger.info("HF repo verified: %s", HF_REPO)
    except Exception as e:
        logger.warning("HF repo check failed: %s (will create later)", e)

    # 4. Config saved
    config_path = os.path.join(CHECKPOINT_DIR, "config.txt")
    with open(config_path, "w") as f:
        f.write(str(config))
    logger.info("Config saved: %s", config_path)

    # 5. Init W&B
    params = compute_symbio_params(config)
    run = wandb.init(
        project="symbiogenesis",
        name="symbio-teacher-10m",
        config={
            "architecture": "SymbioGPT",
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "organelles": list(config.organelles),
            "n_organelles": config.n_organelles,
            "n_monarch_heads": config.n_monarch_heads,
            "conv_kernel_size": config.conv_kernel_size,
            "ffn_mult": config.ffn_mult,
            "context_length": config.context_length,
            "vocab_size": config.vocab_size,
            "weight_tying": config.weight_tying,
            "free_energy_beta": config.free_energy_beta,
            "total_params": params,
        },
        tags=["teacher", "symbio", "10m", "philosophy"],
    )
    logger.info("W&B run: %s", run.url)
    return run


def train(args):
    """Main training loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    global TRAIN_PATH
    config = TEACHER_CONFIG

    # Override from CLI
    if args.train_path:
        TRAIN_PATH = args.train_path
        logger.info("Using custom train path: %s", TRAIN_PATH)
    if args.beta is not None:
        from dataclasses import replace
        config = replace(config, free_energy_beta=args.beta)

    total_params = compute_symbio_params(config)
    logger.info(
        "SymbioGPT-10M: d=%d, L=%d, organelles=%s, params=%s",
        config.d_model, config.n_layers, config.organelles, f"{total_params:,}",
    )

    # Pre-flight
    wandb_run = preflight_check(config)
    import wandb

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Data
    tokenizer = BPETokenizer.from_files(VOCAB_PATH, MERGES_PATH)
    logger.info("Tokenizer: vocab_size=%d", tokenizer.vocab_size)

    train_inputs, train_labels, val_inputs, val_labels = load_data(
        tokenizer, config.context_length, max_tokens=args.max_tokens,
    )

    # Model
    model = SymbioGPT(config).to(device)
    actual_params = sum(p.numel() for p in model.parameters())
    logger.info("Model created: %s actual params", f"{actual_params:,}")
    wandb.config.update({"actual_params": actual_params})

    # AMP (mixed precision) setup
    amp_dtype = None
    if device.type == "cuda" and args.precision != "fp32":
        if args.precision == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            logger.info("Using BF16 mixed precision")
        else:
            amp_dtype = torch.float16
            logger.info("Using FP16 mixed precision")
    else:
        logger.info("Using FP32 (no mixed precision)")

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # torch.compile (PyTorch 2.0+)
    if args.compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
        logger.info("torch.compile done")

    # Optimizer
    batch_size = args.batch_size
    grad_accum_steps = args.grad_accum
    effective_batch = batch_size * grad_accum_steps
    total_steps = args.steps
    base_lr = args.lr
    warmup_steps = min(total_steps // 10, 500)
    beta = config.free_energy_beta

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    wandb.config.update({
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": effective_batch,
        "total_steps": total_steps,
        "base_lr": base_lr,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "precision": args.precision,
        "torch_compile": args.compile,
    })

    # Training loop
    n_train = len(train_inputs)
    step = 0
    best_val_loss = float("inf")
    t_start = time.time()

    logger.info(
        "Starting training: %d steps, batch=%d×%d=%d, lr=%.1e, beta=%.4f, precision=%s",
        total_steps, batch_size, grad_accum_steps, effective_batch,
        base_lr, beta, args.precision,
    )

    model.train()
    accum_ce = 0.0
    accum_fe = 0.0
    micro_step = 0

    while step < total_steps:
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            if step >= total_steps:
                break

            idx = perm[i : i + batch_size]
            batch_in = train_inputs[idx].to(device)
            batch_tgt = train_labels[idx].to(device)

            with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
                logits = model(batch_in)
                B, T, V = logits.shape
                ce_loss = F.cross_entropy(
                    logits.reshape(B * T, V),
                    batch_tgt.reshape(B * T),
                )

                if beta > 0:
                    fe_penalty = complexity_penalty(model)
                    loss = (ce_loss + beta * fe_penalty) / grad_accum_steps
                else:
                    loss = ce_loss / grad_accum_steps
                    fe_penalty = torch.tensor(0.0)

            scaler.scale(loss).backward()
            accum_ce += ce_loss.item()
            accum_fe += fe_penalty.item() if isinstance(fe_penalty, torch.Tensor) else 0.0
            micro_step += 1

            if micro_step < grad_accum_steps:
                continue

            # Gradient step (after accumulation)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            avg_ce = accum_ce / grad_accum_steps
            avg_fe = accum_fe / grad_accum_steps
            accum_ce = 0.0
            accum_fe = 0.0
            micro_step = 0

            # Logging
            if step % 50 == 0:
                elapsed = time.time() - t_start
                tokens_per_sec = (step + 1) * effective_batch * config.context_length / max(elapsed, 1)
                gate_entropy = compute_gate_entropy(model)
                lr_now = scheduler.get_last_lr()[0]

                wandb.log({
                    "train/ce_loss": avg_ce,
                    "train/free_energy": avg_fe,
                    "train/total_loss": avg_ce + beta * avg_fe,
                    "train/gate_entropy": gate_entropy,
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/lr": lr_now,
                    "train/tokens_per_sec": tokens_per_sec,
                }, step=step)

                if step % 200 == 0:
                    logger.info(
                        "[step %5d/%d] CE=%.4f FE=%.4f GateH=%.3f LR=%.2e tok/s=%.0f elapsed=%.0fs",
                        step, total_steps, avg_ce, avg_fe,
                        gate_entropy, lr_now, tokens_per_sec, elapsed,
                    )

            # Validation
            if step > 0 and step % 500 == 0:
                val_loss, val_ppl = evaluate(
                    model, val_inputs, val_labels, batch_size, device, amp_dtype,
                )
                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": val_ppl,
                }, step=step)
                logger.info(
                    "[step %5d] val_loss=%.4f val_ppl=%.1f %s",
                    step, val_loss, val_ppl,
                    "** NEW BEST **" if val_loss < best_val_loss else "",
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(CHECKPOINT_DIR, "symbio_best.pt")
                    torch.save(model.state_dict(), best_path)
                model.train()

            # Checkpoint
            if step > 0 and step % 2000 == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"symbio_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                logger.info("Checkpoint: %s", ckpt_path)

            step += 1

    # Final eval
    val_loss, val_ppl = evaluate(model, val_inputs, val_labels, batch_size, device)
    logger.info("Final: val_loss=%.4f, val_ppl=%.1f", val_loss, val_ppl)
    wandb.log({"val/final_loss": val_loss, "val/final_ppl": val_ppl})

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "symbio_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info("Final model saved: %s", final_path)

    # Log gate specialization
    gate_weights = model.get_gate_weights()
    organelle_names = list(config.organelles)
    for i, w in enumerate(gate_weights):
        mean_w = w.mean(dim=1)
        for j, name in enumerate(organelle_names):
            wandb.summary[f"gate/layer{i}/{name}"] = mean_w[j].item()
        logger.info(
            "Gate layer %d: %s",
            i,
            " | ".join(f"{name}={mean_w[j]:.3f}" for j, name in enumerate(organelle_names)),
        )

    # Upload to HuggingFace
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=final_path,
            path_in_repo="symbio_final.pt",
            repo_id=HF_REPO,
            commit_message="Upload SymbioGPT-10M teacher (PyTorch)",
        )
        api.upload_file(
            path_or_fileobj=os.path.join(CHECKPOINT_DIR, "config.txt"),
            path_in_repo="config.txt",
            repo_id=HF_REPO,
            commit_message="Upload SymbioGPT config",
        )
        logger.info("Uploaded to HF: %s", HF_REPO)
    except Exception as e:
        logger.warning("HF upload failed: %s", e)

    wandb.finish()
    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train SymbioGPT-10M teacher")
    parser.add_argument("--steps", type=int, default=27_000, help="Training steps (default: ~Chinchilla optimal for 11M)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=None, help="Free energy beta (overrides config)")
    parser.add_argument("--max_tokens", type=int, default=0, help="Max tokens to load (0=all)")
    parser.add_argument("--train_path", type=str, default="", help="Override train.txt path (e.g. curated)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                        help="Training precision (default: fp16)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
