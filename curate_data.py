"""Curate training data for Chinchilla-optimal SymbioGPT training.

Scores paragraph-level chunks on quality heuristics and selects the
best ones up to a target token budget. This avoids overfitting on
low-quality data and ensures each token maximally benefits training.

Quality metrics:
  - Vocabulary diversity (type-token ratio)
  - Mean word length (complexity proxy)
  - Punctuation quality (proper sentence structure)
  - Alphabetic ratio (reject tables/numbers)
  - No excessive repetition (penalize repeated n-grams)

Usage:
    python curate_data.py [--target_tokens N] [--min_chunk_words N]
"""
import argparse
import collections
import logging
import os
import re
import sys
import time

import torch

sys.path.insert(0, "/home/ubuntu/Dev/symbiogenesis")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from symbiogenesis.transformer_tokenizer import BPETokenizer

logger = logging.getLogger(__name__)

SOURCE_DIR = "/home/ubuntu/Dev/buildwithbooks/text-pipeline/output"
OUTPUT_DIR = "/home/ubuntu/Dev/juliaGPT/SymbioSLM/data"
TRAIN_PATH = os.path.join(SOURCE_DIR, "train.txt")
VAL_PATH = os.path.join(SOURCE_DIR, "val.txt")
VOCAB_PATH = os.path.join(SOURCE_DIR, "vocab.json")
MERGES_PATH = os.path.join(SOURCE_DIR, "merges.txt")


def score_chunk(text: str) -> float:
    """Score a text chunk on quality heuristics. Higher = better."""
    words = text.split()
    n_words = len(words)
    if n_words < 20:
        return 0.0

    # 1. Vocabulary diversity: unique / total (type-token ratio)
    #    Cap at 200 words to avoid penalizing long passages
    sample = words[:200]
    unique = len(set(w.lower() for w in sample))
    ttr = unique / len(sample)

    # 2. Mean word length (complexity proxy, 4-8 is ideal for English prose)
    mean_wl = sum(len(w) for w in words) / n_words
    wl_score = 1.0 - abs(mean_wl - 5.5) / 5.5  # peaks at 5.5

    # 3. Alphabetic ratio (reject tables, numbers, code)
    alpha_chars = sum(1 for c in text if c.isalpha())
    total_chars = len(text)
    alpha_ratio = alpha_chars / max(total_chars, 1)
    if alpha_ratio < 0.6:
        return 0.0  # reject chunks that are mostly non-alphabetic

    # 4. Punctuation quality: proper sentences end with . ! ?
    sentences = re.split(r'[.!?]+', text)
    n_sentences = max(len([s for s in sentences if len(s.strip()) > 10]), 1)
    avg_sent_len = n_words / n_sentences
    # Ideal sentence length: 10-30 words
    sent_score = 1.0 - abs(avg_sent_len - 20) / 30
    sent_score = max(0.0, sent_score)

    # 5. Repetition penalty: count repeated 4-grams
    if n_words >= 20:
        fourgrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
        fg_counts = collections.Counter(fourgrams)
        repeated = sum(c - 1 for c in fg_counts.values() if c > 1)
        rep_penalty = min(repeated / max(len(fourgrams), 1), 0.5)
    else:
        rep_penalty = 0.0

    # 6. Dialogue penalty: excessive quotes suggest low-quality fiction
    quote_ratio = text.count('"') / max(total_chars, 1)
    dialogue_penalty = min(quote_ratio * 10, 0.3)

    # Combine scores
    score = (
        0.35 * ttr +
        0.20 * max(0, wl_score) +
        0.20 * sent_score +
        0.15 * alpha_ratio +
        0.10 * (1.0 - rep_penalty) -
        dialogue_penalty
    )

    return max(0.0, score)


def chunk_file(path: str, lines_per_chunk: int = 15) -> list:
    """Split a text file into fixed-size line groups.

    The corpus is one sentence per line with no paragraph breaks.
    Groups consecutive lines into chunks for quality scoring.
    """
    logger.info("Reading %s...", path)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    logger.info("Read %d lines", len(lines))

    chunks = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk = " ".join(line.strip() for line in lines[i:i + lines_per_chunk] if line.strip())
        if chunk:
            chunks.append(chunk)

    logger.info("Split into %d chunks (%d lines/chunk, avg %d chars)", len(chunks),
                lines_per_chunk,
                sum(len(c) for c in chunks) // max(len(chunks), 1))
    return chunks


def curate(args):
    """Main curation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    target_tokens = args.target_tokens
    target_chars = target_tokens * 4  # rough estimate

    logger.info("Target: %s tokens (~%s chars)", f"{target_tokens:,}", f"{target_chars:,}")

    # Score training data chunks
    t0 = time.time()
    chunks = chunk_file(TRAIN_PATH, lines_per_chunk=args.lines_per_chunk)

    logger.info("Scoring %d chunks...", len(chunks))
    scored = []
    for i, chunk in enumerate(chunks):
        s = score_chunk(chunk)
        if s > 0:
            scored.append((s, len(chunk), chunk))
        if (i + 1) % 50000 == 0:
            logger.info("  scored %d/%d chunks...", i + 1, len(chunks))

    logger.info("Scored %d chunks in %.1fs (%d rejected with score=0)",
                len(scored), time.time() - t0, len(chunks) - len(scored))

    # Sort by quality (best first)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Show score distribution
    scores = [s for s, _, _ in scored]
    logger.info("Score distribution: min=%.3f, median=%.3f, p75=%.3f, p90=%.3f, max=%.3f",
                scores[-1],
                scores[len(scores) // 2],
                scores[len(scores) // 4],
                scores[len(scores) // 10],
                scores[0])

    # Select top chunks up to target
    selected_text = []
    selected_chars = 0
    for score, clen, chunk in scored:
        if selected_chars >= target_chars:
            break
        selected_text.append(chunk)
        selected_chars += clen

    logger.info("Selected %d/%d chunks (%s chars, est ~%s tokens)",
                len(selected_text), len(scored),
                f"{selected_chars:,}", f"{selected_chars // 4:,}")

    # Write curated text
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    curated_path = os.path.join(OUTPUT_DIR, "train_curated.txt")
    with open(curated_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(selected_text))
    logger.info("Curated text saved: %s", curated_path)

    # Tokenize curated text
    logger.info("Tokenizing curated text...")
    tokenizer = BPETokenizer.from_files(VOCAB_PATH, MERGES_PATH)
    t1 = time.time()
    curated_text = "\n\n".join(selected_text)
    tokens = tokenizer.encode(curated_text)
    logger.info("Tokenized: %s tokens in %.1fs", f"{len(tokens):,}", time.time() - t1)

    # Cache tokens
    cache_path = os.path.join(OUTPUT_DIR, "train_curated.txt.tokens.pt")
    torch.save(torch.tensor(tokens, dtype=torch.int32), cache_path)
    logger.info("Cached tokens: %s", cache_path)

    # Also tokenize and cache val set (smaller, fast enough)
    logger.info("Tokenizing validation set...")
    val_cache = os.path.join(OUTPUT_DIR, "val.txt.tokens.pt")
    if not os.path.exists(val_cache):
        with open(VAL_PATH) as f:
            val_text = f.read()
        val_tokens = tokenizer.encode(val_text)
        torch.save(torch.tensor(val_tokens, dtype=torch.int32), val_cache)
        logger.info("Val tokens cached: %s (%s tokens)", val_cache, f"{len(val_tokens):,}")
    else:
        logger.info("Val cache exists: %s", val_cache)

    # Show quality samples
    logger.info("\n=== Top 3 highest-scoring chunks ===")
    for i, (s, _, chunk) in enumerate(scored[:3]):
        logger.info("Score=%.3f: %s...", s, chunk[:200].replace("\n", " "))

    logger.info("\n=== 3 lowest-scoring (still included) chunks ===")
    cutoff_idx = len(selected_text) - 1
    for i in range(max(0, cutoff_idx - 2), cutoff_idx + 1):
        s, _, chunk = scored[i]
        logger.info("Score=%.3f: %s...", s, chunk[:200].replace("\n", " "))

    logger.info("Done! Use --train_path %s for training", curated_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate training data for SymbioGPT")
    parser.add_argument("--target_tokens", type=int, default=220_000_000,
                        help="Target token count (Chinchilla optimal for 11M = ~220M)")
    parser.add_argument("--lines_per_chunk", type=int, default=15,
                        help="Lines per chunk for scoring")
    parser.add_argument("--min_chunk_words", type=int, default=20,
                        help="Minimum words per chunk")
    args = parser.parse_args()
    curate(args)
