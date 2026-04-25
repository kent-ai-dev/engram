"""
bench/run.py — Reproducible benchmark harness for Engram.

Every test in OPENMYTHOS_TRANSFER.md runs through this script.
Results are written to bench/history/<run_id>.json.

Usage:
  python bench/run.py [options]

Key flags:
  --seed INT              RNG seed (torch + python + numpy). Default: 42
  --corpus PATH ...       Training corpora. Default: dailydialog_tiny + alice
  --holdout PATH ...      Holdout corpora (never seen during training). Default: time_machine
  --epochs INT            Training epochs. Default: 2
  --batch-size INT        Batch size. Default: 128
  --context-size INT      Context window. Default: 32 (from engram_model.CONTEXT_SIZE)
  --n-ponder-train INT    Max ponder steps during training. Default: 3
  --n-ponder-eval INT     Max ponder steps during eval. Default: same as train
  --run-id STR            Override the auto-generated run ID
  --output-dir PATH       Where to write JSON. Default: bench/history

Feature flags (all off by default):
  --use-lti               LTI-stable input injection (Phase 1)
  --use-loop-idx          Loop-index sinusoidal embedding (Phase 2)
  --use-lora              Per-loop LoRA scale (Phase 4)
  --use-rope              RoPE positional encoding (Phase 5)
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chromadb

# Repo root is parent of bench/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from engram_model import (
    AttentionBlock, EngramModule, AttentionBrain,
    EMBED_DIM, CONTEXT_SIZE, N_LAYERS, NGRAM_TABLE_SIZE, SPECIAL_TOKENS,
)

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_corpus(paths: list[str]) -> str:
    text = ""
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text += f.read().lower() + "\n"
    return text


def build_sequences(text: str, context_size: int):
    """Return list of (ctx_words, target_word) tuples."""
    QUESTION_STARTERS = {"what", "how", "can", "do", "is", "tell", "why", "where",
                         "should", "did", "are", "does", "will", "would", "have"}
    ANSWER_STARTERS = {"i", "my", "yes", "no", "it", "the", "that", "we", "perhaps",
                       "of", "memory", "wisdom", "home", "friends", "family",
                       "mistakes", "beauty", "reality", "time"}

    def split_qa(words):
        if len(words) < 6 or words[0] not in QUESTION_STARTERS:
            return None
        for i in range(3, len(words) - 2):
            if words[i] in ANSWER_STARTERS:
                return (["<USER>"] + words[:i], ["<BOT>"] + words[i:])
        return None

    paragraphs = re.split(r'\n\s*\n', text)
    sequences = []
    for para in paragraphs:
        words = re.findall(r'\b\w+\b', para)
        if not words:
            continue
        qa = split_qa(words)
        if qa:
            words = qa[0] + qa[1]
        ctx = ["<START>"] * context_size
        for w in words:
            sequences.append((list(ctx[-context_size:]), w))
            ctx.append(w)
    return sequences


# ---------------------------------------------------------------------------
# Evaluation against holdout
# ---------------------------------------------------------------------------

def eval_holdout(brain: AttentionBrain, embed_cache: dict, holdout_text: str,
                 context_size: int, n_ponder_eval: int, chroma_path: str) -> dict:
    """
    Evaluate on holdout text using ChromaDB for nearest-concept lookups.
    Returns dict with eval_cosine_top1, eval_cosine_mean, eval_perp_proxy.
    """
    brain.eval()
    device = next(brain.parameters()).device
    sequences = build_sequences(holdout_text, context_size)
    if not sequences:
        return {"eval_cosine_top1": 0.0, "eval_cosine_mean": 0.0, "eval_perp_proxy": 0.0}

    # Build vocab matrix on CPU for similarity lookup (move pred to CPU for comparison)
    word_list = list(embed_cache.keys())
    word_to_idx = {w: i for i, w in enumerate(word_list)}
    vocab_matrix = torch.tensor([embed_cache[w] for w in word_list], dtype=torch.float32)
    vocab_norm = F.normalize(vocab_matrix, dim=-1)

    top1_hits = 0
    cosine_sum = 0.0
    perp_sum = 0.0
    n_eval = min(len(sequences), 500)  # cap for speed

    # Use a fixed random subsample for reproducibility
    rng = random.Random(0)
    sample = rng.sample(sequences, n_eval) if len(sequences) >= n_eval else sequences

    orig_max_ponder = brain.max_ponder
    brain.max_ponder = n_ponder_eval

    with torch.no_grad():
        for ctx_words, target_word in sample:
            if target_word not in embed_cache:
                continue
            ctx_tensors = [
                torch.tensor(embed_cache.get(w, [0.0] * EMBED_DIM), dtype=torch.float32)
                for w in ctx_words
            ]
            ctx_stack = torch.stack(ctx_tensors).unsqueeze(0).to(device)
            predicted, _ = brain(ctx_stack)
            pred_norm = F.normalize(predicted.squeeze(0).cpu(), dim=0)

            # Cosine similarities against all vocab (both on CPU)
            sims = (vocab_norm * pred_norm.unsqueeze(0)).sum(-1)

            # Top-1 accuracy
            top1_word = word_list[sims.argmax().item()]
            if top1_word == target_word:
                top1_hits += 1

            # Mean cosine sim to ground-truth
            if target_word in word_to_idx:
                gt_sim = sims[word_to_idx[target_word]].item()
                cosine_sum += gt_sim

                # Perplexity proxy: -log(softmax(sims)[ground_truth])
                log_probs = F.log_softmax(sims, dim=0)
                perp_sum += -log_probs[word_to_idx[target_word]].item()

    brain.max_ponder = orig_max_ponder

    n = max(n_eval, 1)
    return {
        "eval_cosine_top1": top1_hits / n * 100,
        "eval_cosine_mean": cosine_sum / n,
        "eval_perp_proxy": perp_sum / n,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args) -> dict:
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load corpora
    train_text = load_corpus(args.corpus)
    holdout_text = load_corpus(args.holdout)

    # Build sequences
    sequences = build_sequences(train_text, args.context_size)
    if not sequences:
        raise ValueError("No training sequences built — check corpus paths.")
    if args.max_sequences and len(sequences) > args.max_sequences:
        sequences = sequences[: args.max_sequences]

    all_words = []
    for ctx, tgt in sequences:
        all_words.extend(ctx)
        all_words.append(tgt)
    unique_words = list(dict.fromkeys(SPECIAL_TOKENS + all_words))

    embed_cache = {w: torch.randn(EMBED_DIM).tolist() for w in unique_words}
    word_to_id = {w: i for i, w in enumerate(unique_words)}

    # Build model
    brain = AttentionBrain(
        embed_dim=EMBED_DIM,
        context_size=args.context_size,
        n_layers=N_LAYERS,
        max_ponder=args.n_ponder_train,
    )
    engram = EngramModule(EMBED_DIM)
    brain.to(device)
    engram.to(device)

    all_params = list(brain.parameters()) + list(engram.parameters())
    optimizer = optim.Adam(all_params, lr=1e-3)
    total_steps = ((len(sequences) + args.batch_size - 1) // args.batch_size) * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    brain_params = sum(p.numel() for p in brain.parameters())
    engram_params = sum(p.numel() for p in engram.parameters())

    # --- Feature patches (applied if flags are set) ---
    # Phase 1: LTI, Phase 2: loop-idx, Phase 4: LoRA, Phase 5: RoPE
    # These will be implemented in their respective phases.
    # For now, flags are accepted but unused (no-ops).

    # Training loop
    train_loss_curve = []   # avg loss per 100 batches
    grad_norms = []
    ponder_hist = {}

    brain.train()
    engram.train()

    t_train_start = time.time()

    for epoch in range(args.epochs):
        random.shuffle(sequences)

        for batch_start in range(0, len(sequences), args.batch_size):
            batch = sequences[batch_start: batch_start + args.batch_size]
            ctx_word_lists = [s[0] for s in batch]
            target_words = [s[1] for s in batch]

            all_batch_words = list(
                dict.fromkeys(w for ctx_wl in ctx_word_lists for w in ctx_wl) |
                dict.fromkeys(target_words)
            )
            batch_idx = {w: i for i, w in enumerate(all_batch_words)}

            batch_embed = torch.tensor(
                [embed_cache[w] for w in all_batch_words], dtype=torch.float32
            ).to(device).requires_grad_(True)

            ctx_idx = torch.tensor(
                [[batch_idx[w] for w in cw] for cw in ctx_word_lists]
            ).to(device)
            tgt_idx = torch.tensor([batch_idx[w] for w in target_words]).to(device)

            ctx_embeds = batch_embed[ctx_idx]
            target_embeds = batch_embed[tgt_idx]

            ngram_id_seqs = [[word_to_id.get(w, 0) for w in cw[-3:]] for cw in ctx_word_lists]
            ngram_memory = engram.lookup_batch(ngram_id_seqs)

            optimizer.zero_grad()
            predicted, ponder_steps = brain(ctx_embeds, ngram_memory=ngram_memory, engram_module=engram)

            mse_loss = F.mse_loss(predicted, target_embeds)
            ponder_cost = 0.05 * ponder_steps
            cos_sim = F.cosine_similarity(predicted, ngram_memory, dim=-1).mean()
            coherence_penalty = 0.05 * (1.0 - cos_sim)
            loss = mse_loss + ponder_cost + coherence_penalty

            loss.backward()

            # Gradient norm tracking (pre-clip)
            gnorm = torch.nn.utils.clip_grad_norm_(all_params, 1.0).item()
            grad_norms.append(gnorm)

            optimizer.step()
            scheduler.step()

            # Embedding update
            if batch_embed.grad is not None:
                with torch.no_grad():
                    updated = batch_embed - 5e-4 * batch_embed.grad
                    for w, i in batch_idx.items():
                        embed_cache[w] = updated[i].tolist()

            # Track ponder steps histogram
            ps = int(ponder_steps)
            ponder_hist[ps] = ponder_hist.get(ps, 0) + 1

            # Loss curve: log every 100 batches
            global_batch = epoch * ((len(sequences) + args.batch_size - 1) // args.batch_size) + batch_start // args.batch_size
            if global_batch % 100 == 0:
                train_loss_curve.append(round(loss.item(), 6))

    t_train_end = time.time()

    # Eval
    t_eval_start = time.time()
    n_ponder_eval = args.n_ponder_eval if args.n_ponder_eval is not None else args.n_ponder_train
    eval_metrics = eval_holdout(brain, embed_cache, holdout_text, args.context_size, n_ponder_eval, chroma_path="./engram_memory")
    t_eval_end = time.time()

    # Grad norm percentiles
    if grad_norms:
        gn_t = torch.tensor(grad_norms)
        grad_norm_p50 = round(torch.quantile(gn_t, 0.50).item(), 6)
        grad_norm_p99 = round(torch.quantile(gn_t, 0.99).item(), 6)
    else:
        grad_norm_p50 = grad_norm_p99 = 0.0

    # FLOPs: analytical estimate
    flops_per_token = N_LAYERS * EMBED_DIM * args.n_ponder_train * args.context_size * 4  # rough

    result = {
        "run_id": args.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "config": {
            "corpus": args.corpus,
            "holdout": args.holdout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "context_size": args.context_size,
            "n_ponder_train": args.n_ponder_train,
            "n_ponder_eval": n_ponder_eval,
            "max_sequences": args.max_sequences,
            "use_lti": args.use_lti,
            "use_loop_idx": args.use_loop_idx,
            "use_lora": args.use_lora,
            "use_rope": args.use_rope,
        },
        "param_count": brain_params + engram_params,
        "flops_per_token": flops_per_token,
        "train_loss_curve": train_loss_curve,
        "grad_norm_p50": grad_norm_p50,
        "grad_norm_p99": grad_norm_p99,
        "ponder_steps_hist": ponder_hist,
        "wall_time_train": round(t_train_end - t_train_start, 2),
        "wall_time_eval": round(t_eval_end - t_eval_start, 2),
        **eval_metrics,
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Engram benchmark harness")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corpus", nargs="+",
                   default=["corpus/dailydialog_tiny.txt",
                            "corpus/11_alice_s_adventures_in_wonderland.txt"])
    p.add_argument("--holdout", nargs="+",
                   default=["corpus/35_the_time_machine.txt"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    p.add_argument("--n-ponder-train", type=int, default=3)
    p.add_argument("--n-ponder-eval", type=int, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="bench/history")
    p.add_argument("--max-sequences", type=int, default=None,
                   help="Cap training sequences for quick smoke tests (e.g. 2000)")
    p.add_argument("--via-modal", action="store_true",
                   help="Run training on Modal L4 GPU instead of locally")
    # Feature flags
    p.add_argument("--use-lti", action="store_true")
    p.add_argument("--use-loop-idx", action="store_true")
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--use-rope", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.via_modal:
        sys.path.insert(0, _REPO_ROOT)
        from bench.modal_bench import launch_via_modal
        if args.run_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            flags = "".join(
                f"+{f}" for f, v in [("lti", args.use_lti), ("lidx", args.use_loop_idx),
                                      ("lora", args.use_lora), ("rope", args.use_rope)] if v
            ) or "baseline"
            args.run_id = f"seed{args.seed}_{flags}_{ts}"
        return launch_via_modal(args)

    if args.run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        flags = "".join(
            f"+{f}" for f, v in [("lti", args.use_lti), ("lidx", args.use_loop_idx),
                                  ("lora", args.use_lora), ("rope", args.use_rope)] if v
        ) or "baseline"
        args.run_id = f"seed{args.seed}_{flags}_{ts}"

    print(f"[bench] run_id={args.run_id} seed={args.seed}")
    print(f"[bench] corpus={args.corpus}")
    print(f"[bench] holdout={args.holdout}")
    print(f"[bench] epochs={args.epochs} batch={args.batch_size} ponder_train={args.n_ponder_train}")

    result = train(args)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.run_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[bench] eval_cosine_top1={result['eval_cosine_top1']:.2f}%  "
          f"eval_cosine_mean={result['eval_cosine_mean']:.4f}  "
          f"grad_norm_p99={result['grad_norm_p99']:.4f}")
    print(f"[bench] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
