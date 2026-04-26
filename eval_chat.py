"""
eval_chat.py — Human-judgement evaluation harness.

Loads the trained model + ChromaDB vocab, runs a fixed set of conversational
prompts, prints a clean transcript for human review. No online learning,
no episodic-memory contamination — pure evaluation of what was trained.

Usage:
    uv run eval_chat.py                    # default settings
    uv run eval_chat.py --temperature 0.7  # less random
    uv run eval_chat.py --gen-steps 20     # longer replies
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
import chromadb

from engram_model import (
    EngramModule, AttentionBrain, EMBED_DIM, CONTEXT_SIZE, N_LAYERS,
)

CHROMA_PATH = "./engram_memory"
WEIGHTS_PATH = "./engram_weights.pth"
ENGRAM_PATH = "./engram_memory_module.pth"
W2ID_PATH = "./engram_word_to_id.pth"

# Three buckets so judgement is structured:
#   greetings:  trivially easy — model should at least produce dialog-shaped text
#   chitchat:   common patterns from dailydialog corpus
#   harder:     reasoning / world knowledge — likely to fail; tells us the ceiling
PROMPT_BUCKETS = {
    "greetings": [
        "hello",
        "hi how are you",
        "good morning",
        "what is your name",
        "nice to meet you",
    ],
    "chitchat": [
        "what do you like to do",
        "tell me about yourself",
        "how was your day",
        "what are your hobbies",
        "do you have any friends",
        "what is your favorite food",
    ],
    "harder": [
        "what is the capital of france",
        "tell me a story",
        "why is the sky blue",
        "can you help me with math",
        "what do you think about love",
    ],
}


def nearest_words(predicted_t, vocab_matrix_norm, word_list, word_to_idx, n, penalty):
    """Cosine-similarity nearest-word search using pre-normalized vocab matrix."""
    pred_norm = F.normalize(predicted_t, dim=0)
    sims = vocab_matrix_norm @ pred_norm
    if penalty:
        for w, pen in penalty.items():
            if w in word_to_idx:
                sims[word_to_idx[w]] -= pen
    k = min(n, len(word_list))
    top_sims, top_idx = torch.topk(sims, k, largest=True)
    return [(word_list[i.item()], top_sims[j].item()) for j, i in enumerate(top_idx)]


def generate_reply(brain, engram_module, word_to_id, embed_cache,
                   word_list, vocab_matrix_norm, word_to_idx,
                   prompt_words, context_size, gen_steps, temperature, top_k):
    """Single forward-only generation. No learning, no episodic memory."""
    SKIP = {"<START>", "<BOT>", "<USER>"}
    context = ["<START>"] * context_size
    for w in (["<USER>"] + prompt_words + ["<BOT>"]):
        context.append(w)

    reply = []
    recent = []
    ponder_steps_log = []

    for _ in range(gen_steps):
        ctx_tensors = [
            torch.tensor(embed_cache.get(w, [0.0] * EMBED_DIM), dtype=torch.float32)
            for w in context[-context_size:]
        ]
        ctx_stack = torch.stack(ctx_tensors).unsqueeze(0)

        with torch.no_grad():
            ngram_memory = None
            if engram_module is not None and word_to_id is not None:
                ids = [word_to_id.get(w, 0) for w in context[-3:]]
                ngram_memory = engram_module.lookup(ids).unsqueeze(0)

            predicted, n_steps = brain(ctx_stack, ngram_memory=ngram_memory, engram_module=engram_module)
            predicted = predicted.squeeze(0)

        ponder_steps_log.append(int(n_steps))

        # Penalize recent words to avoid loops; ban specials
        penalty = {w: 1.0 for w in set(recent[-4:])}
        for tok in SKIP:
            penalty[tok] = float("inf")

        candidates = nearest_words(predicted, vocab_matrix_norm, word_list, word_to_idx, top_k, penalty)
        filtered = [(w, s) for w, s in candidates if s > -float("inf")]
        if not filtered:
            break

        words = [w for w, _ in filtered]
        sims_t = torch.tensor([s for _, s in filtered])
        probs = F.softmax(sims_t / temperature, dim=-1)
        chosen_idx = torch.multinomial(probs, 1).item()
        chosen_word = words[chosen_idx]

        if chosen_word == "<USER>":
            break

        reply.append(chosen_word)
        recent.append(chosen_word)
        context.append(chosen_word)

    return reply, ponder_steps_log


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--gen-steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", type=int, default=2,
                   help="How many sampled replies per prompt (different seeds)")
    p.add_argument("--output", default=None,
                   help="Write JSON results here (default: eval_runs/chat_<timestamp>.json)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: {WEIGHTS_PATH} not found. Run training first.")
        sys.exit(1)
    if not os.path.exists(CHROMA_PATH):
        print(f"ERROR: {CHROMA_PATH} not found. Run training first.")
        sys.exit(1)

    print(f"Loading vocab from {CHROMA_PATH}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    vocab_collection = chroma_client.get_collection("engram_vocab")
    all_data = vocab_collection.get(include=["embeddings"])
    embed_cache = {w: list(emb) for w, emb in zip(all_data["ids"], all_data["embeddings"])}
    vocab_size = len(embed_cache)
    print(f"  vocab_size={vocab_size}")

    word_list = list(embed_cache.keys())
    word_to_idx = {w: i for i, w in enumerate(word_list)}
    vocab_matrix = torch.tensor([embed_cache[w] for w in word_list], dtype=torch.float32)
    vocab_matrix_norm = F.normalize(vocab_matrix, dim=-1)

    print(f"Loading brain from {WEIGHTS_PATH}...")
    state_dict = torch.load(WEIGHTS_PATH, weights_only=True)
    use_rope = any(k.startswith("blocks.") and k.endswith(".freqs_cis") for k in state_dict)
    brain = AttentionBrain(embed_dim=EMBED_DIM, context_size=CONTEXT_SIZE, n_layers=N_LAYERS, use_rope=use_rope)
    brain.load_state_dict(state_dict, strict=False)
    brain.eval()
    brain_params = sum(p.numel() for p in brain.parameters())
    print(f"  brain_params={brain_params:,}, use_rope={use_rope}")

    engram_module = None
    word_to_id = None
    if os.path.exists(ENGRAM_PATH) and os.path.exists(W2ID_PATH):
        engram_module = EngramModule(EMBED_DIM)
        engram_module.load_state_dict(torch.load(ENGRAM_PATH, weights_only=True))
        word_to_id = torch.load(W2ID_PATH, weights_only=False)
        engram_module.eval()
        print(f"  engram_module loaded ({len(word_to_id):,} word IDs)")
    else:
        print("  no engram memory module found — running without N-gram memory")

    transcript = []
    print("\n" + "=" * 70)
    print("EVAL TRANSCRIPT — judge each reply for coherence/relevance")
    print(f"  config: temperature={args.temperature}, top_k={args.top_k}, gen_steps={args.gen_steps}, samples={args.samples}")
    print("=" * 70)

    for bucket_name, prompts in PROMPT_BUCKETS.items():
        print(f"\n--- bucket: {bucket_name} ---")
        for prompt in prompts:
            print(f"\n[USER] {prompt}")
            prompt_words = re.findall(r"\b\w+\b", prompt.lower())

            samples = []
            for sample_idx in range(args.samples):
                torch.manual_seed(args.seed + sample_idx)
                reply, ponder_log = generate_reply(
                    brain, engram_module, word_to_id, embed_cache,
                    word_list, vocab_matrix_norm, word_to_idx,
                    prompt_words, CONTEXT_SIZE, args.gen_steps, args.temperature, args.top_k,
                )
                reply_text = " ".join(reply) if reply else "(empty)"
                avg_ponder = sum(ponder_log) / max(len(ponder_log), 1)
                print(f"[BOT #{sample_idx + 1}] {reply_text}")
                print(f"        (avg_ponder={avg_ponder:.1f}, len={len(reply)})")
                samples.append({"reply": reply_text, "avg_ponder": avg_ponder, "length": len(reply)})

            transcript.append({
                "bucket": bucket_name,
                "prompt": prompt,
                "samples": samples,
            })

    # Save JSON
    output = args.output
    if not output:
        os.makedirs("eval_runs", exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = f"eval_runs/chat_{ts}.json"

    with open(output, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "temperature": args.temperature,
                "top_k": args.top_k,
                "gen_steps": args.gen_steps,
                "samples": args.samples,
                "seed": args.seed,
            },
            "model": {
                "brain_params": brain_params,
                "vocab_size": vocab_size,
                "use_rope": use_rope,
            },
            "transcript": transcript,
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"transcript saved to {output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
