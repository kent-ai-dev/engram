# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "chromadb",
# ]
# ///

"""
eval_brain.py — Non-interactive evaluation of the Engram model.
Runs test prompts, captures output, calculates coherence, and saves JSON results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chromadb
import re
import os
import json
import sys
from datetime import datetime

EMBED_DIM = 64
CONTEXT_SIZE = 8
N_LAYERS = 3
TEMPERATURE = 0.9
TOP_K = 10
CHROMA_PATH = "./engram_memory"
WEIGHTS_PATH = "./engram_weights.pth"

TEST_PROMPTS = [
    "what is the capital of france",
    "tell me a story about adventure",
    "how do you make coffee",
    "what do you think about friendship",
    "can you help me understand mathematics",
]

# Accept optional config name as argument
CONFIG_NAME = sys.argv[1] if len(sys.argv) > 1 else "default"
ITERATION = sys.argv[2] if len(sys.argv) > 2 else "0"


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, context_size):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        mask = torch.tril(torch.ones(context_size, context_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        T = x.size(1)
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scale = x.size(-1) ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        scores = scores.masked_fill(self.mask[:T, :T].unsqueeze(0) == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        x = self.ln1(x + torch.matmul(attn, V))
        x = self.ln2(x + self.ff(x))
        return x


class AttentionBrain(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, context_size=CONTEXT_SIZE, n_layers=N_LAYERS, max_ponder=3):
        super().__init__()
        self.pos_embed = nn.Embedding(context_size, embed_dim)
        self.blocks = nn.ModuleList([AttentionBlock(embed_dim, context_size) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.halt_gate = nn.Linear(embed_dim, 1)
        self.max_ponder = max_ponder

    def forward(self, x):
        T = x.size(1)
        positions = torch.arange(T, dtype=torch.long)
        x = x + self.pos_embed(positions).unsqueeze(0)

        output = torch.zeros_like(x[:, -1, :])
        remaining = torch.ones(x.size(0), 1)
        n_steps = 0

        for _ in range(self.max_ponder):
            for block in self.blocks:
                x = block(x)
            last_token = x[:, -1, :]
            halt_prob = torch.sigmoid(self.halt_gate(last_token))
            output = output + remaining * last_token
            remaining = remaining * (1 - halt_prob)
            n_steps += 1
            if not self.training and remaining.max().item() < 0.05:
                break

        output = output + remaining * last_token
        return self.ln_final(output), n_steps


def nearest_words(predicted_t, word_list, vocab_matrix, word_to_idx, n=TOP_K, penalty=None):
    """Vectorized nearest-word search using pre-built vocab_matrix tensor."""
    predicted_t = F.normalize(predicted_t, dim=0)
    dists = torch.norm(vocab_matrix - predicted_t.unsqueeze(0), dim=1).clone()
    if penalty:
        for w, pen in penalty.items():
            if w in word_to_idx:
                dists[word_to_idx[w]] += pen
    k = min(n, len(word_list))
    top_dists, top_idx = torch.topk(dists, k, largest=False)
    return [(word_list[i.item()], top_dists[j].item()) for j, i in enumerate(top_idx)]


def generate_response(brain, embed_cache, word_list, vocab_matrix, word_to_idx, prompt_words, context_size):
    """Generate a response for a given prompt, return (reply_words, avg_ponder_steps, avg_surprise).
    
    word_list, vocab_matrix, word_to_idx: pre-built for fast nearest-word search.
    """
    SKIP = {"<START>", "<BOT>", "<USER>"}
    GEN_STEPS = 12  # reduced for speed

    context = ["<START>"] * context_size
    for w in (["<USER>"] + prompt_words + ["<BOT>"]):
        context.append(w)

    # Build context embedding lookup: use vocab_matrix for known words
    def get_embed(word):
        if word in word_to_idx:
            return vocab_matrix[word_to_idx[word]]
        return torch.zeros(EMBED_DIM)

    reply = []
    recent = []
    ponder_steps_list = []
    surprise_list = []

    for step in range(GEN_STEPS):
        ctx_tensors = [get_embed(w) for w in context[-context_size:]]
        ctx_stack = torch.stack(ctx_tensors)

        with torch.no_grad():
            predicted, n_steps = brain(ctx_stack.unsqueeze(0))
            predicted = predicted.squeeze(0)

        ponder_steps_list.append(n_steps)

        penalty = {w: 3.0 for w in set(recent[-4:])}
        for tok in SKIP:
            penalty[tok] = float("inf")

        candidates = nearest_words(predicted, word_list, vocab_matrix, word_to_idx, penalty=penalty)
        filtered = [(w, d) for w, d in candidates if d < float("inf")]
        if not filtered:
            break

        words_list = [w for w, _ in filtered]
        dists_t = torch.tensor([d for _, d in filtered])
        probs = F.softmax(-dists_t / TEMPERATURE, dim=-1)
        chosen_idx = torch.multinomial(probs, 1).item()
        chosen_word = words_list[chosen_idx]

        # Record surprise as the distance to the chosen word
        chosen_dist = filtered[chosen_idx][1]
        surprise_list.append(chosen_dist)

        if chosen_word == "<USER>":
            break

        reply.append(chosen_word)
        recent.append(chosen_word)
        context.append(chosen_word)

    avg_ponder = sum(ponder_steps_list) / max(len(ponder_steps_list), 1)
    avg_surprise = sum(surprise_list) / max(len(surprise_list), 1)
    return reply, avg_ponder, avg_surprise


def calculate_coherence(reply_words, embed_cache):
    """
    Coherence score: ratio of reply words that are real vocabulary words
    (not special tokens, not purely numeric).
    Words in embed_cache that aren't special tokens are considered 'real'.
    """
    if not reply_words:
        return 0.0
    special = {"<start>", "<user>", "<bot>"}
    real_word_count = sum(
        1 for w in reply_words
        if w.lower() not in special and re.match(r"^[a-zA-Z]+$", w) and w in embed_cache
    )
    return real_word_count / len(reply_words)


def main():
    weights_path = WEIGHTS_PATH
    chroma_path = CHROMA_PATH

    if not os.path.exists(weights_path):
        print(f"ERROR: No weights found at {weights_path}")
        sys.exit(1)
    if not os.path.exists(chroma_path):
        print(f"ERROR: No ChromaDB found at {chroma_path}")
        sys.exit(1)

    print(f"Loading Engram for evaluation (config={CONFIG_NAME}, iter={ITERATION})...")

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    vocab_collection = chroma_client.get_collection("engram_vocab")

    all_data = vocab_collection.get(include=["embeddings"])
    embed_cache = {
        word: list(emb) for word, emb in zip(all_data["ids"], all_data["embeddings"])
    }
    vocab_size = len(embed_cache)
    print(f"Loaded {vocab_size} concept vectors.")

    # Explicitly release ChromaDB client to unlock SQLite file for next training run
    del vocab_collection
    del all_data
    try:
        chroma_client._producer = None
        chroma_client._consumer = None
    except Exception:
        pass
    del chroma_client
    import gc
    gc.collect()

    # Pre-build vocab matrix for fast vectorized nearest-word search
    word_list = list(embed_cache.keys())
    word_to_idx = {w: i for i, w in enumerate(word_list)}
    vocab_matrix = torch.tensor(
        [embed_cache[w] for w in word_list], dtype=torch.float32
    )
    print(f"Vocab matrix built: {vocab_matrix.shape}")

    # Get episode count if available
    episode_count = 0
    try:
        ep_col = chroma_client.get_collection("engram_episodes")
        episode_count = ep_col.count()
    except Exception:
        pass

    # Load brain -- use global EMBED_DIM/CONTEXT_SIZE/N_LAYERS (patched by train_runner)
    brain = AttentionBrain(embed_dim=EMBED_DIM, context_size=CONTEXT_SIZE, n_layers=N_LAYERS)
    brain.load_state_dict(torch.load(weights_path, weights_only=True))
    brain.eval()

    results_per_prompt = []
    all_ponder_steps = []
    all_surprises = []
    all_coherences = []
    all_response_lengths = []

    for prompt in TEST_PROMPTS:
        print(f"\nPrompt: {prompt}")
        prompt_words = re.findall(r"\b\w+\b", prompt.lower())
        reply, avg_ponder, avg_surprise = generate_response(
            brain, embed_cache, word_list, vocab_matrix, word_to_idx, prompt_words, CONTEXT_SIZE
        )
        coherence = calculate_coherence(reply, embed_cache)
        response_text = " ".join(reply) if reply else "(no response)"
        print(f"  Response: {response_text}")
        print(f"  Ponder: {avg_ponder:.2f}, Surprise: {avg_surprise:.4f}, Coherence: {coherence:.2f}, Len: {len(reply)}")

        all_ponder_steps.append(avg_ponder)
        all_surprises.append(avg_surprise)
        all_coherences.append(coherence)
        all_response_lengths.append(len(reply))

        results_per_prompt.append({
            "prompt": prompt,
            "response": response_text,
            "avg_ponder_steps": avg_ponder,
            "avg_surprise": avg_surprise,
            "coherence_score": coherence,
            "response_length": len(reply),
        })

    # Aggregate metrics
    metrics = {
        "config_name": CONFIG_NAME,
        "iteration": int(ITERATION),
        "timestamp": datetime.utcnow().isoformat(),
        "vocab_size": vocab_size,
        "episode_count": episode_count,
        "avg_surprise": sum(all_surprises) / max(len(all_surprises), 1),
        "avg_ponder_steps": sum(all_ponder_steps) / max(len(all_ponder_steps), 1),
        "coherence_score": sum(all_coherences) / max(len(all_coherences), 1),
        "avg_response_length": sum(all_response_lengths) / max(len(all_response_lengths), 1),
        "prompts": results_per_prompt,
    }

    # Save results
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = f"eval_results_{CONFIG_NAME}_{ITERATION}_{timestamp_str}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== EVAL SUMMARY (config={CONFIG_NAME}) ===")
    print(f"  Vocab size:       {vocab_size:,}")
    print(f"  Episode count:    {episode_count}")
    print(f"  Avg surprise:     {metrics['avg_surprise']:.4f}")
    print(f"  Avg ponder steps: {metrics['avg_ponder_steps']:.2f}")
    print(f"  Coherence score:  {metrics['coherence_score']:.2f}")
    print(f"  Avg resp length:  {metrics['avg_response_length']:.1f}")
    print(f"  Results saved:    {output_file}")

    # Print a machine-readable summary line for train_runner to parse
    print(f"EVAL_RESULT: {json.dumps({'file': output_file, **{k: v for k, v in metrics.items() if k != 'prompts'}})}")


if __name__ == "__main__":
    main()
