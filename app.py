"""
Engram — Gradio Chat Interface for Hugging Face Spaces.
Custom neural net chatbot trained from scratch. No external LLMs, no pre-trained weights.
"""

import json
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

from engram_model import AttentionBrain, EngramModule

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPERATURE = 0.9
TOP_K = 10
GEN_STEPS = 20


# ── Inference ─────────────────────────────────────────────────────────────────


def nearest_words(predicted_t, word_list, vocab_matrix, word_to_idx, n=TOP_K, penalty=None):
    predicted_t = F.normalize(predicted_t, dim=0)
    dists = torch.norm(vocab_matrix - predicted_t.unsqueeze(0), dim=1).clone()
    if penalty:
        for w, pen in penalty.items():
            if w in word_to_idx:
                dists[word_to_idx[w]] += pen
    k = min(n, len(word_list))
    top_dists, top_idx = torch.topk(dists, k, largest=False)
    return [(word_list[i.item()], top_dists[j].item()) for j, i in enumerate(top_idx)]


def generate_response(brain, word_list, vocab_matrix, word_to_idx, prompt_words, context_size,
                      engram_module=None, word_to_id=None):
    SKIP = {"<START>", "<BOT>", "<USER>"}
    embed_dim = vocab_matrix.shape[1]
    context = ["<START>"] * context_size
    for w in ["<USER>"] + prompt_words + ["<BOT>"]:
        context.append(w)

    def get_embed(word):
        if word in word_to_idx:
            return vocab_matrix[word_to_idx[word]]
        return torch.zeros(embed_dim)

    reply = []
    recent = []

    for step in range(GEN_STEPS):
        ctx_tensors = [get_embed(w) for w in context[-context_size:]]
        ctx_stack = torch.stack(ctx_tensors)

        with torch.no_grad():
            # Compute N-gram memory
            ngram_memory = None
            if engram_module is not None and word_to_id is not None:
                ctx_words = context[-3:]
                ids = [word_to_id.get(w, 0) for w in ctx_words]
                ngram_memory = engram_module.lookup(ids).unsqueeze(0)

            predicted, _ = brain(ctx_stack.unsqueeze(0),
                                 ngram_memory=ngram_memory,
                                 engram_module=engram_module)
            predicted = predicted.squeeze(0)

        penalty = {w: 3.0 for w in set(recent[-4:])}
        for tok in SKIP:
            penalty[tok] = float("inf")

        candidates = nearest_words(predicted, word_list, vocab_matrix, word_to_idx, penalty=penalty)
        filtered = [(w, d) for w, d in candidates if d < float("inf")]
        if not filtered:
            break

        words_list_c = [w for w, _ in filtered]
        dists_t = torch.tensor([d for _, d in filtered])
        probs = F.softmax(-dists_t / TEMPERATURE, dim=-1)
        chosen_idx = torch.multinomial(probs, 1).item()
        chosen_word = words_list_c[chosen_idx]

        if chosen_word == "<USER>":
            break

        reply.append(chosen_word)
        recent.append(chosen_word)
        context.append(chosen_word)

    return " ".join(reply) if reply else "..."


# ── Load Model ────────────────────────────────────────────────────────────────

def load_engram():
    """Load weights and vocab embeddings from files."""
    # Load vocab from JSON (exported from ChromaDB)
    vocab_path = os.path.join(BASE_DIR, "vocab_embeddings.json")
    with open(vocab_path, "r") as f:
        embed_cache = json.load(f)

    word_list = list(embed_cache.keys())
    word_to_idx = {w: i for i, w in enumerate(word_list)}
    vocab_matrix = torch.tensor(
        [embed_cache[w] for w in word_list], dtype=torch.float32
    )

    # Load brain weights — auto-detect hyperparams from checkpoint
    weights_path = os.path.join(BASE_DIR, "engram_weights.pth")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    embed_dim = state_dict["ln_final.weight"].shape[0]
    context_size = state_dict["pos_embed.weight"].shape[0]
    n_layers = max(int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")) + 1

    # Detect RoPE checkpoints by presence of freqs_cis buffers
    use_rope = any(k.startswith("blocks.") and k.endswith(".freqs_cis") for k in state_dict)
    brain = AttentionBrain(embed_dim=embed_dim, context_size=context_size, n_layers=n_layers, use_rope=use_rope)
    brain.load_state_dict(state_dict, strict=False)
    brain.eval()

    # Load EngramModule if available
    engram_module = None
    word_to_id = None
    engram_path = os.path.join(BASE_DIR, "engram_memory_module.pth")
    w2id_path = os.path.join(BASE_DIR, "engram_word_to_id.pth")
    if os.path.exists(engram_path) and os.path.exists(w2id_path):
        engram_module = EngramModule(embed_dim)
        engram_module.load_state_dict(torch.load(engram_path, map_location="cpu", weights_only=True))
        word_to_id = torch.load(w2id_path, map_location="cpu", weights_only=False)
        engram_module.eval()
        engram_params = sum(p.numel() for p in engram_module.parameters())
        print(f"Loaded Engram memory module ({engram_params:,} params)")

    return brain, word_list, vocab_matrix, word_to_idx, context_size, len(embed_cache), engram_module, word_to_id


print("Loading Engram model...")
brain, word_list, vocab_matrix, word_to_idx, context_size, vocab_size, engram_module, word_to_id = load_engram()
engram_status = f", engram={'yes' if engram_module else 'no'}"
print(f"Engram loaded: {vocab_size:,} words, embed={vocab_matrix.shape[1]}, ctx={context_size}{engram_status}")


# ── Gradio Interface ──────────────────────────────────────────────────────────

def chat(message, history):
    prompt_words = re.findall(r"\b\w+\b", message.lower())
    if not prompt_words:
        return "..."
    return generate_response(brain, word_list, vocab_matrix, word_to_idx, prompt_words, context_size,
                             engram_module=engram_module, word_to_id=word_to_id)


demo = gr.ChatInterface(
    fn=chat,
    title="Engram — Custom Neural Net Chat",
    description=(
        f"Trained from scratch on conversational data. No LLM APIs, no pre-trained weights. "
        f"Vocab: {vocab_size:,} words | Architecture: AttentionBrain + Conditional Memory "
        f"(embed={vocab_matrix.shape[1]}, ctx={context_size}, layers={max(int(k.split('.')[1]) for k in brain.state_dict() if k.startswith('blocks.')) + 1})"
    ),
    examples=["Hello how are you", "What do you like to do", "Tell me about yourself"],
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()
