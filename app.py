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

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPERATURE = 0.9
TOP_K = 10
GEN_STEPS = 20
NGRAM_TABLE_SIZE = 4999

# ── Model Architecture ───────────────────────────────────────────────────────


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
        scores = scores.masked_fill(
            self.mask[:T, :T].unsqueeze(0) == 0, float("-inf")
        )
        attn = F.softmax(scores, dim=-1)
        x = self.ln1(x + torch.matmul(attn, V))
        x = self.ln2(x + self.ff(x))
        return x


class EngramModule(nn.Module):
    """N-gram embedding tables with learned gating."""
    HASH_PRIME = 31

    def __init__(self, embed_dim, table_size=NGRAM_TABLE_SIZE):
        super().__init__()
        half_dim = embed_dim // 2
        self.embed_dim = embed_dim
        self.table_size = table_size
        self.bigram_table = nn.Embedding(table_size, half_dim)
        self.trigram_table = nn.Embedding(table_size, half_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.normal_(self.bigram_table.weight, std=0.02)
        nn.init.normal_(self.trigram_table.weight, std=0.02)

    def hash_bigram(self, id1, id2):
        return ((id1 * self.HASH_PRIME) ^ id2) % self.table_size

    def hash_trigram(self, id1, id2, id3):
        return (((id1 * self.HASH_PRIME) ^ id2) * self.HASH_PRIME ^ id3) % self.table_size

    def lookup(self, word_ids):
        if len(word_ids) >= 2:
            bh = self.hash_bigram(word_ids[-2], word_ids[-1])
            bigram_emb = self.bigram_table(torch.tensor(bh))
        else:
            bigram_emb = torch.zeros(self.embed_dim // 2)
        if len(word_ids) >= 3:
            th = self.hash_trigram(word_ids[-3], word_ids[-2], word_ids[-1])
            trigram_emb = self.trigram_table(torch.tensor(th))
        else:
            trigram_emb = torch.zeros(self.embed_dim // 2)
        return torch.cat([bigram_emb, trigram_emb], dim=-1)

    def gate(self, hidden_state, memory_vector):
        k = self.W_K(memory_vector)
        v = self.W_V(memory_vector)
        alpha = torch.sigmoid(
            (F.normalize(hidden_state, dim=-1) * F.normalize(k, dim=-1)).sum(-1, keepdim=True)
            / (hidden_state.size(-1) ** 0.5)
        )
        return alpha * v


class AttentionBrain(nn.Module):
    def __init__(self, embed_dim, context_size, n_layers, max_ponder=3):
        super().__init__()
        self.pos_embed = nn.Embedding(context_size, embed_dim)
        self.blocks = nn.ModuleList(
            [AttentionBlock(embed_dim, context_size) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        self.halt_gate = nn.Linear(embed_dim, 1)
        self.max_ponder = max_ponder

    def forward(self, x, ngram_memory=None, engram_module=None):
        T = x.size(1)
        positions = torch.arange(T, dtype=torch.long)
        x = x + self.pos_embed(positions).unsqueeze(0)

        output = torch.zeros_like(x[:, -1, :])
        remaining = torch.ones(x.size(0), 1)

        for _ in range(self.max_ponder):
            for block_idx, block in enumerate(self.blocks):
                x = block(x)
                if block_idx == 0 and ngram_memory is not None and engram_module is not None:
                    gated_mem = engram_module.gate(x[:, -1, :], ngram_memory)
                    x = x.clone()
                    x[:, -1, :] = x[:, -1, :] + gated_mem
            last_token = x[:, -1, :]
            halt_prob = torch.sigmoid(self.halt_gate(last_token))
            output = output + remaining * last_token
            remaining = remaining * (1 - halt_prob)
            if not self.training and remaining.max().item() < 0.05:
                break

        output = output + remaining * last_token
        return self.ln_final(output)


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

            predicted = brain(ctx_stack.unsqueeze(0),
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

    brain = AttentionBrain(embed_dim=embed_dim, context_size=context_size, n_layers=n_layers)
    brain.load_state_dict(state_dict)
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
