# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "chromadb",
# ]
# ///

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chromadb
import re
import os

EMBED_DIM = 64
CONTEXT_SIZE = 8
N_LAYERS = 3
BRAIN_LR = 1e-3
EMBED_LR = 1e-3
TEMPERATURE = 0.9
TOP_K = 10
CHROMA_PATH = "./engram_memory"
NGRAM_TABLE_SIZE = 4999


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
        # x: (B, T, D)
        T = x.size(1)
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scale = x.size(-1) ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        scores = scores.masked_fill(self.mask[:T, :T].unsqueeze(0) == 0, float("-inf"))
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
    """Fixed-size reasoning engine — vocab-independent.
    Includes adaptive pondering: loops through blocks up to max_ponder times,
    with a learned halt gate deciding when to stop."""
    def __init__(self, embed_dim=EMBED_DIM, context_size=CONTEXT_SIZE, n_layers=N_LAYERS, max_ponder=3):
        super().__init__()
        self.pos_embed = nn.Embedding(context_size, embed_dim)
        self.blocks = nn.ModuleList([AttentionBlock(embed_dim, context_size) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.halt_gate = nn.Linear(embed_dim, 1)  # ~65 new params
        self.max_ponder = max_ponder

    def forward(self, x, ngram_memory=None, engram_module=None):
        # x: (B, T, D) — returns (prediction, n_steps) where prediction is (B, D)
        T = x.size(1)
        positions = torch.arange(T, dtype=torch.long)
        x = x + self.pos_embed(positions).unsqueeze(0)

        output = torch.zeros_like(x[:, -1, :])
        remaining = torch.ones(x.size(0), 1)
        n_steps = 0

        for _ in range(self.max_ponder):
            for block_idx, block in enumerate(self.blocks):
                x = block(x)
                # Inject N-gram memory after first block (layer 0)
                if block_idx == 0 and ngram_memory is not None and engram_module is not None:
                    gated_mem = engram_module.gate(x[:, -1, :], ngram_memory)
                    x = x.clone()
                    x[:, -1, :] = x[:, -1, :] + gated_mem
            last_token = x[:, -1, :]
            halt_prob = torch.sigmoid(self.halt_gate(last_token))

            output = output + remaining * last_token
            remaining = remaining * (1 - halt_prob)
            n_steps += 1

            if not self.training and remaining.max().item() < 0.05:
                break

        output = output + remaining * last_token  # distribute remaining mass
        return self.ln_final(output), n_steps


def nearest_words(predicted_t, embed_cache, n=TOP_K, penalty=None):
    """
    In-memory nearest-neighbor search using L2 distance.
    Works against the live embed_cache, so chat-session learning is reflected immediately.
    """
    # Normalize predicted vector for semantic similarity
    predicted_t = F.normalize(predicted_t, dim=0)

    words = list(embed_cache.keys())
    stacked = torch.tensor([embed_cache[w] for w in words], dtype=torch.float32)
    dists = torch.norm(stacked - predicted_t.unsqueeze(0), dim=1)

    if penalty:
        for w, pen in penalty.items():
            if w in embed_cache:
                idx = words.index(w)
                dists[idx] += pen

    k = min(n, len(words))
    top_dists, top_idx = torch.topk(dists, k, largest=False)
    return [(words[i.item()], top_dists[j].item()) for j, i in enumerate(top_idx)]


def generate(brain, embed_cache, context, engram_module=None, word_to_id=None,
             episode_collection=None, show_thoughts=True):
    reply = []
    recent = []
    SKIP = {"<START>", "<BOT>"}

    for step in range(20):
        ctx_tensors = [
            torch.tensor(embed_cache.get(w, [0.0] * EMBED_DIM), dtype=torch.float32)
            for w in context[-CONTEXT_SIZE:]
        ]
        ctx_stack = torch.stack(ctx_tensors)

        with torch.no_grad():
            # Compute N-gram memory for current context
            ngram_memory = None
            if engram_module is not None and word_to_id is not None:
                ctx_words = context[-3:]
                ids = [word_to_id.get(w, 0) for w in ctx_words]
                ngram_memory = engram_module.lookup(ids).unsqueeze(0)  # (1, D)

            predicted, n_steps = brain(ctx_stack.unsqueeze(0),
                                       ngram_memory=ngram_memory,
                                       engram_module=engram_module)
            predicted = predicted.squeeze(0)

            # Query episodic memory and blend with gated prediction
            if episode_collection is not None and episode_collection.count() > 0:
                results = episode_collection.query(
                    query_embeddings=[predicted.tolist()], n_results=3
                )
                if results["ids"][0]:
                    episode_words = [m["target"] for m in results["metadatas"][0]]
                    episode_embeds = [torch.tensor(embed_cache[w]) for w in episode_words if w in embed_cache]
                    if episode_embeds:
                        episode_signal = torch.stack(episode_embeds).mean(dim=0)
                        if engram_module is not None:
                            # Learned gate replaces fixed 0.3 blend
                            gated_episode = engram_module.gate(predicted, episode_signal)
                            predicted = predicted + gated_episode
                        else:
                            predicted = 0.7 * predicted + 0.3 * episode_signal
                        if show_thoughts:
                            print(f"    Retrieved episodes: {episode_words[:3]}")

        # Repetition penalty: push recently used words further away in distance space
        penalty = {w: 3.0 for w in set(recent[-4:])}
        for tok in SKIP:
            penalty[tok] = float("inf")

        candidates = nearest_words(predicted, embed_cache, penalty=penalty)

        # Filter skip tokens and convert distances → probabilities via temperature
        filtered = [(w, d) for w, d in candidates if d < float("inf")]
        if not filtered:
            break

        words_list = [w for w, _ in filtered]
        dists_t = torch.tensor([d for _, d in filtered])
        # Negate: closer distance = higher score
        probs = F.softmax(-dists_t / TEMPERATURE, dim=-1)
        chosen_idx = torch.multinomial(probs, 1).item()
        chosen_word = words_list[chosen_idx]

        if show_thoughts:
            top3 = [(w, round(d, 3)) for w, d in filtered[:3]]
            print(f"  Thought {step + 1} ({n_steps} ponder steps): {top3}")

        if chosen_word == "<USER>":
            break

        reply.append(chosen_word)
        recent.append(chosen_word)
        context.append(chosen_word)

    return reply


def main():
    if not os.path.exists("engram_weights.pth") or not os.path.exists(CHROMA_PATH):
        print("No trained brain found. Run ingest.py first.")
        return

    print("Booting Engram...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    vocab_collection = chroma_client.get_collection("engram_vocab")

    # Create or load episodic memory collection
    try:
        episode_collection = chroma_client.get_collection("engram_episodes")
        episode_count = episode_collection.count()
        print(f"Loaded episodic memory: {episode_count} episodes")
    except:
        episode_collection = chroma_client.create_collection("engram_episodes")
        print("Created new episodic memory collection")

    # Load all word vectors from ChromaDB into memory
    all_data = vocab_collection.get(include=["embeddings"])
    # Convert to plain Python lists so torch.tensor() doesn't slow-path through numpy
    embed_cache = {
        word: list(emb) for word, emb in zip(all_data["ids"], all_data["embeddings"])
    }
    print(f"Loaded {len(embed_cache)} concept vectors from ChromaDB.")

    brain = AttentionBrain()
    brain.load_state_dict(torch.load("engram_weights.pth", weights_only=True))

    # Load EngramModule if available
    engram_module = None
    word_to_id = None
    if os.path.exists("engram_memory_module.pth") and os.path.exists("engram_word_to_id.pth"):
        engram_module = EngramModule(EMBED_DIM)
        engram_module.load_state_dict(torch.load("engram_memory_module.pth", weights_only=True))
        word_to_id = torch.load("engram_word_to_id.pth", weights_only=False)
        engram_module.eval()
        engram_params = sum(p.numel() for p in engram_module.parameters())
        print(f"Loaded Engram memory module ({engram_params:,} params, {len(word_to_id):,} word IDs)")
    else:
        print("No Engram memory module found — running without N-gram memory.")

    all_params = list(brain.parameters())
    if engram_module is not None:
        all_params += list(engram_module.parameters())
    optimizer = optim.Adam(all_params, lr=BRAIN_LR)

    print("Engram is alive. Type 'quit' to exit.\n")

    context = ["<START>"] * CONTEXT_SIZE
    SKIP_TOKENS = {"<START>", "<BOT>"}
    surprise_ema = 1.0  # Exponential moving average of surprise
    turn_counter = 0

    while True:
        user_text = input("\nYou: ").lower().strip()
        if user_text == "quit":
            break

        turn_counter += 1
        user_words = ["<USER>"] + re.findall(r"\b\w+\b", user_text) + ["<BOT>"]

        # New words get a random starting position in concept space
        for w in user_words:
            if w not in embed_cache:
                embed_cache[w] = torch.randn(EMBED_DIM).tolist()
            # Also add to word_to_id if engram module is loaded
            if word_to_id is not None and w not in word_to_id:
                word_to_id[w] = len(word_to_id)

        # Online learning: carve an engram for this turn
        brain.train()
        if engram_module is not None:
            engram_module.train()
        step_counter = 0
        for word in user_words:
            ctx_tensors = [
                torch.tensor(embed_cache[w], dtype=torch.float32, requires_grad=True)
                for w in context[-CONTEXT_SIZE:]
            ]
            ctx_stack = torch.stack(ctx_tensors)
            target_t = torch.tensor(embed_cache[word], dtype=torch.float32, requires_grad=True)

            # Compute N-gram memory
            ngram_memory = None
            if engram_module is not None and word_to_id is not None:
                ctx_words = context[-3:]
                ids = [word_to_id.get(w, 0) for w in ctx_words]
                ngram_memory = engram_module.lookup(ids).unsqueeze(0)

            optimizer.zero_grad()
            predicted, n_steps = brain(ctx_stack.unsqueeze(0),
                                       ngram_memory=ngram_memory,
                                       engram_module=engram_module)
            predicted = predicted.squeeze(0)
            loss = F.mse_loss(predicted, target_t)

            # Surprise-gated learning: weight loss by relative surprise
            surprise = loss.item()
            surprise_ema = 0.9 * surprise_ema + 0.1 * surprise
            relative_surprise = surprise / (surprise_ema + 1e-8)
            surprise_weight = min(1.0 + relative_surprise, 3.0)
            weighted_loss = loss * surprise_weight

            weighted_loss.backward()
            optimizer.step()

            # Print surprise for visibility
            print(f"  Learning '{word}': surprise={surprise:.4f}, relative={relative_surprise:.2f}x")

            # Store high-surprise moments as episodic memories
            if relative_surprise > 1.5:
                episode_id = f"ep_{turn_counter}_{step_counter}"
                episode_collection.add(
                    ids=[episode_id],
                    embeddings=[predicted.detach().tolist()],
                    metadatas=[{"target": word, "surprise": round(surprise, 4), "turn": turn_counter}],
                    documents=[" ".join(context[-4:])]
                )
                print(f"    → Stored episode: {episode_id}")

            for w, t in zip(context[-CONTEXT_SIZE:], ctx_tensors):
                if t.grad is not None:
                    embed_cache[w] = (t.detach() - EMBED_LR * t.grad).tolist()

            if target_t.grad is not None:
                embed_cache[word] = (target_t.detach() - EMBED_LR * target_t.grad).tolist()

            context.append(word)
            step_counter += 1

        # Generate
        brain.eval()
        if engram_module is not None:
            engram_module.eval()
        print("\n--- Engram's Subconscious ---")
        reply = generate(brain, embed_cache, context,
                         engram_module=engram_module, word_to_id=word_to_id,
                         episode_collection=episode_collection)
        print("----------------------------")
        print(f"\nEngram: {' '.join(reply) if reply else '(thinking...)'}")

    # Flush updated embeddings back to ChromaDB in chunks (ChromaDB batch limit)
    print("\nSaving session learning to ChromaDB...")
    all_ids = list(embed_cache.keys())
    all_embeds = [embed_cache[w] for w in all_ids]
    chunk = 5000
    for i in range(0, len(all_ids), chunk):
        vocab_collection.upsert(ids=all_ids[i:i+chunk], embeddings=all_embeds[i:i+chunk], documents=all_ids[i:i+chunk])

    torch.save(brain.state_dict(), "engram_weights.pth")
    if engram_module is not None:
        torch.save(engram_module.state_dict(), "engram_memory_module.pth")
    if word_to_id is not None:
        torch.save(word_to_id, "engram_word_to_id.pth")
    episode_count = episode_collection.count()
    print(f"Session saved. {episode_count} episodes in memory. Goodbye.")


if __name__ == "__main__":
    main()
