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
from engram_model import (
    AttentionBlock, EngramModule, AttentionBrain,
    EMBED_DIM, CONTEXT_SIZE, N_LAYERS, NGRAM_TABLE_SIZE,
)

BRAIN_LR = 1e-3
EMBED_LR = 1e-3
TEMPERATURE = 0.9
TOP_K = 10
CHROMA_PATH = "./engram_memory"

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
