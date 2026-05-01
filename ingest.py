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
import shutil
import random
from engram_model import (
    AttentionBlock, EngramModule, AttentionBrain,
    EMBED_DIM, CONTEXT_SIZE, N_LAYERS, N_HEADS, NGRAM_TABLE_SIZE, SPECIAL_TOKENS,
)


BATCH_SIZE = 128  # smaller batch for larger model (GPU memory)
BRAIN_LR = 1e-3  # lower LR for deeper model
EMBED_LR = 5e-4
EPOCHS = 5
CHROMA_PATH = "./engram_memory"

# Device selection: use CUDA if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[engram] Using device: {DEVICE}")


def main():

    import argparse

    parser = argparse.ArgumentParser(description="Engram Ingestion Engine")

    parser.add_argument("--books", nargs="*", help="Specific book files to train on (e.g. corpus/84_frankenstein.txt). If omitted, uses all corpus files.")

    args = parser.parse_args()



    print("Booting Engram Ingestion Engine...")

    print(f"Batch size: {BATCH_SIZE} | Context: {CONTEXT_SIZE} | Layers: {N_LAYERS}\n")



    import time as _time



    # Clean up old ChromaDB directory ? retry up to 15 times (Windows holds locks briefly)

    if os.path.exists(CHROMA_PATH):

        for _attempt in range(15):

            try:

                shutil.rmtree(CHROMA_PATH)

                print(f"  Removed old ChromaDB dir (attempt {_attempt+1})")

                break

            except (PermissionError, OSError) as _pe:

                print(f"  ChromaDB dir locked (attempt {_attempt+1}/15), waiting 2s... {_pe}")

                _time.sleep(2)

        else:

            # All retries failed ? rename the old dir out of the way

            _old = CHROMA_PATH + f"_old_{int(_time.time())}"

            try:

                os.rename(CHROMA_PATH, _old)

                print(f"  Renamed locked ChromaDB to {_old}")

            except Exception as _re:

                print(f"  WARNING: Could not remove or rename ChromaDB dir: {_re}")



    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Use get_or_create as a safety net in case dir rename fallback was used

    try:

        vocab_collection = chroma_client.create_collection(name="engram_vocab")

    except Exception:

        # Collection exists (rename fallback path) ? delete and recreate fresh

        try:

            chroma_client.delete_collection(name="engram_vocab")

        except Exception:

            pass

        vocab_collection = chroma_client.create_collection(name="engram_vocab")



    # Read all .txt files from the corpus/ folder + training_data.txt

    corpus_files = []

    if args.books:

        # Use only specified books

        corpus_files = [b for b in args.books if os.path.exists(b)]

        if not corpus_files:

            print(f"Error: None of the specified books found: {args.books}")

            return

    else:

        if os.path.exists("training_data.txt"):

            corpus_files.append("training_data.txt")

        if os.path.exists("corpus"):

            for fname in sorted(os.listdir("corpus")):

                if fname.endswith(".txt"):

                    corpus_files.append(os.path.join("corpus", fname))



    if not corpus_files:

        print("Error: No training data found.")

        print("Add text to training_data.txt or run download_book.py to get a book.")

        return



    raw_text = ""

    for fpath in corpus_files:

        with open(fpath, "r", encoding="utf-8") as f:

            raw_text += f.read().lower() + "\n"

        print(f"Loaded: {fpath}")



    # Build sequences from paragraphs (blank line = boundary)

    QUESTION_STARTERS = {"what", "how", "can", "do", "is", "tell", "why", "where",

                         "should", "did", "are", "does", "will", "would", "have"}

    ANSWER_STARTERS = {"i", "my", "yes", "no", "it", "the", "that", "we", "perhaps",

                       "of", "memory", "wisdom", "home", "friends", "family",

                       "mistakes", "beauty", "reality", "time"}



    def split_qa(line_words):

        """If line looks like Q&A, return ([<USER> + question], [<BOT> + answer]). Else None."""

        if len(line_words) < 6:

            return None

        if line_words[0] not in QUESTION_STARTERS:

            return None

        # Find the likely split point: look for a pronoun/noun restart after position 3

        for i in range(3, len(line_words) - 2):

            if line_words[i] in ANSWER_STARTERS:

                return (["<USER>"] + line_words[:i], ["<BOT>"] + line_words[i:])

        return None



    paragraphs = re.split(r'\n\s*\n', raw_text)

    sequences = []

    qa_pairs_detected = 0



    for para in paragraphs:

        para_words = re.findall(r'\b\w+\b', para)

        if not para_words:

            continue



        # Try to detect Q&A pattern

        qa_split = split_qa(para_words)

        if qa_split:

            question_words, answer_words = qa_split

            para_words = question_words + answer_words

            qa_pairs_detected += 1



        ctx = ["<START>"] * CONTEXT_SIZE

        for word in para_words:

            sequences.append((list(ctx[-CONTEXT_SIZE:]), word))

            ctx.append(word)



    # Collect unique words after sequence building

    all_words = []

    for ctx_list, target in sequences:

        all_words.extend(ctx_list)

        all_words.append(target)

    unique_words = list(dict.fromkeys(SPECIAL_TOKENS + all_words))



    print(f"\n{len(paragraphs):,} paragraphs | {qa_pairs_detected} Q&A pairs detected")

    print(f"{len(sequences):,} total sequences | {len(unique_words):,} unique tokens")



    # In-memory embedding cache ??? source of truth during training

    embed_cache = {w: torch.randn(EMBED_DIM).tolist() for w in unique_words}



    # Build word_to_id mapping for N-gram hashing
    word_to_id = {w: i for i, w in enumerate(unique_words)}

    brain = AttentionBrain()
    engram = EngramModule(EMBED_DIM)
    brain.to(DEVICE)
    engram.to(DEVICE)
    all_params = list(brain.parameters()) + list(engram.parameters())
    # AdamW with weight decay — without it, MSE training on broad target
    # distributions converges to "predict the mean" (v4/v5 mode collapse).
    optimizer = optim.AdamW(all_params, lr=BRAIN_LR, weight_decay=0.01)
    # Cosine LR schedule: decay from BRAIN_LR to 1e-5 over training
    total_steps = ((len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE) * EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    brain_params = sum(p.numel() for p in brain.parameters())
    engram_params = sum(p.numel() for p in engram.parameters())
    print(f"Brain parameters: {brain_params:,} | Engram memory: {engram_params:,} | Total: {brain_params + engram_params:,}")



    n_batches = (len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nTraining: {len(sequences):,} sequences -> {n_batches:,} batches/epoch x {EPOCHS} epochs\n")



    brain.train()
    engram.train()

    for epoch in range(EPOCHS):

        random.shuffle(sequences)

        total_loss = 0.0

        total_ponder_steps = 0

        n_steps = 0



        for batch_start in range(0, len(sequences), BATCH_SIZE):

            batch = sequences[batch_start : batch_start + BATCH_SIZE]

            ctx_word_lists = [s[0] for s in batch]

            target_words = [s[1] for s in batch]



            # Collect unique words in this batch for the embedding mini-matrix

            all_batch_words = list(

                dict.fromkeys(w for ctx_wl in ctx_word_lists for w in ctx_wl) |

                dict.fromkeys(target_words)

            )

            batch_idx = {w: i for i, w in enumerate(all_batch_words)}



            # Build a local embedding matrix for this batch (gradients flow through it)

            batch_embed = torch.tensor(

                [embed_cache[w] for w in all_batch_words], dtype=torch.float32

            ).to(DEVICE).requires_grad_(True)



            # (B, T) index tensors ??? (B, T, D) and (B, D)

            ctx_idx = torch.tensor([[batch_idx[w] for w in cw] for cw in ctx_word_lists]).to(DEVICE)

            tgt_idx = torch.tensor([batch_idx[w] for w in target_words]).to(DEVICE)



            ctx_embeds = batch_embed[ctx_idx]      # (B, T, D)

            target_embeds = batch_embed[tgt_idx]   # (B, D)

            # Compute N-gram hashes for each sequence in the batch
            ngram_id_seqs = []
            for cw in ctx_word_lists:
                ids = [word_to_id.get(w, 0) for w in cw[-3:]]
                ngram_id_seqs.append(ids)
            ngram_memory = engram.lookup_batch(ngram_id_seqs)  # (B, D)

            optimizer.zero_grad()

            predicted, ponder_steps = brain(ctx_embeds, ngram_memory=ngram_memory, engram_module=engram)

            mse_loss = F.mse_loss(predicted, target_embeds)

            ponder_cost = 0.05 * ponder_steps  # encourage halt gate to stop early for easy tokens

            # Coherence penalty: ngram_memory and predicted should agree in direction.
            # cos_sim in [-1, 1]; penalty = 1 - sim, so opposing vectors get penalized ~2,
            # aligned vectors get ~0. Weight 0.05 keeps it subordinate to mse_loss.
            cos_sim = F.cosine_similarity(predicted, ngram_memory, dim=-1).mean()
            coherence_penalty = 0.05 * (1.0 - cos_sim)

            loss = mse_loss + ponder_cost + coherence_penalty

            loss.backward()

            torch.nn.utils.clip_grad_norm_(all_params, 1.0)  # prevent gradient explosion across 24 effective passes

            optimizer.step()

            scheduler.step()



            # Write gradient-updated embeddings back to cache

            if batch_embed.grad is not None:

                with torch.no_grad():

                    updated = batch_embed - EMBED_LR * batch_embed.grad

                    for w, i in batch_idx.items():

                        embed_cache[w] = updated[i].tolist()



            total_loss += loss.item()

            total_ponder_steps += ponder_steps

            n_steps += 1



            if batch_start % (BATCH_SIZE * 50) == 0:

                avg = total_loss / n_steps

                pct = batch_start / len(sequences) * 100

                print(f"  Epoch {epoch + 1} | {pct:4.0f}% | Avg Loss: {avg:.4f}")



        avg_loss = total_loss / n_steps

        avg_ponder = total_ponder_steps / n_steps

        print(f"  Epoch {epoch + 1} complete ??? Avg Loss: {avg_loss:.4f} | Avg Ponder Steps: {avg_ponder:.2f}")

        # Per-epoch checkpoint save — defense against subprocess/function timeout.
        # Saves brain + engram + word_to_id every epoch; the final ChromaDB write
        # at the bottom of training only happens on clean completion. If the run
        # is killed mid-training, we still recover the most recent completed-epoch
        # weights (without normalized embeddings — those get re-normalized at load).
        try:
            brain_state = {k: v.detach().cpu() for k, v in brain.state_dict().items()}
            engram_state = {k: v.detach().cpu() for k, v in engram.state_dict().items()}
            torch.save(brain_state, "engram_weights.pth")
            torch.save(engram_state, "engram_memory_module.pth")
            torch.save(word_to_id, "engram_word_to_id.pth")
            print(f"  [checkpoint] saved post-epoch-{epoch + 1} weights to volume")
        except Exception as ckpt_err:
            print(f"  [checkpoint] WARNING: per-epoch save failed: {ckpt_err}")



    # Normalize embeddings before saving

    print(f"\nNormalizing {len(embed_cache):,} concept vectors...")

    for w in embed_cache:

        vec = torch.tensor(embed_cache[w])

        embed_cache[w] = F.normalize(vec, dim=0).tolist()



    # Flush all learned embeddings to ChromaDB

    print(f"Saving {len(embed_cache):,} concept vectors to ChromaDB...")

    ids = list(embed_cache.keys())

    embeds = [embed_cache[w] for w in ids]

    # ChromaDB add() has a limit per call; batch it

    chunk = 5000

    for i in range(0, len(ids), chunk):

        vocab_collection.add(ids=ids[i:i+chunk], embeddings=embeds[i:i+chunk], documents=ids[i:i+chunk])



    brain.cpu()
    engram.cpu()
    torch.save(brain.state_dict(), "engram_weights.pth")
    torch.save(engram.state_dict(), "engram_memory_module.pth")
    torch.save(word_to_id, "engram_word_to_id.pth")

    print(f"Brain saved  ??? engram_weights.pth")
    print(f"Engram module saved ??? engram_memory_module.pth ({engram_params:,} params)")
    print(f"Word-to-ID map saved ??? engram_word_to_id.pth ({len(word_to_id):,} words)")

    print(f"Vocab saved  ??? {CHROMA_PATH}/ ({len(embed_cache):,} concepts)")

    print("Run test_brain.py to chat.")





if __name__ == "__main__":

    main()





