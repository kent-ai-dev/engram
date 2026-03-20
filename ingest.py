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



EMBED_DIM = 128

CONTEXT_SIZE = 8

N_LAYERS = 6

BATCH_SIZE = 512

BRAIN_LR = 3e-3

EMBED_LR = 1e-3

EPOCHS = 1

CHROMA_PATH = "./engram_memory"

SPECIAL_TOKENS = ["<START>", "<USER>", "<BOT>"]





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





class AttentionBrain(nn.Module):

    """

    Fixed-size reasoning engine ??? completely vocab-independent.

    Size is O(embed_dim^2 * n_layers). Does not grow as vocabulary grows.

    Includes adaptive pondering: loops through blocks up to max_ponder times,

    with a learned halt gate deciding when to stop.

    """

    def __init__(self, embed_dim=EMBED_DIM, context_size=CONTEXT_SIZE, n_layers=N_LAYERS, max_ponder=3):

        super().__init__()

        self.pos_embed = nn.Embedding(context_size, embed_dim)

        self.blocks = nn.ModuleList([AttentionBlock(embed_dim, context_size) for _ in range(n_layers)])

        self.ln_final = nn.LayerNorm(embed_dim)

        self.halt_gate = nn.Linear(embed_dim, 1)  # ~65 new params

        self.max_ponder = max_ponder



    def forward(self, x):

        # x: (B, T, D) ??? raw concept vectors from ChromaDB

        # Returns: (prediction, n_steps) where prediction is (B, D)

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



        output = output + remaining * last_token  # distribute remaining mass

        return self.ln_final(output), n_steps





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



    brain = AttentionBrain()

    optimizer = optim.Adam(brain.parameters(), lr=BRAIN_LR)

    print(f"Brain parameters: {sum(p.numel() for p in brain.parameters()):,} (fixed regardless of vocab size)")



    n_batches = (len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nTraining: {len(sequences):,} sequences -> {n_batches:,} batches/epoch x {EPOCHS} epochs\n")



    brain.train()

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

            ).requires_grad_(True)



            # (B, T) index tensors ??? (B, T, D) and (B, D)

            ctx_idx = torch.tensor([[batch_idx[w] for w in cw] for cw in ctx_word_lists])

            tgt_idx = torch.tensor([batch_idx[w] for w in target_words])



            ctx_embeds = batch_embed[ctx_idx]      # (B, T, D)

            target_embeds = batch_embed[tgt_idx]   # (B, D)



            optimizer.zero_grad()

            predicted, ponder_steps = brain(ctx_embeds)           # (B, D), int

            mse_loss = F.mse_loss(predicted, target_embeds)

            ponder_cost = 0.01 * ponder_steps  # encourage efficiency

            loss = mse_loss + ponder_cost

            loss.backward()

            optimizer.step()



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



    torch.save(brain.state_dict(), "engram_weights.pth")

    print(f"Brain saved  ??? engram_weights.pth")

    print(f"Vocab saved  ??? {CHROMA_PATH}/ ({len(embed_cache):,} concepts)")

    print("Run test_brain.py to chat.")





if __name__ == "__main__":

    main()





