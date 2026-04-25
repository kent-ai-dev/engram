import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 256
CONTEXT_SIZE = 32
N_LAYERS = 8
N_HEADS = 8
NGRAM_TABLE_SIZE = 50021
SPECIAL_TOKENS = ["<START>", "<USER>", "<BOT>"]


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, context_size, n_heads=N_HEADS):
        super().__init__()
        assert embed_dim % n_heads == 0, f"embed_dim {embed_dim} not divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
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
        B, T, D = x.size()
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        scores = scores.masked_fill(self.mask[:T, :T].unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(out)
        x = self.ln1(x + out)
        x = self.ln2(x + self.ff(x))
        return x


class EngramModule(nn.Module):
    """N-gram embedding tables with learned gating.
    Hashes word sequences (bigrams, trigrams) into embedding tables for O(1) lookup.
    Includes learned gating for both N-gram memory and episodic memory injection.
    Based on: 'Conditional Memory via Scalable Lookup' (Cheng et al., 2026).
    """
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
        """Look up N-gram embeddings for a sequence of word IDs.
        word_ids: list of ints (length >= 2).
        Returns: tensor of shape (embed_dim,)
        """
        if len(word_ids) >= 2:
            bh = self.hash_bigram(word_ids[-2], word_ids[-1])
            bigram_emb = self.bigram_table(torch.tensor(bh, device=self.bigram_table.weight.device))
        else:
            bigram_emb = torch.zeros(self.embed_dim // 2, device=self.bigram_table.weight.device)

        if len(word_ids) >= 3:
            th = self.hash_trigram(word_ids[-3], word_ids[-2], word_ids[-1])
            trigram_emb = self.trigram_table(torch.tensor(th, device=self.trigram_table.weight.device))
        else:
            trigram_emb = torch.zeros(self.embed_dim // 2, device=self.trigram_table.weight.device)

        return torch.cat([bigram_emb, trigram_emb], dim=-1)

    def lookup_batch(self, word_id_seqs):
        """Batch lookup for multiple sequences.
        word_id_seqs: list of lists of ints.
        Returns: tensor of shape (B, embed_dim)
        """
        results = [self.lookup(seq) for seq in word_id_seqs]
        return torch.stack(results)

    def gate(self, hidden_state, memory_vector):
        """Context-aware gating: learned blend of memory into hidden state."""
        k = self.W_K(memory_vector)
        v = self.W_V(memory_vector)
        alpha = torch.sigmoid(
            (F.normalize(hidden_state, dim=-1) * F.normalize(k, dim=-1)).sum(-1, keepdim=True)
            / (hidden_state.size(-1) ** 0.5)
        )
        return alpha * v


class AttentionBrain(nn.Module):
    """Fixed-size reasoning engine — completely vocab-independent.
    Size is O(embed_dim^2 * n_layers). Does not grow as vocabulary grows.
    Includes adaptive pondering: loops through blocks up to max_ponder times,
    with a learned halt gate deciding when to stop.
    """
    def __init__(self, embed_dim=EMBED_DIM, context_size=CONTEXT_SIZE, n_layers=N_LAYERS, max_ponder=3):
        super().__init__()
        self.pos_embed = nn.Embedding(context_size, embed_dim)
        self.blocks = nn.ModuleList([AttentionBlock(embed_dim, context_size) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.halt_gate = nn.Linear(embed_dim, 1)
        self.max_ponder = max_ponder

    def forward(self, x, ngram_memory=None, engram_module=None):
        # x: (B, T, D) — raw concept vectors from ChromaDB
        # Returns: (prediction, n_steps) where prediction is (B, D)
        T = x.size(1)
        positions = torch.arange(T, dtype=torch.long, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        output = torch.zeros_like(x[:, -1, :])
        remaining = torch.ones(x.size(0), 1, device=x.device)
        n_steps = 0

        for ponder_idx in range(self.max_ponder):
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
            n_steps += 1

            if not self.training and remaining.max().item() < 0.05:
                break

        output = output + remaining * last_token
        return self.ln_final(output), n_steps
