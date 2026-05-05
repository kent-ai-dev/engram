import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 256
CONTEXT_SIZE = 32
N_LAYERS = 8
N_HEADS = 8
NGRAM_TABLE_SIZE = 50021
SPECIAL_TOKENS = ["<START>", "<USER>", "<BOT>"]


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0, device=None) -> torch.Tensor:
    """Precompute complex RoPE frequencies. Returns (max_seq_len, head_dim // 2) complex64.
    Ported from OpenMythos open_mythos/main.py lines 124-169.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs_2d = torch.outer(t, freqs)            # (max_seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(freqs_2d), freqs_2d)  # complex64


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to Q or K.
    x: (B, n_heads, T, head_dim)
    freqs_cis: (max_seq_len, head_dim // 2) complex64
    """
    B, H, T, D = x.shape
    x_c = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    fc = freqs_cis[:T].unsqueeze(0).unsqueeze(0)   # (1, 1, T, D//2)
    return torch.view_as_real(x_c * fc).reshape(B, H, T, D).to(x.dtype)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, context_size, n_heads=N_HEADS, use_rope=False):
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
        # Precompute at 4× context_size when RoPE is on to allow inference-time extrapolation.
        max_len = context_size * 4 if use_rope else context_size
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("mask", mask)
        self.use_rope = use_rope
        if use_rope:
            freqs_cis = precompute_rope_freqs(self.head_dim, max_len)
            self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x):
        # Pre-LN architecture: LN BEFORE each sublayer, residual added without LN.
        # Switched from post-LN after v5 diagnostic showed input differentiation
        # collapsed inside block 0: attention softmax went to near-uniform and
        # the FF residual was 22× larger than its input, drowning out the signal.
        B, T, D = x.size()

        # Attention sublayer
        h = self.ln1(x)
        Q, K, V = self.W_q(h), self.W_k(h), self.W_v(h)
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            Q = apply_rope(Q, self.freqs_cis)
            K = apply_rope(K, self.freqs_cis)
        scale = self.head_dim ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        scores = scores.masked_fill(self.mask[:T, :T].unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(out)
        x = x + out

        # Feed-forward sublayer
        x = x + self.ff(self.ln2(x))
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


class LTIInjection(nn.Module):
    """LTI-stable input injection — re-anchors the residual stream to the
    original input every ponder iteration, preventing drift across loop depth.

    Diagonal form: one log_A per channel, one scalar log_dt.
    A = exp(-exp(log_A)) is always in (0, 1).
    x_new = A * x + (1 - A) * (e + dt * delta)

    Ported from OpenMythos open_mythos/main.py lines 684-742.
    Init: log_A = -2.0 → A ≈ 0.87 (near identity, avoids A collapsing to ~0).
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.full((embed_dim,), -2.0))
        self.log_dt = nn.Parameter(torch.zeros(1))

    def get_A(self) -> torch.Tensor:
        return torch.exp(-torch.exp(self.log_A))

    def forward(self, x: torch.Tensor, e: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        # x: current hidden state (B, T, D)
        # e: original pos-embedded input frozen before ponder loop (B, T, D)
        # delta: blocks(x_before) - x_before, the transformer residual (B, T, D)
        A = self.get_A()                    # (D,)
        dt = torch.exp(self.log_dt)         # scalar > 0
        return A * x + (1.0 - A) * (e + dt * delta)


def loop_index_embedding(x: torch.Tensor, loop_idx: int, loop_dim: int = 32) -> torch.Tensor:
    """Add sinusoidal loop-index signal to x.
    Ported from OpenMythos open_mythos/main.py lines 541-570.
    loop_dim channels encode which ponder iteration we're in; rest are zero.
    """
    B, T, D = x.shape
    half = loop_dim // 2
    div_term = torch.exp(
        torch.arange(0, half, dtype=torch.float32, device=x.device)
        * -(math.log(10000.0) / half)
    )
    t = torch.tensor(float(loop_idx), device=x.device)
    emb = torch.zeros(D, device=x.device)
    emb[:half] = torch.sin(t * div_term)
    emb[half:loop_dim] = torch.cos(t * div_term)
    return x + emb.unsqueeze(0).unsqueeze(0)


class AttentionBrain(nn.Module):
    """Fixed-size reasoning engine — completely vocab-independent.
    Size is O(embed_dim^2 * n_layers). Does not grow as vocabulary grows.
    Includes adaptive pondering: loops through blocks up to max_ponder times,
    with a learned halt gate deciding when to stop.
    """
    def __init__(self, embed_dim=EMBED_DIM, context_size=CONTEXT_SIZE, n_layers=N_LAYERS,
                 max_ponder=5, use_lti=False, use_loop_idx=False, use_rope=True,
                 n_heads=N_HEADS):
        super().__init__()
        self.pos_embed = nn.Embedding(context_size, embed_dim)
        self.use_rope = use_rope
        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, context_size, n_heads=n_heads, use_rope=use_rope)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.halt_gate = nn.Linear(embed_dim, 1)
        self.max_ponder = max_ponder
        self.injection = LTIInjection(embed_dim) if use_lti else None
        self.use_loop_idx = use_loop_idx
        self.loop_dim = embed_dim // 8  # 32 for embed_dim=256

    def forward(self, x, ngram_memory=None, engram_module=None):
        # x: (B, T, D) — raw concept vectors from ChromaDB
        # Returns: (prediction, n_steps) where prediction is (B, D)
        T = x.size(1)
        if not self.use_rope:
            positions = torch.arange(T, dtype=torch.long, device=x.device)
            x = x + self.pos_embed(positions).unsqueeze(0)

        # Freeze input after positional embedding for LTI re-anchoring
        e = x if self.injection is not None else None

        output = torch.zeros_like(x[:, -1, :])
        remaining = torch.ones(x.size(0), 1, device=x.device)
        n_steps = 0

        for ponder_idx in range(self.max_ponder):
            if self.use_loop_idx:
                x = loop_index_embedding(x, ponder_idx, self.loop_dim)

            x_before = x
            for block_idx, block in enumerate(self.blocks):
                x = block(x)
                if block_idx == 0 and ngram_memory is not None and engram_module is not None:
                    gated_mem = engram_module.gate(x[:, -1, :], ngram_memory)
                    x = x.clone()
                    x[:, -1, :] = x[:, -1, :] + gated_mem

            if self.injection is not None:
                delta = x - x_before
                x = self.injection(x, e, delta)

            last_token = x[:, -1, :]
            halt_prob = torch.sigmoid(self.halt_gate(last_token))

            output = output + remaining * last_token
            remaining = remaining * (1 - halt_prob)
            n_steps += 1

            if not self.training and remaining.max().item() < 0.05:
                break

        output = output + remaining * last_token
        return self.ln_final(output), n_steps
