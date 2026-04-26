"""
diag_constant_output.py — isolate where the v5_rope model collapses input
diversity into a near-constant output.

Layered probes, cheapest first:
  A. Input sanity     — context tensors actually differ?
  B. Per-block trace  — does each AttentionBlock preserve input differentiation?
  C. Pondering trace  — does the halt gate collapse output?
  D. Random-init      — does an UNTRAINED model show the same bug? (architecture vs weights)
  E. Last-position    — does only the last position matter, or are earlier positions zeroed?
  F. Sensitivity      — what minimum input perturbation produces a noticeable output change?

Run from /mnt/c/Users/Administrator/Documents/Github/engram.
"""

import torch
import torch.nn.functional as F
import chromadb
from engram_model import (
    AttentionBrain, AttentionBlock, EngramModule,
    EMBED_DIM, CONTEXT_SIZE, N_LAYERS, N_HEADS,
)

# ---------------------------------------------------------------------------
# Setup: load real vocab + trained model
# ---------------------------------------------------------------------------
chroma = chromadb.PersistentClient(path="./engram_memory")
data = chroma.get_collection("engram_vocab").get(include=["embeddings"])
embed_cache = {w: list(e) for w, e in zip(data["ids"], data["embeddings"])}

state_dict = torch.load("./engram_weights.pth", weights_only=True)
trained = AttentionBrain(EMBED_DIM, CONTEXT_SIZE, N_LAYERS, use_rope=True)
trained.load_state_dict(state_dict, strict=False)
trained.eval()

torch.manual_seed(0)
fresh = AttentionBrain(EMBED_DIM, CONTEXT_SIZE, N_LAYERS, use_rope=True)
fresh.eval()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ctx_tensor(words):
    """Build a (1, CONTEXT_SIZE, EMBED_DIM) tensor from words, left-padded with <START>."""
    pad = ["<START>"] * CONTEXT_SIZE + words
    pad = pad[-CONTEXT_SIZE:]
    return torch.stack([
        torch.tensor(embed_cache.get(w, [0.0] * EMBED_DIM), dtype=torch.float32)
        for w in pad
    ]).unsqueeze(0), pad

def cos(a, b):
    return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

def report(name, a, b):
    print(f"  {name:50s} cos={cos(a, b):.6f}  ||Δ||={(a-b).norm().item():.6e}")

A_words = ["hello", "how", "are", "you"]
B_words = ["tell", "me", "a", "story"]
C_words = ["the", "cat", "sat", "on", "the", "mat", "yesterday"]

xA, ctxA = ctx_tensor(A_words)
xB, ctxB = ctx_tensor(B_words)
xC, ctxC = ctx_tensor(C_words)

# ---------------------------------------------------------------------------
# A. Input sanity
# ---------------------------------------------------------------------------
print("\n=== A. INPUT SANITY ===")
print(f"  xA shape: {tuple(xA.shape)}, xB shape: {tuple(xB.shape)}")
print(f"  Last position word — A: {ctxA[-1]!r}, B: {ctxB[-1]!r}, C: {ctxC[-1]!r}")
print(f"  Last token embedding norms — A: {xA[0,-1].norm():.4f}, B: {xB[0,-1].norm():.4f}, C: {xC[0,-1].norm():.4f}")
report("xA full vs xB full",     xA, xB)
report("xA[-1] vs xB[-1]",       xA[0, -1], xB[0, -1])
report("xA[28..31] vs xB[28..31]", xA[0, 28:32], xB[0, 28:32])

# ---------------------------------------------------------------------------
# B. Per-block trace (trained model)
# ---------------------------------------------------------------------------
print("\n=== B. PER-BLOCK TRACE (trained) ===")
print("  Tracking cos at position -1 across all layers (no pondering, just blocks)\n")
with torch.no_grad():
    a, b, c = xA.clone(), xB.clone(), xC.clone()
    print(f"  layer 0 (input)      cos(a,b)={cos(a[0,-1], b[0,-1]):.6f}  norm_a[-1]={a[0,-1].norm():.4f}")
    for i, blk in enumerate(trained.blocks):
        a = blk(a); b = blk(b); c = blk(c)
        cab = cos(a[0,-1], b[0,-1])
        cac = cos(a[0,-1], c[0,-1])
        # Also check earlier positions to detect "everything pinned"
        c_earlier = cos(a[0, 15], b[0, 15])
        print(f"  after layer {i+1:2d}     cos(a,b)[-1]={cab:.6f}  cos(a,c)[-1]={cac:.6f}  cos(a,b)[15]={c_earlier:.6f}  norm_a[-1]={a[0,-1].norm():.2f}")

# ---------------------------------------------------------------------------
# C. Pondering trace (full forward) — does halt gate or pondering collapse output?
# ---------------------------------------------------------------------------
print("\n=== C. FULL FORWARD WITH AND WITHOUT PONDERING ===")
with torch.no_grad():
    # Default forward (with pondering)
    pa, sa = trained(xA)
    pb, sb = trained(xB)
    print(f"  default forward      ponder=({sa},{sb})")
    report("    output pa vs pb",  pa, pb)

    # Force max_ponder = 1 (no pondering)
    trained.max_ponder = 1
    pa1, _ = trained(xA)
    pb1, _ = trained(xB)
    report("    max_ponder=1",     pa1, pb1)

    # Force max_ponder = 3 again
    trained.max_ponder = 3

    # Skip the halt-gate output accumulator: read x[:,-1,:] directly after blocks (no pondering loop)
    # Simulating one forward pass through blocks + final LN
    a, b = xA.clone(), xB.clone()
    for blk in trained.blocks:
        a = blk(a); b = blk(b)
    last_a = trained.ln_final(a[:, -1, :])
    last_b = trained.ln_final(b[:, -1, :])
    report("    raw last-pos +ln",  last_a, last_b)

# ---------------------------------------------------------------------------
# D. Random-init model — does the bug also exist with untrained weights?
# ---------------------------------------------------------------------------
print("\n=== D. UNTRAINED MODEL (architecture-vs-weights) ===")
with torch.no_grad():
    pa, _ = fresh(xA)
    pb, _ = fresh(xB)
    report("  fresh forward(xA) vs forward(xB)", pa, pb)
    a, b = xA.clone(), xB.clone()
    for blk in fresh.blocks:
        a = blk(a); b = blk(b)
    report("  fresh raw last-pos               ", a[:, -1, :], b[:, -1, :])

# ---------------------------------------------------------------------------
# E. Single-block sensitivity check (trained block 0)
# ---------------------------------------------------------------------------
print("\n=== E. SINGLE BLOCK SENSITIVITY (trained block 0) ===")
with torch.no_grad():
    block0 = trained.blocks[0]
    a, b = xA.clone(), xB.clone()
    out_a = block0(a)
    out_b = block0(b)
    print(f"  before block 0  cos[-1]={cos(a[0,-1], b[0,-1]):.6f}")
    print(f"  after  block 0  cos[-1]={cos(out_a[0,-1], out_b[0,-1]):.6f}")
    print(f"  before block 0  cos[15]={cos(a[0,15], b[0,15]):.6f}")
    print(f"  after  block 0  cos[15]={cos(out_a[0,15], out_b[0,15]):.6f}")

# ---------------------------------------------------------------------------
# F. Sensitivity to a single-position perturbation
# ---------------------------------------------------------------------------
print("\n=== F. SENSITIVITY TO PERTURBATION ===")
with torch.no_grad():
    base = xA.clone()
    perturb = base.clone()
    perturb[0, -1] = perturb[0, -1] + torch.randn_like(perturb[0, -1]) * 0.5  # ~50% noise on last token
    pa, _ = trained(base)
    pb, _ = trained(perturb)
    report("  output(xA) vs output(xA+noise)",       pa, pb)

    # Replace last position with a totally different word
    pa, _ = trained(xA)
    xA_alt = xA.clone()
    if "story" in embed_cache:
        xA_alt[0, -1] = torch.tensor(embed_cache["story"], dtype=torch.float32)
    pa_alt, _ = trained(xA_alt)
    report("  output(xA) vs output(xA, last←story)", pa, pa_alt)

print("\nDone. Read the columns above to localize where input differentiation collapses.")
