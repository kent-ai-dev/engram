"""
diag_block0.py — drill into block 0 to find which sub-component collapses
input differentiation. Inspects W_q, W_k, W_v, W_o, attention pattern, FF.
"""

import torch
import torch.nn.functional as F
import chromadb
from engram_model import AttentionBrain, EngramModule, EMBED_DIM, CONTEXT_SIZE, N_LAYERS, apply_rope

chroma = chromadb.PersistentClient(path="./engram_memory")
data = chroma.get_collection("engram_vocab").get(include=["embeddings"])
embed_cache = {w: list(e) for w, e in zip(data["ids"], data["embeddings"])}

state_dict = torch.load("./engram_weights.pth", weights_only=True)
brain = AttentionBrain(EMBED_DIM, CONTEXT_SIZE, N_LAYERS, use_rope=True)
brain.load_state_dict(state_dict, strict=False)
brain.eval()

block = brain.blocks[0]

def ctx_tensor(words):
    pad = (["<START>"] * CONTEXT_SIZE + words)[-CONTEXT_SIZE:]
    return torch.stack([
        torch.tensor(embed_cache.get(w, [0.0] * EMBED_DIM), dtype=torch.float32)
        for w in pad
    ]).unsqueeze(0), pad

def cos(a, b):
    return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

xA, _ = ctx_tensor(["hello", "how", "are", "you"])
xB, _ = ctx_tensor(["tell", "me", "a", "story"])

print("=== Weight magnitudes in block 0 ===")
for name, p in block.named_parameters():
    print(f"  {name:40s} shape={tuple(p.shape)}  norm={p.norm().item():.4f}  max|·|={p.abs().max().item():.4f}")

print("\n=== Step through block 0 forward ===")
with torch.no_grad():
    a, b = xA.clone(), xB.clone()
    print(f"  IN          cos[-1]={cos(a[0,-1], b[0,-1]):+.6f}  norm_a[-1]={a[0,-1].norm():.4f}")

    # 1. Q, K, V projections
    Qa = block.W_q(a); Qb = block.W_q(b)
    Ka = block.W_k(a); Kb = block.W_k(b)
    Va = block.W_v(a); Vb = block.W_v(b)
    print(f"  W_q out     cos[-1]={cos(Qa[0,-1], Qb[0,-1]):+.6f}  norm={Qa[0,-1].norm():.4f}")
    print(f"  W_k out     cos[-1]={cos(Ka[0,-1], Kb[0,-1]):+.6f}  norm={Ka[0,-1].norm():.4f}")
    print(f"  W_v out     cos[-1]={cos(Va[0,-1], Vb[0,-1]):+.6f}  norm={Va[0,-1].norm():.4f}")

    # 2. Multi-head reshape
    B, T, D = Qa.shape
    Qa = Qa.view(B, T, block.n_heads, block.head_dim).transpose(1,2)
    Qb = Qb.view(B, T, block.n_heads, block.head_dim).transpose(1,2)
    Ka = Ka.view(B, T, block.n_heads, block.head_dim).transpose(1,2)
    Kb = Kb.view(B, T, block.n_heads, block.head_dim).transpose(1,2)
    Va = Va.view(B, T, block.n_heads, block.head_dim).transpose(1,2)
    Vb = Vb.view(B, T, block.n_heads, block.head_dim).transpose(1,2)

    # 3. RoPE
    Qa = apply_rope(Qa, block.freqs_cis)
    Kb_ = apply_rope(Kb, block.freqs_cis)
    Ka = apply_rope(Ka, block.freqs_cis)
    Qb = apply_rope(Qb, block.freqs_cis)
    Vb_ = Vb  # V doesn't get RoPE
    print(f"  Q after RoPE last-pos head0  cos={cos(Qa[0,0,-1], Qb[0,0,-1]):+.6f}  norm={Qa[0,0,-1].norm():.4f}")
    print(f"  K after RoPE last-pos head0  cos={cos(Ka[0,0,-1], Kb_[0,0,-1]):+.6f}")

    # 4. Attention scores
    scale = block.head_dim ** 0.5
    scoresA = torch.matmul(Qa, Ka.transpose(-2,-1)) / scale
    scoresB = torch.matmul(Qb, Kb_.transpose(-2,-1)) / scale
    scoresA = scoresA.masked_fill(block.mask[:T,:T].unsqueeze(0).unsqueeze(0)==0, float("-inf"))
    scoresB = scoresB.masked_fill(block.mask[:T,:T].unsqueeze(0).unsqueeze(0)==0, float("-inf"))
    attnA = F.softmax(scoresA, dim=-1)
    attnB = F.softmax(scoresB, dim=-1)
    print(f"  attn[-1] head0 row sum={attnA[0,0,-1].sum().item():.4f}, max={attnA[0,0,-1].max().item():.4f}, argmax={attnA[0,0,-1].argmax().item()}")
    print(f"  attnA vs attnB at pos -1 head 0  cos={cos(attnA[0,0,-1], attnB[0,0,-1]):+.6f}")
    print(f"  attn[-1] last-row delta ||A-B||  ={(attnA[0,0,-1] - attnB[0,0,-1]).norm().item():.6f}")

    # 5. Weighted sum -> attention output
    outA = torch.matmul(attnA, Va).transpose(1,2).contiguous().view(B, T, D)
    outB = torch.matmul(attnB, Vb_).transpose(1,2).contiguous().view(B, T, D)
    print(f"  attention out (pre W_o) cos[-1]={cos(outA[0,-1], outB[0,-1]):+.6f}  norm={outA[0,-1].norm():.4f}")

    # 6. W_o projection
    outA_o = block.W_o(outA); outB_o = block.W_o(outB)
    print(f"  W_o out                 cos[-1]={cos(outA_o[0,-1], outB_o[0,-1]):+.6f}  norm={outA_o[0,-1].norm():.4f}")

    # 7. residual + LN1
    res_a = a + outA_o; res_b = b + outB_o
    print(f"  residual (x + W_o.out)  cos[-1]={cos(res_a[0,-1], res_b[0,-1]):+.6f}  norm={res_a[0,-1].norm():.4f}")
    ln1_a = block.ln1(res_a); ln1_b = block.ln1(res_b)
    print(f"  after LN1               cos[-1]={cos(ln1_a[0,-1], ln1_b[0,-1]):+.6f}  norm={ln1_a[0,-1].norm():.4f}")

    # 8. FF + residual + LN2
    ff_a = block.ff(ln1_a); ff_b = block.ff(ln1_b)
    print(f"  FF out                  cos[-1]={cos(ff_a[0,-1], ff_b[0,-1]):+.6f}  norm={ff_a[0,-1].norm():.4f}")
    out_a = block.ln2(ln1_a + ff_a); out_b = block.ln2(ln1_b + ff_b)
    print(f"  after LN2 (final blk0)  cos[-1]={cos(out_a[0,-1], out_b[0,-1]):+.6f}  norm={out_a[0,-1].norm():.4f}")

    # 9. Sanity: ratio of W_o out vs residual at the critical step
    print(f"\n  Magnitude ratio: ||W_o.out|| / ||x|| at pos -1 = "
          f"{outA_o[0,-1].norm().item() / a[0,-1].norm().item():.2f}")
    print(f"  Magnitude ratio: ||FF.out|| / ||LN1.out|| at pos -1 = "
          f"{ff_a[0,-1].norm().item() / ln1_a[0,-1].norm().item():.2f}")

print("\nWhich substep first collapsed cos[-1] toward 1.0?")
