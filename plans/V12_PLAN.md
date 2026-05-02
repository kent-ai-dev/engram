# v12 Plan — Switch the Loss Function

**Status:** active. Launching.
**Date queued:** 2026-05-02 (immediately after v11 deploy)

## Hypothesis being tested

> Engram's MSE-on-embeddings training objective is the bottleneck for coherence — not capacity, not corpus volume.

**v9 refuted capacity** (3.5× params on same corpus produced same word-salad). **v11 refuted corpus volume** (vocab grew 9,509 → 14,704 with everyday-conversations included; same word-salad). The two obvious knobs are exhausted. What's left is the structural choice that makes engram *non-standard*: it predicts a continuous concept embedding and minimizes MSE, instead of predicting categorical token logits and minimizing softmax cross-entropy.

Engram's architecture asks "what *direction* in concept space comes next?". Standard LMs ask "*which* of the V tokens comes next?". The first is regression, the second is classification. **For nominal-categorical data like tokens, regression rewards predicting the average of plausible neighbors — exactly the word-salad failure mode we observe across v6–v11.**

## Research basis (web-confirmed)

- **["Cross-entropy consistently outperforms MSE for training classifiers, and you will never see MSE used as the training objective for a language model."](https://dataloopr.com/blog/cross-entropy-vs-mse-choosing-the-right-loss-function-in-ml-18/)** Direct quote.
- Gradient strength: at p=0.01 with true label 1, cross-entropy gradient = 100, MSE gradient = 1.98. ~50× stronger wrong-prediction signal.
- Cross-entropy is mathematically the maximum-likelihood objective for categorical data. MSE is the MLE objective for *Gaussian* targets — wrong assumption for tokens.
- [SmolLM2](https://arxiv.org/abs/2502.02737), the closest "tiny LM done right" reference, uses standard cross-entropy. No regression-on-embeddings anywhere in its lineage.
- [Recent number-token-loss research](https://arxiv.org/html/2411.02083v2) *adds* MSE-like terms *to* cross-entropy for arithmetic — never replaces cross-entropy.

## What's frozen from v11

- Architecture: 384D / 12L / 12H / RoPE / Pre-LN / AdamW (21.5M brain params)
- Corpus: `dailydialog_clean.txt` + `everyday_conversations.txt` (5.3 MB combined)
- BATCH_SIZE=96, BRAIN_LR=1e-3, EMBED_LR=5e-4, EPOCHS=5, NGRAM_TABLE_SIZE=100003
- N-gram memory module (EngramModule)
- Per-epoch checkpoint save logic in `ingest.py`
- Modal function timeout 12h, subprocess timeout 11h40m

**Single-variable test.** The only thing changing is the loss function in `ingest.py`.

## What changes

### Old loss (MSE + coherence penalty)

```python
predicted, ponder_steps = brain(ctx_embeds, ngram_memory=ngram_memory, engram_module=engram)
target_embeds = batch_embed[tgt_idx]   # (B, D)

mse_loss = F.mse_loss(predicted, target_embeds)
ponder_cost = 0.05 * ponder_steps
cos_sim = F.cosine_similarity(predicted, ngram_memory, dim=-1).mean()
coherence_penalty = 0.05 * (1.0 - cos_sim)

loss = mse_loss + ponder_cost + coherence_penalty
```

This is regression on continuous vectors. The coherence penalty was a hack to anchor the regression toward n-gram-memory direction — only meaningful with MSE.

### New loss (temperature-scaled cosine cross-entropy)

```python
# Built ONCE before training:
vocab_words = list(embed_cache.keys())
word_to_global_idx = {w: i for i, w in enumerate(vocab_words)}
vocab_matrix_global = torch.stack([torch.tensor(embed_cache[w]) for w in vocab_words]).to(DEVICE)
vocab_matrix_global = F.normalize(vocab_matrix_global, dim=-1)  # frozen, no grad

# Per batch:
predicted, ponder_steps = brain(ctx_embeds, ngram_memory=ngram_memory, engram_module=engram)
predicted_norm = F.normalize(predicted, dim=-1)
target_global_idx = torch.tensor([word_to_global_idx[w] for w in target_words]).to(DEVICE)

# Cosine-similarity logits: dot(predicted_norm, vocab_norm.T) ∈ [-1, 1]
# Multiply by inverse-temperature to sharpen the softmax distribution.
logits = (predicted_norm @ vocab_matrix_global.T) * INV_TEMPERATURE   # (B, V)

ce_loss = F.cross_entropy(logits, target_global_idx)
ponder_cost = 0.05 * ponder_steps

loss = ce_loss + ponder_cost
```

**Key implementation choices:**

1. **Frozen vocab matrix.** Embeddings come from ChromaDB (sentence-transformer-pretrained). Pre-v12 code refined them per batch via gradient write-back. We're dropping that. Why: (a) refinement made the target a moving target, hurting convergence stability, (b) cross-entropy classification is well-defined against a fixed target distribution, (c) sentence-transformer embeddings are already semantically meaningful — refinement was probably hurting more than helping.
2. **Cosine similarity as logit (with temperature).** Both prediction and vocab are L2-normalized, so dot product = cosine similarity ∈ [-1, 1]. Without temperature, softmax over [-1, 1] is too flat (max prob ≈ 0.88 vs uniform 1/V). `INV_TEMPERATURE=10.0` (= temperature 0.1) sharpens it. CLIP uses learned temperatures around this scale; we hardcode for simplicity in v12.
3. **Inference geometry stays compatible.** Existing generation in `eval_chat.py` / `server.py` picks nearest by cosine — argmax of cosine similarities = argmax of softmax(scaled cosine). Already aligned, no inference-time change needed.
4. **N-gram memory module unchanged.** Still feeds into the brain's attention via `engram_module=engram` arg. Just no longer constrains the loss directly.

## Eval criteria (unchanged)

1. Different replies for different prompts
2. ≥80% real-English tokens
3. ≥3 of 5 chitchat prompts produce coherent dialog-shaped replies

If v12 passes all three: emit `<promise>ENGRAM_COHERENT</promise>`, ship to live frontend, archive the autonomous loop. The MSE-on-embeddings hypothesis is confirmed as the root cause.

## What if v12 still fails

Then the loss function is *not* the bottleneck either, and the diagnosis points at one of:

1. **The vocab embeddings themselves.** Sentence-transformer pretrained vectors form a topology optimized for semantic similarity — possibly too smooth for syntactic patterns. v13 would test learnable token embeddings (initialized from ChromaDB but trainable, with gradients flowing through).
2. **Generation strategy.** Top-k nearest-cosine with penalty might be flattening the distribution at inference even after training fixed it. v13 alternate would switch to softmax sampling on the new logits.
3. **Attention masking bug.** Worth re-auditing. Unlikely after this many runs but the cost of looking is low.

We do not pre-commit to any of these — diagnose first.

## Cost ceiling

Cumulative spend through v11-redux ≈ $40 of the $50 ceiling. v12 estimated ~$5-6 (same wall-time as v11 — loss change doesn't affect throughput). After v12, we'd be ~$45 — at the ceiling. **If v12 fails, the next decision requires user approval to extend the budget.**

## Files involved

- `ingest.py` — loss block patched (lines 384-394 of pre-v12 code)
- `configs/engram_v12_config.json` — new
- `plans/V12_PLAN.md` — this file

## What's intentionally NOT changed

- Architecture (frozen)
- Corpus (frozen — same as v11)
- Generation strategy (cosine-aligned with new training objective)
- N-gram memory module
- Optimizer, learning rates, batch size, epochs
- Per-epoch checkpoint logic (already there)
- Modal timeouts (already 12h / 11h40m)
