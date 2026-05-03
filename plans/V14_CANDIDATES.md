# v14 Candidates — Decision Tree Keyed on v13 Eval

**Status:** drafted while v13_xent_temp30 trains (2026-05-03).
**Decide:** after v13 download + `eval_chat.py` produces a transcript.
**Principle:** v14 must exercise an **engram architectural lever**, not another textbook ML knob. The autonomous loop has spent 4 iterations (v9–v12) ruling out conventional explanations (capacity, corpus, loss function). v13 closes out the loss-temperature dimension. From here, every iteration should test a claim that would be impossible to make about a stock transformer.

## What v13 will tell us

v13 holds architecture and corpus identical to v12 and only changes `INV_TEMPERATURE` 10 → 30. Three observable outcomes:

| v13 eval shape | Diagnosis | v14 branch |
|----------------|-----------|------------|
| **PASS** — distinct + ≥80% English + ≥3/5 chitchat coherent | Loss temperature was the calibration block. Every prior architectural innovation was *latent* — already correct, just under-trained. | **Archive loop.** Emit `ENGRAM_COHERENT`. v14 is unnecessary. Document the calibration lesson and ship. |
| **PARTIAL** — distinct, mostly real English, sentence-shape but not yet coherent | Temperature dimension exhausted; the model has the right tokens but not the right *compute allocation* per token. | **Branch A: Stronger Adaptive Pondering** (engram innovation #3 — adaptive compute) |
| **FAIL** — clean loss plateau (~0.6 nats, near floor) but eval is still word-salad | The cross-entropy classifier converged against the available embedding geometry — there's no more signal to extract from frozen sentence-transformer vectors. | **Branch B: Learnable ChromaDB Embeddings** (engram innovation #1 — vocab/brain separation, vocab geometry as bottleneck) |
| **FAIL — no convergence** — loss still descending at 5+ nats above floor at epoch 5 | Even sharp xent at INV_TEMP=30 isn't enough. The fixed inputs are starving the gradient. | **Branch C: Episodic Memory at Training Time** (engram innovation #5 — recall as training signal, not just inference garnish) |

The autonomous loop's `current_hypothesis` field gets updated to whichever branch fires, and a `plans/V14_*_PLAN.md` is written before launch.

---

## Branch A — Stronger Adaptive Pondering

### Hypothesis

> v13 produced near-coherent output, but the brain isn't using its adaptive-compute budget. With ponder cap=3 and ponder cost=0.05, the halt gate learns to bail at step 1 because the cost outweighs the marginal loss reduction. Raising the cap and lowering the cost lets the brain "think harder" on novel tokens — the dopaminergic-effort lever that distinguishes engram from a fixed-depth transformer.

### Innovation engaged

**Innovation #3 — Adaptive pondering (PonderNet-style).** Currently the loop count is bounded at 1–3 with a 0.05 cost coefficient that effectively pulls toward 1. v14 makes the lever actually meaningful.

### Single-variable change(s)

In `engram_model.py`:
- `MAX_PONDER_STEPS: 3 → 5`

In `ingest.py`:
- `ponder_cost = 0.05 * ponder_steps → 0.02 * ponder_steps`
- All other hyperparameters frozen at v13 values (`INV_TEMPERATURE=30`)

This is **two coupled changes** rather than strict single-variable, but they're inseparable: raising the cap without lowering the cost just makes the gate bail at 1 anyway.

### Eval signal

- Track `avg_ponder_steps` per prompt in `eval_chat.py` output. v9–v13 show ponder ≈ 3.0 (which means the gate is stuck at the cap, not actually deciding). v14-A success looks like ponder varying meaningfully across prompts (e.g. 1–2 for greetings, 4–5 for "what is the capital of france").

### Cost: ~$8

Slightly higher than v13's $6 because ponder=5 means up to 5x forward passes per token in the backprop graph. Same wall time approximately.

### Failure mode

If avg_ponder still pegs at 5 after training, the halt gate is degenerate — wider/deeper pondering won't help. Escalate to Branch B.

---

## Branch B — Learnable ChromaDB Embeddings

### Hypothesis

> v13 trained the brain perfectly given the constraint that vocab embeddings come from a frozen sentence-transformer. But sentence-transformer geometry optimizes for semantic similarity ("cat" near "dog"), not syntactic prediction ("cat" followed by "is"). The cross-entropy classifier has converged against the wrong target topology. Letting the ChromaDB vectors learn — while keeping the brain frozen at its current weights — tests whether vocab geometry is the residual bottleneck.

### Innovation engaged

**Innovation #1 — Reasoning ≠ vocabulary; ChromaDB-as-vocab is a learnable, swappable component.** This is engram's most foundational architectural bet. We have never actually exercised the "learnable" half of "learnable 96-D coordinates" under cross-entropy training. v12 plan explicitly named this as the fallback if pure xent didn't crack coherence.

### Single-variable change

In `ingest.py` (line ~318, around the `vocab_matrix_global` setup):

```python
# v14-B: vocab_matrix_global becomes a learnable parameter.
# Initialize from current ChromaDB embeddings, then let xent gradient
# refine them via the standard nn.Parameter pathway.
vocab_matrix_global = nn.Parameter(
    torch.tensor([embed_cache[w] for w in vocab_words_global],
                 dtype=torch.float32).to(DEVICE)
)
# Add to optimizer with separate (lower) LR:
optimizer.add_param_group({"params": [vocab_matrix_global], "lr": EMBED_LR * 0.5})
```

Critically: **brain weights initialized from v13 final and frozen for first epoch**, then unfrozen. This isolates whether the bottleneck is vocab vs. brain.

### Eval signal

- Loss should drop substantially past v13's plateau in the first frozen-brain epoch (which would otherwise be impossible — the brain can't change its predictions, only the targets can move toward them).
- After unfreezing: coherence should improve qualitatively if vocab geometry was indeed the issue.
- Diagnostic: post-training, look at neighbors of common words in updated vs. original ChromaDB. If "the" is now near "is", "a", "and" instead of near "this", "that" (semantic), the lever worked.

### Cost: ~$8

Same wall time as v13. Slight memory bump for vocab matrix gradients (V × D = 14704 × 384 = ~22 MB extra).

### Failure mode

If loss still plateaus and eval still fails, both vocab geometry and brain are not the issue — strongly suggests the corpus is informationally exhausted at 5.3 MB. Escalate to corpus expansion (the conventional `smol-smoltalk` route, last resort).

---

## Branch C — Episodic Memory at Training Time

### Hypothesis

> v13 still hadn't converged at epoch 5 because the brain's only signal is `(context_window → next_token)` cross-entropy. The episodic memory module — engram's distinguishing feature — sits dormant during training and only fires at inference. Training with episodic retrieval engaged means the brain learns to *use* its own retrieval system as part of the prediction, not as a post-hoc decoration.

### Innovation engaged

**Innovation #5 — Episodic memory as a first-class citizen.** Currently this is an inference-only feature. v14-C makes it part of the training loop: during each forward pass, retrieve the top-K nearest brain-state episodes from ChromaDB and blend them via the existing learned gate. The brain is then trained end-to-end with retrieval in the loop, so the gate weights and brain layers co-adapt to use recall.

### Single-variable change (substantial implementation lift)

In `ingest.py` per-batch loop:

```python
# v14-C: retrieve episodic memory for each batch element, feed into brain
# alongside the n-gram memory. This requires:
# 1. A ChromaDB episodic collection populated incrementally during training
#    (start empty; add (brain_state, target_embedding) pairs each batch).
# 2. The brain's forward signature already accepts an extra "episodic" tensor
#    via the same gating module as n-gram memory; just wire it in.
# 3. A warmup period: first 10% of epoch 1, no retrieval (collection too empty).
predicted, ponder_steps = brain(
    ctx_embeds,
    ngram_memory=ngram_memory,
    episodic_memory=episodic_retrieve(brain_state, k=3) if epoch_progress > 0.1 else None,
    engram_module=engram,
)
```

This is the most invasive of the three branches — it requires the episodic collection to be writable during training, retrieval to happen on-GPU efficiently, and a warmup window so the gate doesn't see noise before there's anything to retrieve.

### Eval signal

- Loss progression should be *non-monotonic* in epoch 1 (warmup → retrieval kicks in → brief loss spike as gate adjusts → resumed descent). If loss stays smoothly monotonic, retrieval isn't actually being used.
- Eval transcript should show the model "remembering" prior turns within an interaction — a behavior no other variant has produced.
- Post-training, the episodic gate's mean activation tells you whether recall is being used (gate ≈ 0 = unused, gate > 0.3 = active).

### Cost: ~$12

Higher because each batch step now does an extra ChromaDB nearest-neighbor query (~50ms per batch on L4). Retrieval-in-the-loop is genuinely more compute. Also higher dev time before launch — needs a small prototype run to make sure the retrieval mechanics don't crash.

### Failure mode

If episodic gate weights collapse to ~0 by end of training, the model decided retrieval wasn't useful. That's also informative — it would say the architecture's bet on episodic memory needs rethinking.

---

## What's intentionally NOT chosen

### Corpus expansion (`smol-smoltalk` 50–80 MB subsample)

Stays on the back burner. Reason: it's the most engram-distinguishing-irrelevant move available. If we add 10× more data and the model becomes coherent, we have learned **nothing about whether engram's architectural bets are paying off** — only that data was the binding constraint, which would be true for any architecture. We use this only after Branches A and B (and possibly C) have all been exhausted, as the "if everything architectural is right, then maybe just data" check.

### Further temperature sweep (INV_TEMP 30 → 100)

The next stop after v13 in pure-hyperparameter space, but: if INV_TEMP=30 already drives loss within a few nats of its theoretical floor (0.59), the calibration dimension is essentially closed. Pushing to 100 gives a smaller incremental gain and still exercises no architectural lever. Skip.

### Larger model (more layers, more dim)

v9 already refuted capacity (3.5× params on same corpus produced same word-salad). Re-doing this with v13's better loss would mix two variables — better to validate the loss-fix at small scale first, then scale once.

---

## Cost ledger projection

| Run | Cumulative spend (USD) | Of $150 ceiling |
|-----|------------------------|-----------------|
| v9–v12 (already spent) | ~45 | 30% |
| v13 (in flight) | ~51 | 34% |
| v14 (Branch A or B) | ~59 | 39% |
| v14-C if chosen | ~63 | 42% |
| v15 fallback (corpus) | ~71-75 | 47-50% |

We have substantial runway under the $150 ceiling — at least 5 more iterations possible at current per-run cost.

---

## Update flow when v13 lands

1. `eval_chat.py` produces transcript at `eval_runs/chat_<ts>.json`.
2. Read transcript, classify into PASS / PARTIAL / FAIL-clean-plateau / FAIL-no-convergence.
3. Based on table above, pick branch.
4. Write `plans/V14_<branch>_PLAN.md` (full single-variable spec, copy from this doc's branch section + add hard-won-lessons).
5. Update `.claude/ralph-loop.local.md`: bump `current_hypothesis`, `current_run`, increment iteration.
6. Edit `ingest.py` (and `engram_model.py` for Branch A) for the chosen single-variable swap.
7. Commit + push to main (Modal git-clones main).
8. Launch via the standard Modal direct-CLI pattern (the `modal_train.py launch --config` wrapper has a flag-passing bug; use `python3 -m modal run --detach scripts/modal_train.py --config-path <file>` directly).
9. ScheduleWakeup 1800s for the next loop tick.
