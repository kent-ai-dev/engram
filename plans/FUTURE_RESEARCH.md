# Future Research — v15+ Candidates

**When to consult this doc:** after V14_CANDIDATES.md branches A/B/C/D have all been tested and either ENGRAM_COHERENT was emitted (in which case archive) or none unlocked coherence (in which case escalate to one of the ideas here).

**Why not in v14 plan:** v14 is single-variable swaps against existing code. Each idea here requires non-trivial implementation work — between a few hundred lines and a partial rewrite of the training loop. Premature inclusion would dilute the single-variable discipline the loop relies on.

**Sequencing principle:** each v15+ candidate has prerequisites among v14 branches. Don't escalate to PCN before testing Branch D's simpler scalar surprise modulation. Don't escalate to TTT before Branch C confirms the existing episodic mechanism even works in a training loop.

---

## Candidate 1: ∇-Reasoner — Test-Time Gradient Descent in Latent Space

**Source:** "LLM Reasoning via Test-Time Gradient Descent in Latent Space" (March 2026)

**Hypothesis:** Engram's predicted concept vector is mapped to ChromaDB via nearest-cosine. Inserting a few steps of gradient descent on that vector against an energy function before the lookup refines the prediction without retraining.

**Engram axis:** Cuts across innovations #1 (vocab as separable component) and #3 (adaptive compute) — the energy minimization is itself a form of test-time reasoning in the concept space.

**Why this is the standout of the four:** zero training-time changes. Pure inference enhancement. Could be prototyped against any existing v13/v14 weights. Plays to engram's actual structural advantage: unlike a stock transformer, engram already has an explicit concept-vector intermediate representation that's a natural target for refinement.

**Single-variable spec (eval_chat.py + server.py inference path):**

```python
# After: predicted, n_steps = brain(...)
predicted = predicted.requires_grad_(True)
for _ in range(N_REFINE_STEPS):
    # Energy = -cosine(predicted_norm, ngram_memory_norm)
    # Plus optional: distance to episodic memory state, halt-gate confidence
    energy = -F.cosine_similarity(
        F.normalize(predicted, dim=-1),
        F.normalize(ngram_memory.squeeze(0), dim=-1),
        dim=-1,
    )
    grad = torch.autograd.grad(energy, predicted)[0]
    predicted = (predicted - REFINE_LR * grad).detach().requires_grad_(True)
predicted = predicted.detach()
# Then proceed to nearest-cosine lookup as before
```

**Prerequisites:** None — works against any trained engram model.

**Eval signal:** generation quality should improve on prompts where N-gram memory has useful patterns ("hi how are you"). Compare top-5 candidate cosine scores before vs. after refinement; refinement should sharpen the distribution toward semantically coherent options.

**Failure mode:** if the refined vector consistently picks the same top-1 as the unrefined vector, the energy function isn't doing useful work — try different energy formulations (consistency across multiple ponder loops, or distance to episodic memory) before discarding.

**Cost:** ~$0 (inference only). Implementation lift: small — a few hundred lines in the inference path.

---

## Candidate 2: Unbounded Recurrent Depth at Test Time

**Source:** "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (Geiping et al., NeurIPS 2025 / early 2026)

**Hypothesis:** Engram's adaptive pondering (innovation #3) is currently capped at 3 loops. Geiping et al. show that with appropriate training, a recurrent block can be unrolled to depths much greater than seen during training, allowing reasoning to scale with inference compute on a per-prompt basis.

**Engram axis:** Innovation #3, taken to its principled limit. Engram already has a `halt_gate` — this would let it actually reach the unbounded depths that PonderNet's original framing implies.

**Why this is candidate #2 not candidate #1:** can't just remove the cap on a v13 model and expect stable behavior at depth 20. The Geiping work requires *training* with specific properties (separate depth/width tokens, careful initialization, depth-aware loss) so the recurrent block stays stable when unrolled deeply. So this is a training-time change, not a pure inference enhancement.

**Single-variable spec:**

In `engram_model.py` and `ingest.py`:
- During training: randomly sample ponder steps in [1, 12] (instead of capped at 3) for each batch element
- Add depth-dependent loss reweighting (Geiping uses `1/sqrt(depth)`)
- At inference time: cap → ~50, with halt_gate confidence threshold to terminate

**Prerequisites:** Branch A first. If MAX_PONDER 3→5 with lower cost shows that pondering meaningfully engages, then this scaling makes sense. If Branch A reveals the halt gate is degenerate, the deeper version won't help.

**Eval signal:** Per-prompt avg_ponder_steps should vary across difficulty buckets (greetings ~1-2, harder ~5+). Math/logic-style prompts (currently degenerate) should show higher ponder counts AND better quality.

**Failure mode:** training instability at depth 12+ (which Geiping's tricks specifically address). If we hit it, fall back to depth 8 or implement just the random-depth-sampling without the unrolling-at-inference part.

**Cost:** ~$10-15 (training is more expensive due to variable depth). Implementation lift: medium — modifications to brain forward pass and training loop.

---

## Candidate 3: Predictive Coding Networks (PCN)

**Source:** "Towards scaling deep neural networks with predictive coding" (Sussex thesis, April 2026)

**Hypothesis:** Replaces standard backpropagation with local prediction-error minimization at each layer via iterative equilibration. Each layer has explicit error neurons; weight updates happen only after local errors stabilize. Theoretically approximates a trust-region second-order method, much more stable than scalar gradient multiplication.

**Engram axis:** Innovation #4 (signal-modulated learning), but the principled architectural version rather than v14-D's scalar approximation.

**Why this is candidate #3 not candidate #1:** biggest lift of the four. PCN isn't a swap — it's a full rewrite of the training procedure. Per-layer error nodes, iterative inference loop nested inside each batch step, careful initialization to avoid divergence. We should test the simpler scalar version (Branch D) first to confirm signal-modulation is even worth pursuing as an axis.

**Engineering reality:** there's no clean way to drop PCN into ingest.py as a single-variable swap. It would require:
- Adding error tensors at each layer of `AttentionBrain`
- Adding an inner inference loop (~10-20 iterations) inside each forward pass
- Replacing `loss.backward()` with PCN's local gradient computation
- Different optimizer per error node

**Single-variable spec:** N/A. Estimated 500-1000 lines of new code. Should be a `plans/V17_PCN_PLAN.md` of its own.

**Prerequisites:**
1. Branch D landed
2. Branch D showed surprise modulation helps but was unstable (the failure mode that PCN's stability claim addresses)
3. Cumulative spend allows ~$15-20 budget for first PCN run

**Eval signal:** loss curve should be smoother than xent + clip_grad_norm, with no spikes during training. Stability of per-layer error tensors over training should decrease monotonically.

**Failure mode:** equilibration fails to converge → divergent training. If hit, fall back to running PCN only on the last 2 layers, not the full stack.

**Cost:** ~$15-20 per training run (more iterations per batch). Implementation lift: large — multi-week rewrite of the brain training procedure.

---

## Candidate 4: Test-Time Training (TTT) for Episodic Memory

**Source:** "Reimagining LLM Memory: Using Context as Training Data Unlocks Models That Learn at Test-Time" (Sun, Choi et al., January 2026)

**Hypothesis:** Replaces engram's "retrieve from ChromaDB + blend via gate" with "compress current context into Layer 0 weights via fast-weight updates." Memory becomes literally encoded in network weights at inference time, not retrieved-and-blended.

**Engram axis:** Innovation #5 (persistence as architectural), more radical interpretation than the current implementation. Memory IS the layer.

**Why this is candidate #4 not candidate #1:** *replaces* an engram component rather than augmenting it. Higher risk because we'd lose the existing episodic ChromaDB infrastructure if it didn't work. And we've never actually validated the existing episodic mechanism in a training loop — Branch C is that test. If Branch C works, TTT is a refinement; if Branch C fails because the existing mechanism is too weak, TTT is the upgrade.

**Single-variable spec (after Branch C validates the mechanism):**

In `engram_model.py`:
- Designate one layer (probably Layer 0 since memory injection happens there) as the "memory layer"
- Add a fast-weight tensor of same shape as the layer's main weights
- During inference, perform K steps of gradient descent on the fast-weights against a self-supervised loss on the current context
- Add fast-weights to main weights for the actual forward pass
- Reset fast-weights at session boundary OR persist them per-user (engram's persistence claim makes this natural)

**Prerequisites:**
1. Branch C landed (confirms episodic mechanism works in training loop)
2. Successful ENGRAM_COHERENT path established or near-established (we want to add TTT to a working baseline, not use it as a Hail Mary)

**Eval signal:** within-conversation coherence should improve on multi-turn dialogues — model should "remember" prior turns within an interaction without explicit retrieval. Compare turn 1 vs. turn 5 of a conversation: turn 5 should reference turn 1 contextually if TTT works, generically if it doesn't.

**Failure mode:** catastrophic forgetting — fast-weight updates degrade the base model's capabilities. Mitigations: smaller learning rate, rank-1 fast-weight updates (LoRA-style), explicit reset on long pauses.

**Cost:** ~$10-15 per training run (no extra cost) + ~$0 inference cost beyond the K fast-weight steps. Implementation lift: medium-large — designing the fast-weight mechanism and self-supervised loss is non-trivial.

---

## Sequencing summary

| Candidate | When to consider | Prereq | Cost | Lift |
|-----------|-----------------|--------|------|------|
| ∇-Reasoner | After ANY v13/v14 model is trained — pure inference enhancement | None | $0 | Small |
| Recurrent depth | After Branch A confirms pondering lever works | Branch A | $10-15 | Medium |
| PCN | After Branch D shows scalar surprise helps but is unstable | Branch D | $15-20 | Large |
| TTT for episodic | After Branch C confirms episodic-in-training works | Branch C | $10-15 | Medium-large |

**Recommended exploration order if v14 branches all complete:**

1. **∇-Reasoner first** (zero training cost, can be tested against v13 immediately or any v14 result). If successful, deploy as inference enhancement to existing models.
2. **Recurrent depth next** (if Branch A succeeded — natural extension)
3. **TTT or PCN** after that, depending on which v14 branch revealed the most promising failure mode

**Cumulative budget projection if all four are pursued:**

| Stage | Cumulative spend | % of $150 ceiling |
|-------|------------------|-------------------|
| v9–v13 | ~51 | 34% |
| All v14 branches (A+B+C+D) | ~77 | 51% |
| ∇-Reasoner (v15) | ~77 | 51% (no training cost) |
| Recurrent depth (v16) | ~89 | 59% |
| PCN OR TTT (v17) | ~104-109 | 69-73% |
| All four | ~119 | 79% |

Even in the maximalist scenario, we stay under the $150 ceiling. But realistically only ~2 of these would get done before the loop converges or we hit a different bottleneck.

---

## Why these four and not others

These four were selected because each maps cleanly to one of engram's five distinguishing axes (vs. frontier transformers) AND has substantive 2025-2026 research backing. Other 2026 directions considered and deferred:

- **MoE for engram brain** — would require routing + expert loading; doesn't fit engram's "one small brain" thesis
- **State-space models replacing attention** — would replace the conventional transformer block, which is the *one* part of engram that's intentionally conventional (the novelty is around it, not in it)
- **Speculative decoding** — useful but doesn't exercise any engram-distinguishing architectural claim; pure inference speed optimization

These four are the ones that make engram *more* engram, not just faster or cheaper or bigger.
