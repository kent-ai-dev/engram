# v18 Plan — Stalk-Vocab: Gaussian-Sheaf Vocabulary as a Substrate-Level Test

**Status:** drafted 2026-05-05.
**Decide:** after the v15-A pondering result has been read out and a clean v15+v16 baseline exists; this is not single-variable engineering, it is a substrate-level swap and should not be entangled with an in-flight branch.
**Principle:** every prior plan (v9–v17) holds the *substrate* fixed and tunes one knob. This plan changes the substrate of the vocabulary itself. It is the minimum viable test of the claim made in `plans/FUTURE_RESEARCH.md` self-review: that engram's deepest remaining transformer-inheritance is *meaning-as-a-point*, and that replacing it with *meaning-as-a-stalk-of-a-Gaussian-sheaf* is the first step of a real architectural innovation rather than another textbook ML knob.

## What this plan is and is not

**Is:** a falsifiable, single-GPU, ~$15 experiment that swaps ChromaDB's 96D unit-vector vocab for a 96D Gaussian-stalk vocab — each word stores `(μ, Λ)` instead of `μ` alone. Nearest-cosine lookup is replaced by Wasserstein-2 nearest-stalk. Cross-entropy loss is augmented (not replaced) by a sheaf-Laplacian penalty over the training context window. Everything else — `AttentionBrain`, `EngramModule`, episodic memory, surprise-gated learning, adaptive pondering — is held identical to the v15/v16 baseline. The change is confined to the vocab interface.

**Is not:** a full Stalk-Engram. The destination architecture (sheaf cohomology regularization, renormalization-group hierarchy over stalks, variational EM in place of backprop) is a multi-quarter rewrite. This plan is the *first earned step* — the smallest experiment that tests whether the gluing-axiom inductive bias is load-bearing for compositional generalization at engram's scale.

## Hypothesis

> Engram's vocab currently treats each word as a single point on the unit sphere in R^96. Polysemy, deixis, and context-dependent reference are therefore handled associatively (the brain learns by example which neighborhoods of `predicted` map to which sense). If we instead store each word as a Gaussian `N(μ_w, Λ_w⁻¹)` and predict a Gaussian `N(μ̂, Λ̂)`, with lookup performed by Wasserstein-2 distance and a sheaf-Laplacian penalty enforcing local consistency between adjacent context positions, the model should reach better out-of-distribution compositional generalization at the same parameter count. The mechanism is *topological compression*: requiring the restriction maps between adjacent stalks to glue consistently is an inductive bias that forecloses most of the hypothesis class outright, in the same way equivariant networks are regularized by their symmetry.

If the hypothesis is wrong, the sheaf-Laplacian collapses to triviality (every restriction map → identity, every Λ → isotropic) and the model degenerates to a slow Gaussian-smoothed cosine-vocab transformer with no OOD lift. That degeneration is observable from the precision statistics during training and is the cheapest abandonment signal.

## Engram-axis engaged

**Innovation #1 — vocab/brain separation.** This is the *strongest* engram axis (the only one that is structurally distinct from a transformer rather than just additive). v14-B already proved that vocab geometry is a bottleneck. Stalk-vocab attacks the deeper question: *what should the vocab be, geometrically?* Not unit vectors. Distributions on a fiber of a learned bundle.

## Math

### Stalk

Each vocabulary word `w` is a stalk of the vocabulary sheaf:

    F_w = N(μ_w, Λ_w⁻¹)        μ_w ∈ R^96,  Λ_w ∈ S++^96

In practice we parameterize Λ_w by its Cholesky factor `L_w ∈ R^{96×96}` (lower-triangular with positive diagonal) so that `Λ_w = L_w L_w^T` is automatically positive definite. To keep parameter count manageable, the **default parameterization is diagonal**: `L_w = diag(σ_w)` with `σ_w ∈ R^96_+`. Diagonal precision adds 96 parameters per word — a 2× total vocab cost, not 97×. Full-Cholesky stalks are an ablation toggle (`STALK_PRECISION_RANK`), not the default.

### Brain output

`AttentionBrain` gains a precision head alongside its existing concept-vector head:

    μ̂  = W_μ · h_T          (existing)
    log σ̂ = W_σ · h_T         (new — diagonal log-precision)
    Λ̂ = diag(exp(2 · log σ̂))

`W_σ` is a single linear layer of shape `(EMBED_DIM, EMBED_DIM)` initialized to small values so that `Λ̂ ≈ I` at the start (i.e. the model begins as a near-isotropic Gaussian, recovering cosine-vocab behavior as a degenerate case).

### Wasserstein-2 lookup

Replace `nearest_words` cosine search with Wasserstein-2 between the predicted stalk `(μ̂, Λ̂)` and each vocab stalk `(μ_w, Λ_w)`. For diagonal Gaussians the W₂ distance has a closed form:

    W₂²((μ̂, σ̂), (μ_w, σ_w)) = ‖μ̂ − μ_w‖² + ‖σ̂ − σ_w‖²

That is just Euclidean distance on the stacked `(μ, σ)` vector — implementable as a single matrix multiply against a `(V, 2·EMBED_DIM)` stacked vocab matrix. **Lookup cost is asymptotically identical to current cosine lookup** (one matmul, one top-k). This matters: the falsification test needs the substrate change to be the *only* moving part, not entangled with a 10× slowdown.

### Sheaf-Laplacian penalty

For each adjacent pair of training-context positions `(t, t+1)`, learn a restriction morphism `ρ ∈ R^{96×96}` (one global ρ, not per-position — weight-shared so it captures "the typical local update operator" not a per-position parameter explosion). The sheaf-Laplacian penalty is

    L_sheaf = (1 / (T−1)) · Σ_{t=1..T−1}  ‖ρ · μ_{x_t} − μ_{x_{t+1}}‖²_{H(σ_t, σ_{t+1})}

where `H(·,·)` is the harmonic mean of the diagonal precisions (so that high-confidence stalks dominate the disagreement). This is the **engram-scale, weight-shared, single-edge-type version** of the cellular sheaf Laplacian from Hansen & Ghrist (2019, *Toward a Spectral Theory of Cellular Sheaves*) and Bodnar et al. (2022, *Neural Sheaf Diffusion*). The full sheaf has multiple edge types and per-edge restriction maps; we are deliberately collapsing to one global ρ to keep the experiment minimal. Per-edge ρ is a follow-up if the global version shows signal.

### Total loss

    L = L_xent + λ_sheaf · L_sheaf

with `λ_sheaf` swept over `{0, 1e-3, 1e-2, 1e-1}` in the experiment plan below. λ=0 is a control: it tells us whether *just* the precision head and W₂ lookup do anything without the gluing axiom acting as a regularizer.

## File-by-file diffs

### `engram_model.py`

- Add `STALK_PRECISION_RANK` constant (default `"diagonal"`, alt `"full_cholesky"`). Diagonal is the default and the only one this plan validates.
- Add `self.W_sigma = nn.Linear(EMBED_DIM, EMBED_DIM)` to `AttentionBrain.__init__`, initialized so `log σ̂ ≈ 0`.
- In `AttentionBrain.forward`, after the existing concept-vector projection at line 223+, compute and return `log_sigma` alongside `predicted`. Return signature changes from `(predicted, n_steps)` to `(predicted, log_sigma, n_steps)`.
- Add a global learnable parameter `self.rho = nn.Parameter(torch.eye(EMBED_DIM))` for the sheaf restriction map. (One per model, not per-layer.)
- Estimated change: ~40 lines.

### `ingest.py`

- Allocate a learnable `vocab_log_sigma_global` parallel to the existing `vocab_matrix_global` (currently lines 382–385). Same shape `(V, EMBED_DIM)`. Initialize to zeros (so all stalks start as `N(μ, I)`, recovering current behavior at λ_sheaf=0).
- Add the param group to optimizer at the same LR as `vocab_matrix_global` (current `EMBED_LR * 0.5` at line 385).
- In the training step (around line 478, where `vocab_matrix_normed` is built): build `vocab_stalks_stacked = torch.cat([vocab_matrix_global, vocab_log_sigma_global], dim=-1)` and use it for W₂ lookup logits instead of cosine. Cross-entropy is then over W₂ distance: `logits = −((predicted_stalk - vocab_stalks_stacked).pow(2).sum(-1)) * INV_TEMPERATURE`.
- Add the sheaf-Laplacian penalty (one-line einsum + harmonic mean) and add `λ_sheaf · L_sheaf` to the loss before `loss.backward()`.
- Sync `vocab_log_sigma_global` to ChromaDB at end of training, alongside the existing μ sync at line 564+. Use a *second collection* `engram_vocab_logsigma` rather than trying to stuff Λ into the existing collection's metadata.
- Estimated change: ~80 lines.

### `eval_chat.py`

- Update `nearest_words` (line 64) to take a stacked `vocab_stalks_stacked` and a stacked `pred_stalk = cat([μ̂, log σ̂])`, compute `dists = ((pred_stalk - vocab_stalks_stacked).pow(2).sum(-1))` and `topk(-dists, k)`. Penalty handling stays identical.
- Update `generate_reply` (line 77+) to unpack the new 3-tuple from `brain(...)` and assemble `pred_stalk`.
- Load `engram_vocab_logsigma` collection alongside `engram_vocab` at line 158+; build `vocab_stalks_matrix` once before the eval loop.
- Estimated change: ~40 lines.

### `test_brain.py`

- Same surface changes as `eval_chat.py` (3-tuple unpack, stacked stalk lookup, second collection load). Estimated ~30 lines.

### `server.py`

- Same surface changes for the runtime serving path. Estimated ~20 lines.

**Total implementation lift:** ~210 lines of new/changed code, no deletions, no architectural rewrite. The change is large enough to be substantive, small enough to be reverted.

## Constructed polysemy benchmark — `eval_polysemy.py`

The standard chitchat eval cannot detect the killer capability this plan claims. We need a benchmark whose pass condition is *out-of-distribution polysemy resolution*. Construction:

- Pick 30 homonyms with two well-separated senses (`bank`, `bat`, `bark`, `pitcher`, `seal`, `mole`, `match`, …). Source: WordNet senses, filtered for senses that have non-overlapping co-occurrence vocabularies in the training corpus.
- For each homonym, write 4 disambiguating context templates per sense (`"the river ___ was muddy"`, `"i deposited cash at the ___"`, etc.) — 240 prompts total.
- **Train/test split:** during training, expose the model to 3 of the 4 templates per sense. At eval, test on the held-out 4th template — the same homonym, the same sense, but a context combination unseen at training time.
- **Metric:** for each held-out prompt, the model predicts the next word and we score `argmax_w W₂((μ̂, σ̂), (μ_w, σ_w))` against the ground-truth completion. Aggregate: `OOD-polysemy-accuracy` = fraction of held-out prompts where ground truth is in top-3.

This benchmark must be built and committed *before* training the v18 run, not after. Cost: ~2 hours of corpus construction + dataset commit.

## Experiment plan

| Run | INV_TEMP | λ_sheaf | Vocab | Brain | Purpose |
|-----|----------|---------|-------|-------|---------|
| v18-control | 30 | 0 | cosine (current) | current | Re-confirm v15/v16 baseline on the polysemy benchmark — establishes the baseline OOD-polysemy-accuracy number this plan must beat. |
| v18-A | 30 | 0 | stalk (diag Λ) | + W_σ head | Isolate the precision head alone. Does giving the model a Λ output help even without a sheaf penalty? |
| v18-B | 30 | 1e-2 | stalk | + W_σ head + ρ | The actual hypothesis test. Sheaf-Laplacian penalty active. |
| v18-C | 30 | 1e-1 | stalk | + W_σ head + ρ | Stronger penalty. If v18-B works, does more sheaf help? If v18-B is degenerate, does forcing the penalty harder break it differently? |

Each run uses the same corpus, same epochs, same all-else as v15-A, on Modal A10G. Estimated per-run cost: ~$3 (matches v15-A). Total v18 budget: ~$15.

## Metric and decision rule, defined in advance

**Primary metric:** OOD-polysemy-accuracy on the held-out template-4 split, measured at the end of epoch 5.

**Pass condition (proceed to v19 — per-edge ρ, then RG hierarchy):**
- v18-B beats v18-control by **≥3 absolute points** of OOD-polysemy-accuracy, AND
- v18-B does not regress chitchat eval coherence vs. v15-A baseline by more than 1 point on the existing 5-prompt rubric, AND
- training-time precision statistics show **non-degenerate** Λ — the harmonic-mean-precision distribution at end of training is *not* concentrated at the isotropic-identity point. (If it is, the precision head learned to ignore itself.)

**Soft-fail (interpret carefully):**
- v18-A beats v18-control but v18-B does not improve on v18-A → the precision head helps but the sheaf penalty does not. Conclusion: meaning-as-distribution helps, but the gluing axiom does not. Drop the sheaf machinery, keep stalk vocab, move on.

**Hard-fail (abandon Stalk-Engram direction):**
- v18-B does not beat v18-control by ≥3 points, AND
- ρ converges to near-identity (sheaf-Laplacian collapsed to triviality), AND
- chitchat eval is at parity or worse.

In the hard-fail case, the inductive bias is not load-bearing at engram's scale. Document the negative result, archive the branch, and return to the v14-style single-knob loop. **This is the explicit precommitment to falsifiability that `FUTURE_RESEARCH.md`'s ∇-Reasoner / TTT proposals lack.**

## Risks and mitigations

**Risk 1 — Diagonal precision is too weak.** Diagonal Λ cannot represent correlated uncertainty across concept dimensions. *Mitigation:* if v18-A,B,C all degenerate but the precision statistics are non-trivial, run v18-D with `STALK_PRECISION_RANK="full_cholesky"` (adds ~96 params/word, ~14M extra vocab params at full vocab — still cheap). Only take this step if diagonal showed *any* signal.

**Risk 2 — Sheaf-Laplacian regularizer collapses to triviality.** The easiest way to satisfy `‖ρμ_t − μ_{t+1}‖ ≈ 0` is `ρ → I` and `μ_t → μ_{t+1}`. *Mitigation:* monitor `‖ρ − I‖_F` during training; if it goes to zero, the penalty isn't doing useful work. Counter-measure if observed: add a unit-norm constraint on `ρ` (force it onto the orthogonal group via Cayley parameterization). Defer this complication until evidence demands it.

**Risk 3 — Gaussian fibers cannot represent multimodal meaning.** "Bank" has two modes, not one Gaussian. *Mitigation:* none in v18 — accept this as a known weakness and a v19 follow-up (mixture-of-Gaussians stalks). The v18 hypothesis is only that *unimodal Gaussian stalks beat unit-vector points*. If even that is wrong, mixture stalks aren't worth attempting.

**Risk 4 — W₂ lookup becomes unstable when Λ̂ predicts very large precision.** Numerical issue, not a conceptual one. *Mitigation:* clamp `log σ̂` to `[-3, +3]` (precision ratio ≤ e⁶ ≈ 400× isotropic) at the lookup site. Standard practice.

**Risk 5 — The polysemy benchmark is too easy / too hard.** A benchmark we built is a benchmark we can game. *Mitigation:* commit the benchmark *before* running v18-A onward. Hold v18-control as the calibration: if v18-control gets >70% on it, the benchmark is too easy; if <20%, too hard. Tune template difficulty against v18-control alone, not against the experimental runs.

## Why this is the right next experiment, not another `FUTURE_RESEARCH.md` candidate

The `FUTURE_RESEARCH.md` candidates (∇-Reasoner, recurrent depth, PCN, TTT) are all defensible engineering moves, but each is bolting another well-known component onto engram's existing transformer-shaped substrate. None of them attacks a structural assumption engram itself inherits from transformers. The self-review prompt that motivated this plan made the case explicit: *meaning-as-a-point* is a load-bearing assumption that current LLMs share with engram, and the gluing axiom is the cheapest mathematically-grounded inductive bias that relaxes it. Stalk-vocab is the smallest experiment that tests whether that relaxation pays.

If v18 hard-fails, we will know — quickly and cheaply — that the substrate-level direction is not productive at engram's scale, and we can return to the single-knob loop with that question settled. If v18 passes, we have earned the right to attempt v19 (per-edge restriction maps) and v20 (renormalization-group hierarchy over stalks). Either outcome is informative. That is the mark of an experiment worth running.

## Prerequisites

1. v15-A pondering result has been read out and the v15/v16 baseline is clean.
2. `eval_polysemy.py` and the constructed homonym benchmark are committed *before* training v18-A.
3. v18-control has been run on the new benchmark and produces a baseline number that's neither saturated nor at floor.

## Cost

- v18-control: ~$3
- v18-A, v18-B, v18-C: ~$3 each → ~$9
- Optional v18-D (full-Cholesky): ~$3 if invoked
- **Total: $12–$15**, leaving substantial headroom under the $150 ceiling and well below the cumulative-budget projection in `FUTURE_RESEARCH.md`.

## Sequence into the broader Stalk-Engram vision

If v18 passes, the path forward is:

- **v19** — per-edge restriction maps `ρ_e` (one per edge type, where edge types are learned by clustering position-pair contexts) instead of one global ρ.
- **v20** — multi-scale stalks: a coarse-graining transformation maps adjacent-token stalks into clause-level stalks, with weight-tied scaling. This is the renormalization-group step.
- **v21+** — replace cross-entropy with variational free energy on the joint sheaf, using natural-gradient updates on the Fisher manifold of sheaves of Gaussians. This is the substrate replacement of backprop, and it should not be attempted until v18–v20 have each individually shown signal.

v18 is the only step in this sequence that fits in a $15, single-GPU, single-week experiment. Everything beyond it is contingent on v18 producing a positive answer to a sharp question.
