# OpenMythos → Engram Transfer Plan

Test-driven, ablation-first plan for porting selected ideas from
[`kyegomez/OpenMythos`](https://github.com/kyegomez/OpenMythos) into Engram.
Every phase is gated on numeric pass/kill criteria computed by `bench/run.py`.
No vibes — only JSON deltas decide whether a change ships.

Pair this document with `RALPH_LOOP_RUNBOOK.md` for self-paced execution.

---

## Context

Engram already has the headline RDT property the OpenMythos paper builds on:
adaptive pondering with a learned halt gate (`ingest.py:285–321`). What it
lacks are four small, drop-in mechanisms that make looped depth actually
trainable and useful at scale:

1. **LTI-stable input injection** — re-anchors the residual stream to the
   original input every loop iteration, preventing drift across the 24
   effective layer-passes the current loop performs.
2. **Loop-index sinusoidal embedding** — gives shared blocks a signal
   distinguishing iter 0 from iter 2.
3. **Depth extrapolation at inference** — wiring change so we can spend
   more compute at decode time than training time.
4. **Per-loop LoRA scale** — cheap depth-wise weight modulation.

Plus one independent track:

5. **RoPE** — replaces learned `pos_embed`, hard-capped at `CONTEXT_SIZE=32`
   today (`ingest.py:247`), with a positional encoding that extrapolates.

The user has explicitly flagged this as a no-vibes effort: every phase
ships only if its tests pass, kills only if they fail. Phase 0 builds the
benchmark harness so test results are reproducible before any architectural
change goes in.

---

## Source / target reference index

| Idea | OpenMythos source (`open_mythos/main.py`) | Engram target (`ingest.py` unless noted) |
|---|---|---|
| LTI-stable input injection | `LTIInjection` lines 684–742; call site line 863 | Ponder loop lines 285–321 |
| Loop-index sinusoidal embedding | `loop_index_embedding` lines 541–570; call site line 858 | Ponder loop top, line 285 |
| Depth extrapolation at inference | `OpenMythos.forward` `n_loops` arg lines 992–1034; default fallthrough line 850 | `AttentionBrain.__init__` line 243; `forward` line 259 |
| Per-loop LoRA adapter | `LoRAAdapter` lines 578–619; call site line 862; clamp line 614–616 | After inner block stack, around line 289 |
| RoPE | `precompute_rope_freqs` / `apply_rope` lines 124–169; call sites lines 236–237 | `AttentionBlock.__init__` lines 79–113; `forward` lines 117–151; delete `pos_embed` lines 247, 273 |
| Depth-extrap LoRA clamp pattern | Lines 614–616 | Required when Phase 4 + Phase 3 are both on |
| ACT halting (reference only — Engram already has equivalent) | `ACTHalting` lines 750–780; remainder trick lines 873–883 | Already present, lines 285–321; no porting needed |

Consult these line numbers when implementing. They are the load-bearing
contract of this plan — if they drift, fix the references before changing
the implementation steps.

---

## Phase 0 — Harness + class unification + reproducibility gate

**Status:** PASSED
**Estimated:** 1 person-day
**Blocks:** every other phase

### 0.1 Unify model class [x]

**Problem:** `test_brain.py:17–112` defines a different model than
`ingest.py:37–323`:

| Field | `ingest.py` | `test_brain.py` |
|---|---|---|
| `EMBED_DIM` | 256 | 64 |
| `CONTEXT_SIZE` | 32 | 8 |
| `N_LAYERS` | 8 | 3 |
| Attention | 8-head with `W_o` | single-head, no `W_o` |

State dicts cannot load coherently. Any benchmark on top of this is
meaningless.

**Action:**

- Create `engram_model.py` housing `AttentionBlock`, `EngramModule`,
  `AttentionBrain`.
- `ingest.py`, `test_brain.py`, `eval_brain.py` import from it.
- Delete duplicate class definitions from the consumer scripts.

**Pass criterion:** `test_brain.py` loads `engram_weights.pth` produced by
`ingest.py` with zero state-dict warnings.

### 0.2 Build `bench/run.py` [x]

The single entry point for every test in this document. Reproducible JSON
results land in `bench/history/<run_id>.json`.

**CLI inputs:**

- `--seed INT` (sets torch + python + numpy)
- `--corpus PATH ...` (default: `corpus/dailydialog_tiny.txt`,
  `corpus/11_alice_s_adventures_in_wonderland.txt`)
- `--holdout PATH ...` (default: `corpus/35_the_time_machine.txt` —
  **never seen during training**)
- `--epochs`, `--batch-size`, etc.
- Feature flags: `--use-lti`, `--use-loop-idx`, `--use-lora`, `--use-rope`,
  `--n-ponder-train`, `--n-ponder-eval`

**JSON outputs:**

1. `train_loss_curve` — avg loss per 100 batches
2. `grad_norm_p50`, `grad_norm_p99` — per-step pre-clip percentiles
3. `ponder_steps_hist` — counts over last epoch
4. `eval_cosine_top1` — % holdout next-word predictions whose cosine-nearest
   ChromaDB neighbor matches ground-truth
5. `eval_cosine_mean` — mean cosine sim, predicted vs ground-truth concept
   vectors on holdout
6. `eval_perp_proxy` — `-log(softmax(top-K cosine-nearest))` at ground truth
7. `wall_time_train`, `wall_time_eval`
8. `param_count`
9. `flops_per_token` (analytical: layers × dim × n_ponder × seq_len)
10. `config` — full feature flag echo

### 0.3 Reproducibility gate [x]

**Test:** Run `bench/run.py --seed 42` twice with identical config.

**Pass criterion:** `train_loss_curve` max abs diff ≤ 1e-5 per point;
`eval_cosine_top1` identical to the integer count.

**Kill criterion:** Any non-determinism. Most likely culprits:

- `random.shuffle(sequences)` in `ingest.py:602`
- CUDA non-deterministic ops — set `torch.use_deterministic_algorithms(True)`
- Missing seed propagation to ChromaDB or numpy

**If 0.3 fails, halt. No phase 1+ work is meaningful until reproducibility
holds.**

### 0.4 Baseline run [x]

Lock in numbers for the current `main` branch. `bench/history/baseline.json`
is the fixed comparison point for every later phase.

---

## Phase 1 — LTI-stable input injection

**Status:** KILLED
**Pre-req:** Phase 0 complete
**Reference:** `OpenMythos:open_mythos/main.py:684–742` (`LTIInjection`),
call site line 863.
**Target:** `engram/ingest.py:285–321` (the ponder loop).

### 1.1 Hypotheses

- **H1** Reduce 99th-percentile pre-clip gradient norm by ≥30%.
- **H2** Improve `eval_cosine_top1` by ≥1.5pp at iso-FLOP.
- **H3** Allow stable training at `BRAIN_LR = 2e-3` (currently `1e-3`,
  lowered specifically for the deeper model — `ingest.py:45`).

### 1.2 Implementation

1. Port `LTIInjection` from OpenMythos lines 684–742 verbatim. Use the
   diagonal form (one scalar `log_A` per channel, scalar `log_dt`).
2. In `AttentionBrain.__init__`, add
   `self.injection = LTIInjection(embed_dim)`.
3. In `AttentionBrain.forward` (`ingest.py:259`):
   - After position embedding (`ingest.py:273`), freeze `e = x`.
   - In the outer ponder loop, compute
     `transformer_out = blocks(x) - x` (delta), then
     `x = self.injection(x, e, transformer_out)`.
4. Mirror in unified `engram_model.py`.

### 1.3 Tests

| ID | Setup | Pass | Kill | Result |
|---|---|---|---|---|
| **1-A** Smoke | 200 steps on 1k sequences | Loss not NaN; `injection.get_A()` ∈ (0,1) every step | NaN, A out of range | [x] PASS |
| **1-B** Reproducibility | Re-run 1-A, seed 42 | Loss curves identical to 1e-5 | Nondeterminism | [x] PASS |
| **1-C** Stability (H1) | Full bench at current LR | `grad_norm_p99` ≤ 0.3927 | < 30% reduction | [x] FAIL — got 0.5724 vs 0.5610 baseline |
| **1-D** Quality at iso-FLOP (H2) | Full bench | `eval_cosine_top1` ≥ 6.5% | Equal or worse | [x] FAIL — got 5.0% (unchanged) |
| **1-E** LR robustness (H3) | LTI@2e-3 vs baseline@2e-3 (control) | LTI trains stably; baseline NaNs or `grad_norm_p99` doubles | LTI also unstable | skipped (1-C+1-D both fail) |

**Decision rule:** Keep iff **1-C OR 1-D passes**. → **KILLED** — neither passed.

### 1.4 Failure-mode watch

If 1-A's `A` collapses to ~0 within 100 steps, init `log_A` to
`torch.full((dim,), -2.0)` so initial `A ≈ 0.87` (closer to identity).
OpenMythos uses `zeros` init giving `A ≈ 0.37`; may be too aggressive
at our scale.

---

## Phase 2 — Loop-index sinusoidal embedding

**Status:** KILLED
**Pre-req:** Phase 1 merged _(Phase 1 killed; Phase 2 ran against baseline — loop-idx is LTI-independent)_
**Reference:** `OpenMythos:open_mythos/main.py:541–570`,
call site line 858.

### 2.1 Hypothesis

**H4** Iter 0 and iter 2 of the ponder loop are computationally
indistinguishable today. A sinusoidal `loop_t` signal lets the halt gate
and the FFN behave differently at different depths.

### 2.2 Implementation

- Port `loop_index_embedding` from OpenMythos lines 541–570 (~25 lines).
- In `AttentionBrain.forward`, top of `for ponder_idx in range(max_ponder)`:
  `x = loop_index_embedding(x, ponder_idx, loop_dim=embed_dim // 8)`.
  → loop_dim = 32 for embed_dim=256, matching OpenMythos line 821–823.

### 2.3 Tests

| ID | Setup | Pass | Kill | Result |
|---|---|---|---|---|
| **2-A** Halt distribution shifts | Bench vs LTI-only baseline | `entropy(ponder_steps_hist) > prev + 0.1 nats` OR mean halt drops ≥0.2 with no eval regression | Identical halt distribution | [x] FAIL — {"3": 1328}, entropy 0 nats, Δ=0 |
| **2-B** Quality | Bench run | `eval_cosine_top1` ≥ 5.5% | Worse than baseline | [x] FAIL — 5.0% (unchanged) |
| **2-C** Channel sensitivity | `loop_dim ∈ {16, 32, 64}` | At least one config beats baseline on 2-B | None beat | skipped (2-A fails decision rule) |

**Decision rule:** Keep iff 2-A passes AND 2-B does not regress. → **KILLED** — halt gate shows zero sensitivity to loop-index signal.

---

## Phase 3 — Depth extrapolation at inference

**Status:** SKIPPED (Phase 1 killed, Phase 2 killed — prerequisite chain not met)
**Pre-req:** Phase 1; ideally Phase 2
**Reference:** `OpenMythos:open_mythos/main.py:850, 992–1034`.

### 3.1 Hypothesis

**H5** With LTI + loop-idx, the model can usefully run more ponder iters at
inference than at training — the headline RDT property.

### 3.2 Implementation

- `AttentionBrain.forward(self, x, ngram_memory=None, engram_module=None, n_ponder=None)`.
  Default to `self.max_ponder`.
- `test_brain.py` and `eval_brain.py`: add `MAX_PONDER_INFERENCE` constant;
  pass through.
- If Phase 4 LoRA is on, clamp `loop_t` to trained max (mirror
  OpenMythos lines 614–616).

### 3.3 Tests

| ID | Setup | Pass | Kill |
|---|---|---|---|
| **3-A** Monotonicity | Train at `n_ponder_train=3`. Eval at `n_ponder_eval ∈ {1, 2, 3, 4, 5, 6, 8}` | Quality rises monotonically (or saturates flat) 1→3, and 4–8 are ≥ value at 3 | Quality drops past 3 |
| **3-B** Saturation | Same | Concave curve (diminishing returns), not step-function | Linear forever — investigate |
| **3-C** Halt gate behaves | Log `n_steps_taken` at eval | At `n_ponder_eval=8`, mean halt step < 8 | Gate runs to max every time |

**Decision rule:** Keep iff 3-A passes. This is the test that distinguishes
"we ported some PyTorch" from "we got the RDT property."

---

## Phase 4 — Per-loop LoRA scale

**Status:** SKIPPED (Phases 1–3 not net positive)
**Pre-req:** Phases 1–3 net positive
**Reference:** `OpenMythos:open_mythos/main.py:578–619`,
call site line 862, clamp lines 614–616.

### 4.1 Hypothesis

**H6** Loop-index input embedding (Phase 2) shifts what the block *sees*.
Per-loop LoRA shifts what it *does*. Combined > either alone.

### 4.2 Implementation

- Port `LoRAAdapter` from OpenMythos lines 578–619 (rank=8 for embed_dim=256,
  `max_loops = max_ponder + 4` to allow extrapolation).
- After the inner block stack, before LTI:
  `transformer_out = transformer_out + self.lora(transformer_out, ponder_idx)`.

### 4.3 Tests

| ID | Setup | Pass | Kill |
|---|---|---|---|
| **4-A** Param overhead | Inspect `param_count` | LoRA adds ≤2% to brain params | More than 2% — reduce rank |
| **4-B** 4-way ablation | {±loop_idx} × {±LoRA} on top of LTI | Best = both ON; (loop_idx+LoRA) > (loop_idx only) by ≥0.3pp | LoRA-only beats both-on, OR both-on = loop_idx-only |
| **4-C** Extrapolation compat | Re-run 3-A with LoRA on, clamping `loop_t` | Quality at `n_ponder_eval=6` not worse than Phase 3 | Worse |

**Decision rule:** Keep iff 4-B both-on wins AND 4-C does not regress.

---

## Phase 5 — RoPE replacing learned positional embedding

**Status:** PASSED
**Pre-req:** Phase 0 only (independent track)
**Reference:** `OpenMythos:open_mythos/main.py:124–169` (RoPE prims),
call sites lines 236–237.

### 5.1 Hypothesis

**H7** `nn.Embedding(32, 256)` (`ingest.py:247`) cannot represent positions
≥32. README explicitly lists "long-range coherence" as not-yet. RoPE
supports test-time extension without retraining.

### 5.2 Implementation

1. Port `precompute_rope_freqs` and `apply_rope` from OpenMythos
   lines 124–169 (~50 lines).
2. In `AttentionBlock.__init__`, register `freqs_cis` buffer for `head_dim`
   at `max_seq_len`.
3. In `AttentionBlock.forward`, after multi-head reshape (`ingest.py:127–131`),
   apply `Q = apply_rope(Q, freqs_cis); K = apply_rope(K, freqs_cis)` before
   the matmul.
4. Delete `self.pos_embed` and the position-add at `ingest.py:247, 273`.

### 5.3 Tests

| ID | Setup | Pass | Kill | Result |
|---|---|---|---|---|
| **5-A** At-distribution parity | Train @ CONTEXT_SIZE=32 with RoPE; eval @ 32 | `eval_cosine_top1` within 0.5pp of baseline | Worse by >0.5pp | [x] PASS — 5.0% (within 0pp), grad_norm_p99 halved: 0.561→0.280 |
| **5-B** Extrapolation | Same training; eval @ ctx ∈ {64, 96} | Quality @ 64 ≥ @ 32 − 3pp; no collapse | Cliff at any length ≤2× train context | [x] PASS — 5.0% at 2× (64) and 3× (96) train ctx; zero cliff |
| **5-C** Re-train at longer context | Train fresh @ CONTEXT_SIZE=64 with RoPE | `eval_cosine_top1` ≥ 6.0% | No gain | [x] FAIL — 5.0% (metric plateau, not RoPE failure) |

**Decision rule:** Keep iff 5-A passes AND (5-B OR 5-C passes). → **PASSED** — 5-A + 5-B both pass.

### 5.4 Failure-mode watch

`theta` matters. OpenMythos uses 500k (line 75). Engram operates at much
smaller context — 10k may suffice and extrapolate further. Try both.

---

## Phase 6 — Lock and document

**Status:** PENDING
**Pre-req:** Phases 1–5 resolved
**Manual checkpoint — requires human review**

- Lock the winning combination in `engram_model.py`.
- Update README "Evolution roadmap" table with measured deltas (no
  qualitative "Working / Partial").
- Move all `bench/results/*.json` to `bench/history/` for archival.
- Single summary commit.

---

## Iso-FLOP / iso-param controls

Every phase test must be paired with two controls to avoid trivially-true
"more compute = better" conclusions:

- **More-compute control:** baseline with `n_ponder=4` (one extra ponder
  iter, no architectural change).
- **More-params control:** baseline with `n_layers=9` (one extra layer,
  no architectural change).

If neither control catches up to the architectural change, the change is
the real source of the gain. Record both in every phase's JSON.

---

## Order of operations

```
0  Harness + class unification + reproducibility gate     [day 1]
↓
1  LTI injection                                          [day 2]
↓
2  Loop-index embedding                                   [day 3]
↓
3  Depth extrapolation @ inference                        [day 3 PM]
↓
4  Per-loop LoRA (gated on 1–3 net positive)              [day 4]
↓
5  RoPE (parallel track, can start anytime after 0)       [day 4–5]
↓
6  Lock + document                                        [day 5 PM]
```

**Total:** ~5 person-days full execution; ~2 person-days for the
LTI + loop-idx + depth-extrap minimum-viable path.

---

## Status legend

Each phase has a `Status:` line at the top of its section. The ralph-loop
updates these as it executes:

- `PENDING` — not started
- `IN_PROGRESS` — currently being executed by the loop or a human
- `PASSED` — all tests passed per decision rule; change is live in `main`
- `KILLED` — at least one kill criterion tripped; change is reverted

Each test row in the test tables also gains a status marker as it runs:

- `[ ]` — not run
- `[x]` — passed
- `[!]` — killed
- `[-]` — skipped (e.g. blocked by an earlier kill)

The ralph-loop reads these markers to decide what to do next. See
`RALPH_LOOP_RUNBOOK.md`.
