# Ralph-Loop Runbook for `OPENMYTHOS_TRANSFER.md`

How to drive the OpenMythos transfer plan to completion using the
[ralph-loop](https://claude.com/plugins/ralph-loop) skill.

---

## What ralph-loop does

The skill re-feeds the **same prompt** back to the model on every iteration.
This means: the prompt cannot rely on conversation memory. All state must
live in the repo. For this plan, state lives in two places:

1. `plans/OPENMYTHOS_TRANSFER.md` — phase `Status:` lines and per-test
   `[ ]` / `[x]` / `[!]` / `[-]` markers.
2. `bench/history/*.json` — JSON results from every run of `bench/run.py`.

The loop only stops when:

- Output emits the exact `--completion-promise` string, **or**
- The user runs `/cancel-ralph`, **or**
- `--max-iterations` is reached.

> "If a completion promise is set, you may ONLY output it when the
> statement is completely and unequivocally TRUE."
> — ralph-loop skill rule

---

## Slash commands

| Command | Purpose |
|---|---|
| `/ralph-loop "PROMPT" [flags]` | Start a loop |
| `/cancel-ralph` | Stop the active loop (removes `.claude/ralph-loop.local.md`) |
| `/ralph-help` | Show usage |

**Flags:**

- `--max-iterations N` — hard cap on iterations
- `--completion-promise TEXT` — exact string the model must emit to halt
  cleanly

---

## The exact launch command

Copy-paste this into a fresh Claude Code session inside the engram repo:

```
/ralph-loop "Read plans/OPENMYTHOS_TRANSFER.md and ls bench/history/. Find the lowest-numbered phase whose Status is not PASSED or KILLED. Within that phase, find the next test with status [ ]. Execute that test: implement any required code changes per the phase's implementation section, run bench/run.py with the appropriate feature flags, write the result JSON to bench/history/<phase>-<test>-<timestamp>.json, then update the test's status marker in OPENMYTHOS_TRANSFER.md to [x] (passed), [!] (killed per kill criterion), or [-] (skipped because a prior kill made it moot). After the last test in a phase, evaluate the decision rule: if the rule passes, set the phase Status to PASSED and commit the code changes; if it fails, revert the code changes (git checkout -- .) and set the phase Status to KILLED. Append a one-line entry to plans/EXECUTION_LOG.md describing what just happened. If every phase has Status PASSED or KILLED, emit ENGRAM_OPENMYTHOS_TRANSFER_COMPLETE and stop. Otherwise iterate." --max-iterations 30 --completion-promise "ENGRAM_OPENMYTHOS_TRANSFER_COMPLETE"
```

Why this prompt:

- **Reads state from the repo, not memory.** `OPENMYTHOS_TRANSFER.md` and
  `bench/history/` are the source of truth.
- **Picks the next concrete unit of work** (lowest-numbered pending test,
  not "what should I do next").
- **Updates state in the same file the next iteration will read,** so the
  loop genuinely progresses.
- **Has a clean terminal state** — every phase ends in PASSED or KILLED;
  when that's true for all phases, the completion promise fires.
- **Logs to `EXECUTION_LOG.md`** so a human can audit what happened
  without re-reading every commit.

---

## Why `--completion-promise` matters here

The plan has a defined terminal state (Phase 6 lock + archive). Without a
completion promise, the loop keeps re-feeding the prompt forever, even
after every phase is resolved. With it, the loop only stops when the
model can truthfully output `ENGRAM_OPENMYTHOS_TRANSFER_COMPLETE` —
which per the prompt above is gated on every phase being PASSED or
KILLED.

The string `ENGRAM_OPENMYTHOS_TRANSFER_COMPLETE` was chosen to be
unique to this plan; verify with:

```sh
grep -r 'ENGRAM_OPENMYTHOS_TRANSFER_COMPLETE' . --exclude-dir=.git
```

The only matches should be inside `plans/OPENMYTHOS_TRANSFER.md` (none
expected) and `plans/RALPH_LOOP_RUNBOOK.md` (this file). If matches
appear elsewhere, change the promise to a fresh unique string before
launching, or the loop may self-terminate prematurely.

---

## Manual checkpoints (do NOT auto-advance)

The loop is well-suited for the bulk of the work, but two checkpoints
should halt and surface to a human:

1. **Phase 0.3 reproducibility gate.** If two seeded runs do not agree
   to 1e-5, the loop must stop and report. Papering over non-determinism
   makes every later phase result untrustworthy. Recommended:
   add a guard in the prompt above — "if Phase 0.3 fails, do not advance;
   instead emit a human-readable diagnosis and halt this iteration
   without modifying status markers."
2. **Phase 6 lock + README delta table.** Final summary requires human
   review of measured numbers. Set Phase 6 Status to `IN_PROGRESS` when
   reached, write a draft README diff, and **do not** emit the
   completion promise until a human marks Phase 6 PASSED.

To enforce these, you can either:

- Use a stricter prompt that special-cases these phases (longer launch
  command), or
- Run the loop with `--max-iterations` set lower than the expected total,
  inspect the state, then relaunch.

---

## Cancelling cleanly

`/cancel-ralph` removes `.claude/ralph-loop.local.md` and stops the loop.
Repo state (status markers, JSON results, code changes from in-flight
commits) is preserved. To resume later, relaunch with the same command —
the loop picks up at the next pending test.

---

## Pre-flight checklist

Before launching the loop:

- [ ] `bench/run.py` exists and has been run at least once successfully
      (Phase 0 minimally bootstrapped). The loop can do Phase 0 from
      scratch, but it's safer to have the first run done by a human.
- [ ] `bench/history/baseline.json` exists (Phase 0.4). The loop has no
      way to compute "baseline + 1.5pp" without it.
- [ ] Working tree is clean (`git status` reports nothing). The loop will
      commit on every PASSED phase; uncommitted human changes may get
      caught up in `git checkout -- .` on a KILLED phase.
- [ ] You're on `main` and up to date with `origin/main`.
- [ ] `EXECUTION_LOG.md` exists (touch an empty one if needed).

---

## Expected duration

Per `OPENMYTHOS_TRANSFER.md`: ~5 person-days full execution. The loop will
likely take longer in wall-clock time because each test runs
`bench/run.py` end-to-end. Plan for overnight runs; the
`--max-iterations 30` cap exists to prevent runaway costs.
