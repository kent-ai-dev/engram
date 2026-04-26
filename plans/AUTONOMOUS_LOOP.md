# Engram Autonomous Training Loop

How the recursive self-training pipeline works, end to end. The model code
lives in this repo; the orchestration lives in the OpenClaw workspace
(`~/.openclaw/workspace/scripts/`) and runs as a cron-driven agent.

## At a glance

```
            cron 0 3 * * * UTC  (job 5d7a6161 in ~/.openclaw/cron/jobs.json)
                       │
                       ▼
  kairos_durable.py ── file-lock + 30s jitter wrapper
                       │
                       ▼
  engram_autonomous_loop.py ── orchestrator (max 3 iterations, threshold 60.0)
                       │
                       ├─► (1) engram_evaluator.py       → score live server 0-100
                       ├─► (2) engram_training_decision.py → choose hyperparam tweaks
                       ├─► (3) modal_train.py launch     → Modal L4 GPU training
                       ├─► (4) modal_train.py status     → poll until weights_url set
                       ├─► (5) modal_train.py download   → pull weights from Modal volume
                       ├─► (6) engram_deploy.py          → restart :5000 server with new weights
                       └─► (7) engram_evaluator.py       → re-score, log delta, decide loop or stop
```

## Components

### Model side (this repo)

| File | Role |
|------|------|
| `server.py` / `run_engram_server.py` | FastAPI server on port 5000. SSE chat stream at `/chat/stream`, health at `/status`. |
| `ingest.py` | Training entry point. Hyperparameters at the top get patched by the orchestrator. |
| `models/large_iter4/` | Currently deployed weights (256-dim, 8 layers). |
| `engram_weights.pth`, `engram_memory_module.pth`, `engram_word_to_id.pth` | The three checkpoint files Modal uploads back. |
| `engram_memory/` | ChromaDB vocabulary + episodic-memory store (also synced via Modal volume). |
| `training_log.jsonl` | Append-only log of every iteration's score-before/score-after/decision/weights_path. |

The live server runs at `108.181.97.223:5000`.

### Orchestrator side (`~/.openclaw/workspace/scripts/`)

| Script | Role |
|--------|------|
| `engram_autonomous_loop.py` | The loop entry point. Runs eval → decide → train → deploy → re-eval up to 3 times or until score ≥ 60.0. |
| `engram_evaluator.py` | Sends 5 fixed prompts over SSE, scores each on surprise/coherence/length/ponder, returns weighted overall (40/30/20/10). |
| `engram_training_decision.py` | Maps failure modes to hyperparam changes (high surprise → +EPOCHS, low coherence → +BATCH_SIZE, short responses → +CONTEXT_SIZE, low ponder → +N_LAYERS). Detects 3-iteration stagnation and stops. |
| `modal_train.py` | Modal.com app `engram-training`. L4 GPU, 24 GB VRAM, 2-hour timeout, persistent volume `engram-weights`. Subcommands: `launch`, `status`, `download`, `tail`. |
| `engram_deploy.py` | Copies weights to two Windows paths, kills the :5000 process via PowerShell (WSL→Windows bridge), relaunches via `Start-Process`, health-checks `/status`. |
| `engram_update_pages.py` | After each iteration, pushes results to `kent-ai-dev/claw-journal` gh-pages so they're visible at `engram.html`. |
| `kairos_durable.py` | Cron-runner wrapper. File-lock per job-id so two cron fires can't race; 30 s jitter; 2-min stale-lock detection. |
| `engram_watchdog.py` | 15-min health checker (currently disabled). Auto-deploys if a `weights_url` appears but weights haven't been downloaded yet. 30-min deploy cooldown, 2-h restart cooldown. |
| `engram_ralph_loop.py` | Alternative orchestrator that cycles four arch variants (96d×4L, 256d×8L, 192d×6L, 384d×10L). Stops when ≥3/5 chat protocols are passable. |

## The five stages in detail

### 1. Evaluate

`engram_evaluator.py --host http://108.181.97.223:5000` opens an SSE stream
to `/chat/stream` for each of these prompts:

```
hello how are you
tell me a story
what do you think about friendship
can you help me understand
what happened today
```

For each response it captures `surprise` and `ponder_steps` from the `done`
event, computes word count and unique-word ratio, and produces four sub-scores
on a 0–100 scale:

- `surprise_score`  — `(1.5 - surprise) / 1.0 * 100`, clipped (lower surprise = better)
- `coherence_score` — `unique_word_ratio * 100`
- `length_score`    — `min(word_count / 10, 1) * 100`
- `ponder_score`    — `(1 - (ponder - 1)/2) * 100`, inverted so 1 step = 100

Overall = `0.40*surprise + 0.30*coherence + 0.20*length + 0.10*ponder`.
Pass threshold is **60.0**.

### 2. Decide

`engram_training_decision.py` reads the eval JSON and `training_log.jsonl`.
It returns `should_train: false` if either:

- The score already passes threshold, or
- The last 3 logged training runs all had a score delta ≤ 0.5 (stagnation).

Otherwise it identifies failure modes and proposes a config:

| Failure mode | Trigger | Adjustment |
|--------------|---------|------------|
| `high_surprise` | `surprise_score < 40` | `EPOCHS += 2` (cap 10) |
| `repetition_problem` | `coherence_score < 50` | `BATCH_SIZE += 128` (cap 512) |
| `short_responses` | `length_score < 40` | `CONTEXT_SIZE += 16` (cap 64) |
| `low_ponder` | `ponder_score < 30` | `N_LAYERS += 2` (cap 8) |
| `below_threshold` | (default) | `EPOCHS += 1` |

Baseline config: `EMBED_DIM=96, CONTEXT_SIZE=32, N_LAYERS=4, EPOCHS=3, BATCH_SIZE=256`.

### 3. Train (Modal)

`modal_train.py launch` runs `modal run --detach scripts/modal_train.py` in
the background. Inside the container (`pytorch/pytorch:2.5.1-cuda12.4` + L4
GPU), `train_engram` does:

1. Load config from the volume (or use the defaults baked into the function).
2. `git clone https://github.com/kent-ai-dev/engram.git /tmp/engram`.
3. Regex-patch hyperparameters into `ingest.py`.
4. Download the TinyStories validation set into `corpus/tinystories_val.txt`.
5. Run `python ingest.py` to train.
6. `shutil.copy2` the three `.pth` files and `shutil.copytree` `engram_memory/`
   onto the volume mount.
7. `volume.commit()`.

The launcher records `memory/modal_job.json`:

```json
{ "status": "launched", "launched_at": "...", "pid": 1234, "log_file": "..." }
```

### 4. Poll for completion

`engram_autonomous_loop.wait_for_training()` polls every 60 seconds for up to
8 hours. Each poll:

1. Reads `memory/modal_job.json` directly. If `weights_url` is set → done.
2. If `status` is `failed`/`error`/`crashed`/`stopped` → fail.
3. Otherwise calls `modal_train.py status`, which:
   - Lists the volume contents via `v.listdir("/", recursive=True)` (Modal 1.x API).
   - If all three expected `.pth` files are at the volume root, writes
     `weights_url: "modal_volume://engram-weights"` and `status: "completed"`
     into `modal_job.json`. Next poll reads this and returns success.

### 5. Download + deploy

`modal_train.py download`:
- Streams each `.pth` via `v.read_file_into_fileobj(path, fileobj)`.
- Walks `engram_memory/` recursively and mirrors the directory tree.
- Writes everything under `~/.openclaw/workspace/memory/modal_downloaded_weights/`.
- Records `weights_local` in `modal_job.json` once `engram_weights.pth`
  is non-zero on disk.

`engram_deploy.py --weights-path …`:
1. `shutil.copy2` to `…/engram/engram_weights.pth` and
   `…/engram/models/baseline_iter2/engram_weights.pth`.
2. `netstat | findstr :5000` via PowerShell, `Stop-Process` the PID.
3. `Start-Process` of `.venv\Scripts\python.exe run_engram_server.py`
   detached, hidden window.
4. Poll `http://108.181.97.223:5000/status` up to 5 times (3 s apart).
   Server must report `loaded: true`.

### Re-evaluate and decide loop continuation

After deploy, sleep 10 s, re-run the evaluator. Append to `training_log.jsonl`:

```json
{
  "timestamp": "...",
  "iteration": 1,
  "score_before": 42.1,
  "score_after": 51.7,
  "improvement": 9.6,
  "decision": { ... },
  "weights_path": "/.../engram_weights.pth"
}
```

Stop conditions:
- New score ≥ threshold → `PASS`
- Improvement ≤ 0       → `NO_IMPROVEMENT`
- Iteration count == max → `MAX_ITER`

## State and observability

| File | Contents |
|------|----------|
| `~/.openclaw/cron/jobs.json` | Cron job definitions. Job `5d7a6161` is the daily loop. |
| `~/.openclaw/workspace/memory/engram_loop.log` | Full orchestrator log (every poll, every script call). |
| `~/.openclaw/workspace/memory/modal_job.json` | Current Modal training run state. |
| `~/.openclaw/workspace/memory/modal_job.log` | Raw Modal CLI output captured during launch. |
| `~/.openclaw/workspace/memory/ralph_state.json` | Ralph-loop iteration counter + best-score tracking. |
| `~/.openclaw/workspace/memory/engram_watchdog_state.json` | Last watchdog run + consecutive-failure counter. |
| `~/.openclaw/workspace/memory/engram-training-history.md` | Human-curated log of every training run since 2026-03. |
| `<engram>/training_log.jsonl` | Iteration deltas (this repo). |
| `~/.openclaw/workspace/memory/discord_notify_queue.jsonl` | Discord notification queue — main session relays to channel `1467302416117403828` / `1467304221362622555`. |

Notifications are best-effort: scripts append to the queue file and the main
Claude Code session forwards them via the Discord plugin. There is no direct
webhook from the orchestrator to Discord.

## Cron schedule

Defined in `~/.openclaw/cron/jobs.json`:

| ID prefix | Name | Schedule | State |
|-----------|------|----------|-------|
| `5d7a6161` | Engram Daily Training Loop | `0 3 * * *` UTC | currently disabled |
| `663294b5` | Engram Training Watchdog (every) | every 60 min | disabled |
| `b9798c4a` | Engram Training Watchdog (cron) | `*/15 * * * *` | disabled |
| `47383ffb` | SaladCloud Training Watcher | every 5 min | disabled |

The daily loop wakes an isolated Sonnet agent which runs:

```
python3 ~/.openclaw/workspace/scripts/kairos_durable.py \
    --job-id engram-training-loop \
    --command 'python3 ~/.openclaw/workspace/scripts/engram_autonomous_loop.py --max-iterations 3 --threshold 60.0' \
    --jitter-ms 0
```

Timeout: 36 000 s (10 h, generous for the worst-case 8 h training wait + 2 h overhead).

## Manual operations

```bash
# One-off eval against the live server
python3 ~/.openclaw/workspace/scripts/engram_evaluator.py \
    --host http://108.181.97.223:5000 \
    --output /tmp/eval.json

# Run one full iteration manually
python3 ~/.openclaw/workspace/scripts/engram_autonomous_loop.py \
    --max-iterations 1 --threshold 60.0

# Force-train even if passing
python3 ~/.openclaw/workspace/scripts/engram_autonomous_loop.py --force-train

# Check Modal volume state
python3 ~/.openclaw/workspace/scripts/modal_train.py status

# Download trained weights
python3 ~/.openclaw/workspace/scripts/modal_train.py download

# Tail the orchestrator log
tail -f ~/.openclaw/workspace/memory/engram_loop.log
```

## Known issues at time of writing (2026-04-25)

- **Modal SDK breaking changes (1.x).** `Volume.files()`, `Volume.info(path)`,
  and `Volume.download(src, dst)` were removed. `modal_train.py` was patched
  to use `listdir(path, recursive=True)` and `read_file_into_fileobj(path, fileobj)`.
  If the SDK changes again, the symptom is `'Volume' object has no attribute …`
  in `engram_loop.log` and the orchestrator polling forever without ever seeing
  `weights_url` get written.
- **`weights_url` is set lazily.** It is only written when `cmd_status` sees
  all three expected files on the volume. If training fails partway and only
  uploads two of three, the loop will poll until the 8 h `MAX_TRAINING_WAIT`
  expires.
- **Two pollers can race.** If two `kairos_durable` fires happen within the
  jitter window before one acquires the lock file, the orchestrator log will
  show interleaved poll counts (e.g. `poll 386` and `poll 424` in the same
  minute). Lock file lives at `~/.openclaw/workspace/memory/cron_locks/engram-training-loop.lock`.
- **Zero-byte `.pth` partials.** Earlier versions of `cmd_download` left
  empty files when a download failed. Now patched to `unlink` partials and
  to record `weights_local` only if the file is non-zero.
- **The watchdog is currently disabled.** Nothing is auto-recovering stuck
  Modal runs. Re-enable `663294b5` or `b9798c4a` in `jobs.json` to restore
  that safety net.

## Stopping the loop

Edit `~/.openclaw/cron/jobs.json` and set `enabled: false` on job
`5d7a6161-02b3-4146-830f-ed0bc7de4601`. The OpenClaw cron daemon picks up
the change without restart. Any in-flight orchestrator process keeps running
until it completes or hits its 8 h `MAX_TRAINING_WAIT`; kill it manually if
that's not acceptable:

```bash
pkill -f engram_autonomous_loop.py
pkill -f 'modal_train.py status'
```

---

## Update — 2026-04-26: ralph-loop pattern + v4_rope deployment

The cron-driven loop above is the **batch** pattern. There is now a
complementary **interactive** pattern using ralph-loop, suitable for
multi-iteration objectives that span Claude Code sessions.

### When to use which

- **Cron loop (batch)**: continuous unattended improvement. Daily 3am UTC fire.
  Decision logic in `engram_training_decision.py`. Score-based pass/fail.
- **Ralph loop (interactive)**: focused multi-iteration objective (e.g.,
  "train until coherent", "transfer architecture from OpenMythos"). State
  in `.claude/ralph-loop.local.md`. Completion-promise-based stop.

### Ralph-loop state file

`.claude/ralph-loop.local.md`:

```yaml
---
active: true
iteration: 1
session_id:
max_iterations: 30
completion_promise: "ENGRAM_COHERENT"
started_at: "2026-04-26T01:55:00Z"
---

<the prompt that re-feeds each iteration>
```

When `active: true`, ralph-loop re-feeds the prompt at the end of each
session. When the prompt's output anywhere contains the
`completion_promise` literal string, the loop self-stops.

### Past ralph-loops

- **OpenMythos transfer** (2026-04-25, completed):
  `completion_promise: "ENGRAM_OPENMYTHOS_TRANSFER_COMPLETE"`. Six phases
  with strict pass/kill criteria from `bench/run.py`. Only Phase 5 (RoPE)
  shipped; Phases 1, 2 killed; Phases 3, 4 skipped.
  See `plans/OPENMYTHOS_TRANSFER.md` and `plans/EXECUTION_LOG.md`.

### Active ralph-loop — Training to Coherent

- **Started**: 2026-04-26
- **Completion promise**: `ENGRAM_COHERENT`
- **Criterion**:
  1. `eval_chat.py` produces **different replies for different prompts**.
  2. ≥80% of generated tokens are real English words.
  3. ≥3 of 5 chitchat prompts produce a coherent dialog-shaped reply.
- **Kill**: 5 consecutive iterations with no improvement OR cumulative
  Modal spend > $50.

Each iteration **must** (in order):

1. Execute the next concrete training/eval action.
2. Run `eval_chat.py` after every training run.
3. Update **both** status pages — `index.html` (engram repo, `main`)
   and `engram.html` (claw-journal repo, `gh-pages` branch).
4. Append a one-line entry to `plans/EXECUTION_LOG.md`.
5. Commit results to git.
6. Decide pass / escalate / stop.
7. If still iterating, schedule next wakeup via `ScheduleWakeup`
   (1200–1800s).

### Iteration plan

| Iter | Variant | Change | Wall time | Cost |
|------|---------|--------|-----------|------|
| 1 | **v5** | 3 epochs · same 15 MB local corpus · L4 | ~4.5h | ~$3.60 |
| 2 | **v5b** (if v5 fails) | 5 epochs · dailydialog-only (6 MB) · L4 | ~3h | ~$2.40 |
| 3 | **v6** | 50M params · 12 layers · 384-dim · A10G | ~12h | ~$13 |
| 4 | **v7** | + 200 MB conversational corpus · L40S | ~30h | ~$60 |
| 5 | **v8** | 100M params · 1 GB corpus · L40S | ~80h | ~$160 |

### Hard-won lessons (do not lose)

- **`PYTHONUNBUFFERED=1` is mandatory** for `subprocess.run(["python",
  "ingest.py"])` on Modal. Without it, training prints buffer until
  exit and you can't watch progress. Use `python -u` too.
- **Local entrypoints run locally**, not in Modal containers. Don't
  open `/engram-weights/foo` from a `@app.local_entrypoint()` — that
  path only exists inside `train_engram`. Use a separate Modal function
  to upload config to the volume.
- **TinyStories validation** is 18 MB and explodes the dataset to 6.3M
  sequences, which doesn't fit Modal's 2-hour timeout. Now opt-in via
  `USE_TINYSTORIES=true` in the config.
- **Modal default timeout** is `timeout=7200` (2h) in `modal_train.py`.
  Bump it for longer runs; cost is per-minute regardless.
- **Bench `eval_cosine_top1` is broken at small scale** — it uses
  `torch.randn()` embeddings and is stuck at 5.0% across every
  architectural variant. Don't trust it. Use `eval_chat.py`.
- **Server restart on Windows** requires killing the orphan
  multiprocessing fork child, not just the parent. `Stop-Process`
  on the parent leaves the child still binding port 5000.

### Stopping the ralph-loop

- **Manual**: edit `.claude/ralph-loop.local.md`, set `active: false`.
  Also `TaskStop` any active Monitors.
- **Self-stop**: emit `completion_promise` string in any output.
- **Cost stop**: if Modal spend > $50 with no convergence, halt and
  ask the user.
