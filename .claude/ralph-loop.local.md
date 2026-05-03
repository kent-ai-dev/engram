---
active: true
iteration: 1
session_id:
max_iterations: 30
completion_promise: "ENGRAM_COHERENT"
started_at: "2026-05-03T17:30:00Z"
budget_ceiling_usd: 150
budget_spent_usd: 45
current_hypothesis: "v13 INV_TEMP 10 -> 30; sharpens cross-entropy gradient signal"
current_run: "v13_xent_temp30"
modal_job_state: "/home/administrator/.openclaw/workspace/memory/modal_job.json"
---

# Engram Autonomous Loop — Training to Coherent

## What this loop is doing

Iteratively training engram variants until eval shows coherent dialog. Each
iteration tests a single hypothesis, ships the result honestly to the live
frontend (108.181.97.223:5000), and updates both public status pages.

**Stop when** `eval_chat.py` shows: distinct replies per prompt + ≥80% real
English tokens + ≥3/5 chitchat prompts produce coherent dialog. Emit
`ENGRAM_COHERENT` to halt.

## How each iteration runs

When you wake up to this file with `active: true`:

1. **Check Modal job state**: `cat /home/administrator/.openclaw/workspace/memory/modal_job.json`
   - If `status` is `launched` and weights not yet on volume → still training. Reschedule wake-up 1200-1800s. Done for this tick.
   - If `weights_url` is set and `weights_local` is empty → run `python3 ~/.openclaw/workspace/scripts/modal_train.py download`.
   - If `weights_local` is set → proceed to step 2.

2. **Stage the new model**:
   - Create `models/<run_name>/` (e.g. `models/v13_xent_temp30/`).
   - Copy the three `.pth` files + `engram_memory/` into it.
   - Write `models/<run_name>/model_card.json` with the v12 schema (architecture, embed_dim, n_layers, vocab, epoch_losses, INV_TEMPERATURE used, corpus, training_cost_usd, gpu, notes).

3. **Eval**: `cd /mnt/c/Users/Administrator/Documents/Github/engram && uv run eval_chat.py` — outputs to `eval_runs/chat_<ts>.json`.

4. **Deploy** (the deploy-after-training rule applies even on failure):
   - Edit `server.py` `ACTIVE_MODEL = "<run_name>"`.
   - `powershell.exe -Command "Get-CimInstance Win32_Process -Filter ..."` to find uvicorn pids → kill → relaunch via `_start_v4_server.bat`.
   - `curl http://108.181.97.223:5000/status` to confirm.

5. **Update both status pages** via the engram-page-updater agent. Brief it with: the new run's hypothesis, single-variable change, loss progression, eval verdict (with quoted samples), what was learned, and the queued next-run hypothesis.

6. **Decide**:
   - If eval passes all 3 criteria → emit `ENGRAM_COHERENT`, set `active: false` here, archive.
   - If eval fails → identify next hypothesis, update `current_hypothesis` and `current_run` here, write a `plans/V<N>_PLAN.md`, update `ingest.py` for the next single-variable swap.
   - **Cost-stop**: if cumulative spend > $150 → halt and ask user.
   - **Stagnation-stop**: if 5 consecutive iterations show identical eval failure mode → halt, surface the deeper diagnosis to user.

7. **Commit** the new model dir + plan + status-page changes. Push to main (so Modal git-clone gets the next ingest.py for the following iteration).

8. **Launch next training** (only if not stopping):
   - Make sure `ingest.py` and any other code is committed + pushed to main.
   - `python3 ~/.openclaw/workspace/scripts/modal_train.py launch --config configs/engram_v<N>_config.json`.
   - (The `--config` argument expects a filename inside the Modal volume; the launcher uploads it via `_write_config_to_volume`. Ingest.py changes propagate via git clone.)
   - Increment `iteration` here and update `current_run` / `current_hypothesis`.

9. **ScheduleWakeup** 1200-1800s with this same prompt to continue.

## Iteration plan (rolling)

| Iter | Variant | Single-variable change | Status |
|------|---------|------------------------|--------|
| 1 | v13_xent_temp30 | INV_TEMP 10 → 30 (sharpen softmax) | launching |
| 2 (if v13 fails) | v14_xent_temp100 | INV_TEMP 30 → 100 (push to floor 0.18) — only if v13 loss landed near new floor 0.59 but eval still failed |
| 2 (alt) | v14_corpus_smoltalk | If v13 hits floor cleanly with no eval improvement: jump to 50-80 MB smol-smoltalk subsample | queued |
| 3 (if v14 fails) | v15_learnable_embed | Learnable token embeddings (was v12-plan fallback) | queued |

## Hard-won lessons (do not lose)

- `INV_TEMPERATURE` is a constant in `ingest.py:321` — NOT in `MODEL_FIELDS` or `INGEST_FIELDS` regex-patched by `modal_train.py`. Changes propagate only via git clone. Always commit + push before launch.
- Cosine cross-entropy floor scales as `~ln(V) / INV_TEMP * (entropy_factor)`. At V=14704 and dominant cosine ~0.5: floor(10) ≈ 1.77, floor(30) ≈ 0.59, floor(100) ≈ 0.18.
- Modal `git clone` pulls `main` from `https://github.com/kent-ai-dev/engram.git` with no branch flag.
- `PYTHONUNBUFFERED=1` and `python -u` are mandatory inside the Modal container or training prints buffer until exit.
- Server restart on Windows requires killing both parent uvicorn AND the multiprocessing fork child holding port 5000.

## How to halt manually

Set `active: false` in this file. Any in-flight Modal job continues until completion or 12h timeout — kill via `modal app stop engram-training` if needed (don't unless cost-critical, you lose the partial weights).
