---
active: true
iteration: 2
session_id:
max_iterations: 30
completion_promise: "ENGRAM_COHERENT"
started_at: "2026-05-03T17:30:00Z"
budget_ceiling_usd: 150
budget_spent_usd: 51
current_hypothesis: "v13 INV_TEMP 10 -> 30; sharpens cross-entropy gradient signal"
current_run: "v13_xent_temp30"
modal_app_id: "ap-caOqs6mJwgWJ7eXKZaAz3O"
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

1. **Check Modal job state**: `python3 -m modal app list 2>&1 | head` and `python3 -m modal app logs <app_id> 2>&1 | tail -10`. Also check volume: `python3 -c "import modal; v=modal.Volume.from_name('engram-weights'); [print(e.path, e.size) for e in v.listdir('/', recursive=False) if e.path.endswith('.pth')]"`.
   - If still training (epoch < 5 in logs OR volume sizes still match v12: word_to_id=295176, weights=86194735, memory_module=154786552) → reschedule wake-up 1800s. Done for this tick.
   - If weights have changed on volume → proceed to step 2.

2. **Download + stage the new model**:
   - `python3 ~/.openclaw/workspace/scripts/modal_train.py download` (writes to `~/.openclaw/workspace/memory/modal_downloaded_weights/`).
   - Create `models/<run_name>/` (e.g. `models/v13_xent_temp30/`).
   - Copy the three `.pth` files + `engram_memory/` into it.
   - Write `models/<run_name>/model_card.json` with the v12 schema (architecture, embed_dim, n_layers, vocab, epoch_losses parsed from Modal logs, INV_TEMPERATURE used, corpus, training_cost_usd, gpu, notes).

3. **Eval**: `cd /mnt/c/Users/Administrator/Documents/Github/engram && uv run eval_chat.py` — outputs to `eval_runs/chat_<ts>.json`.

4. **Deploy** (the deploy-after-training rule applies even on failure):
   - Edit `server.py` `ACTIVE_MODEL = "<run_name>"`.
   - PowerShell to find uvicorn pids → kill → relaunch via `_start_v4_server.bat`.
   - `curl http://108.181.97.223:5000/status` to confirm.

5. **Update both status pages** via the engram-page-updater agent. Brief it with: the new run's hypothesis, single-variable change, loss progression (from Modal logs), eval verdict (with quoted samples), what was learned, and the queued next-run hypothesis.

6. **Decide** based on eval shape — see `plans/V14_CANDIDATES.md` decision tree:
   - PASS → emit `ENGRAM_COHERENT`, set `active: false` here, archive.
   - PARTIAL coherence → Branch A: stronger pondering (engram innovation #3).
   - FAIL with clean floor plateau → Branch B: learnable ChromaDB embeddings (engram innovation #1).
   - FAIL with no convergence → Branch C: episodic memory at training time (engram innovation #5).
   - **Cost-stop**: if cumulative spend > $150 → halt and ask user.
   - **Stagnation-stop**: if 5 consecutive iterations show identical eval failure mode → halt, surface deeper diagnosis.

7. **Commit** the new model dir + plan + status-page changes. Push to main (so Modal git-clone gets the next ingest.py for the following iteration).

8. **Launch next training** (only if not stopping):
   - Make sure `ingest.py` and any other code is committed + pushed to main.
   - Use direct Modal CLI (the `modal_train.py launch --config` wrapper has a flag-passing bug): `cd /home/administrator/.openclaw/workspace && python3 -m modal run --detach scripts/modal_train.py --config-path /mnt/c/Users/Administrator/Documents/Github/engram/configs/engram_v<N>_config.json`.
   - Update `~/.openclaw/workspace/memory/modal_job.json` with the new app ID, run name, hypothesis.
   - Increment `iteration` here and update `current_run` / `current_hypothesis`.

9. **ScheduleWakeup** 1800s with this same prompt to continue.

## Iteration plan (rolling)

v14 candidates are now **engram-thesis-aligned, branched on v13 eval shape**. See `plans/V14_CANDIDATES.md` for the full decision tree. Summary:

| Iter | Variant | Single-variable change | Trigger |
|------|---------|------------------------|---------|
| 1 (in flight) | v13_xent_temp30 | INV_TEMP 10 → 30 | launched 2026-05-03 |
| 2 (Branch A) | v14_pondering | MAX_PONDER 3→5, ponder_cost 0.05→0.02 | v13 PARTIAL coherence |
| 2 (Branch B) | v14_learnable_embed | ChromaDB vocab as `nn.Parameter`; brain frozen first epoch | v13 hits floor cleanly + still incoherent |
| 2 (Branch C) | v14_episodic_train | Episodic memory retrieval engaged in training loop | v13 fails to converge (plateau >5 nats above floor) |
| 3+ (fallback) | v15_corpus_smoltalk | 10× corpus expansion via smol-smoltalk subsample | only if Branches A/B/C all exhausted |

The earlier conventional candidates (`v14_xent_temp100`, immediate `v14_corpus_smoltalk`) have been **demoted** — they don't exercise any engram architectural lever. Architectural innovations get tested first; data scaling is the last resort.

## Hard-won lessons (do not lose)

- `INV_TEMPERATURE` is a constant in `ingest.py:321` — NOT in `MODEL_FIELDS` or `INGEST_FIELDS` regex-patched by `modal_train.py`. Changes propagate only via git clone. Always commit + push before launch.
- Cosine cross-entropy floor scales as `~ln(V) / INV_TEMP * (entropy_factor)`. At V=14704 and dominant cosine ~0.5: floor(10) ≈ 1.77, floor(30) ≈ 0.59, floor(100) ≈ 0.18.
- Modal `git clone` pulls `main` from `https://github.com/kent-ai-dev/engram.git` with no branch flag.
- The `modal_train.py launch --config <name>` wrapper passes the config as positional after `--`, but the entrypoint takes `--config-path` as a kwarg → command fails. Bypass via direct `modal run --detach scripts/modal_train.py --config-path <abspath>`.
- `PYTHONUNBUFFERED=1` and `python -u` are mandatory inside the Modal container or training prints buffer until exit.
- Server restart on Windows requires killing both parent uvicorn AND the multiprocessing fork child holding port 5000.
- `.pth` files on the Modal volume only update at end-of-training (when `train_engram` returns). Per-epoch checkpoints written by `ingest.py` stay in the container's `/tmp/engram` and are only copied out by the volume.commit() at function exit. Don't poll the volume for mid-training progress — read Modal logs instead.

## How to halt manually

Set `active: false` in this file. Any in-flight Modal job continues until completion or 12h timeout — kill via `modal app stop <app_id>` if needed (don't unless cost-critical, you lose the partial weights).
