---
active: true
iteration: 2
session_id:
max_iterations: 10
completion_promise: "ENGRAM_COHERENT"
started_at: "2026-04-26T02:00:00Z"
last_halted_at: "2026-04-26T07:00:00Z"
last_halt_reason: "v5 diagnosed: Post-LN architecture caused mode collapse — block 0 attention softmax went near-uniform (max 0.033) and FF norm exploded to 360 vs residual 16, drowning input signal. Untrained model showed cos 0.62 (healthy); training collapsed to cos 1.0 (constant output)."
fix_applied: "Pre-LN architecture in engram_model.py AttentionBlock.forward (LN before each sublayer instead of after residual) + AdamW with weight_decay=0.01 in ingest.py and bench/run.py. Untrained Pre-LN model verified to preserve input differentiation (cos +0.15 after 8 layers, ||Δ|| 20.6, norm growth controlled 3→16)."
prior_loop_completed: "ENGRAM_OPENMYTHOS_TRANSFER_COMPLETE at 2026-04-25T22:40Z (commit 581771c)"
---

Engram autonomous training loop. Read plans/AUTONOMOUS_LOOP.md (Update — 2026-04-26 section) for the full iteration plan.

Goal: train an engram model that produces coherent dialog. Stop when eval_chat.py shows (1) different replies for different prompts, (2) ≥80% real-English tokens, (3) ≥3 of 5 chitchat prompts produce a coherent dialog-shaped reply. Emit ENGRAM_COHERENT to stop.

Each iteration must:
1. Identify the next variant per the iteration table in plans/AUTONOMOUS_LOOP.md (currently: v5 → v5b → v6 → v7 → v8).
2. Launch training on Modal via `python3 -m modal run ~/.openclaw/workspace/scripts/modal_train.py --config-path <config>`.
3. After Volume committed, download artifacts via `python3 ~/.openclaw/workspace/scripts/modal_train.py download`.
4. Copy weights into a versioned `models/<variant>/` dir + repo root.
5. Restart the Windows server via `_start_v4_server.bat` after killing the existing orphan child (find via Get-CimInstance and Stop-Process).
6. Run `python3 eval_chat.py` and read the transcript.
7. Judge coherence against the three criteria. If satisfied → emit ENGRAM_COHERENT.
8. If not, escalate to next variant per the table (v5b → v6 → v7 → v8).
9. Update BOTH status pages: `index.html` in engram repo (main) and `engram.html` in claw-journal repo (gh-pages branch).
10. Append one-line entry to plans/EXECUTION_LOG.md.
11. Commit code/docs to engram main; commit engram.html to claw-journal gh-pages.
12. Schedule next wakeup via ScheduleWakeup (1200-1800s) so the loop continues after compaction.

Hard kills:
- 5 consecutive iterations with no improvement in criterion 1 (different replies per prompt) → halt.
- Cumulative Modal spend > $50 → halt and ask the user.

Iteration 1 (v5) plan:
- Config: EPOCHS=3, EMBED_DIM=256, N_LAYERS=8, BATCH_SIZE=128, BRAIN_LR=1e-3.
- Modal timeout: bump to 16200s (4.5h) in modal_train.py if not already.
- Corpus: same 15 MB local (no TinyStories yet — will add in v5b/v6 if needed).
- Cost: ~$3.60.
