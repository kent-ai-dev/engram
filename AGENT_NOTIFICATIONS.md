# Agent Notifications - Engram Training Run

## [2026-03-19 06:30 CDT] ERROR - baseline/medium/large/target_small (Iteration 6)
**Error:** PermissionError: [WinError 32] ChromaDB SQLite file locked when trying rmtree
**Root cause:** Multiple stale Python processes from iteration 5 (train_runner.py 1-5, several ingest.py instances) were still running and holding the ChromaDB SQLite lock. Windows does not allow deletion of files held open by other processes.
**Fix applied:**
1. Killed all stale Python processes (PIDs 9396, 4156, 4412, 7400) using Stop-Process -Force
2. Cleaned up engram_memory/ directory manually
3. Patched ingest.py (lines ~111-126) to retry rmtree up to 10 times with 3s delay on PermissionError, with fallback rename if all retries fail
**Status:** Retrying from iteration 6 with clean environment...

## [2026-03-19 13:15 CDT] RESTART - Fresh training run (iterations 1-15)
**Issues identified:**
1. Previous attempts were failing with UnicodeEncodeError (arrow character → needs PYTHONIOENCODING=utf-8)
2. ChromaDB SQLite locking from stale Python processes (Windows file locking)
3. ChromaDB collections not being properly cleared between runs

**Actions taken:**
1. Killed all stale python*.exe processes
2. Deleted stale chroma.sqlite3 file from engram_memory/
3. Starting fresh train_runner.py 1 15 with $env:PYTHONIOENCODING = "utf-8" set
4. Background process started: PID 12592 (swift-cedar session)

**Status:** Training now running. Currently processing Iteration 1 (baseline config).
- Training corpus: 10 books, ~1M+ words
- Books: training_data.txt + 84_frankenstein.txt
- Baseline config: embed=64, context=8, layers=3, epochs=1
- Estimated time for iteration 1: ~30-60 min (4 configs × sequential training)
- Monitoring in place. Will report every iteration completion.

