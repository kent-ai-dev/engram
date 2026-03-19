# Agent Notifications - Engram Training Run

## [2026-03-19 06:30 CDT] ERROR - baseline/medium/large/target_small (Iteration 6)
**Error:** PermissionError: [WinError 32] ChromaDB SQLite file locked when trying rmtree
**Root cause:** Multiple stale Python processes from iteration 5 (train_runner.py 1-5, several ingest.py instances) were still running and holding the ChromaDB SQLite lock. Windows does not allow deletion of files held open by other processes.
**Fix applied:**
1. Killed all stale Python processes (PIDs 9396, 4156, 4412, 7400) using Stop-Process -Force
2. Cleaned up engram_memory/ directory manually
3. Patched ingest.py (lines ~111-126) to retry rmtree up to 10 times with 3s delay on PermissionError, with fallback rename if all retries fail
**Status:** Retrying from iteration 6 with clean environment...

