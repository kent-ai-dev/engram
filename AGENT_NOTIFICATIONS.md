# Agent Notifications

## 2026-03-23 02:32 UTC — SaladCloud Crash Loop Fixed

**Issue:** Container `engram-1774230203` was crash-looping since 01:44 UTC — Exit:1/Exit:0 alternating every 6-8 minutes.

**Root Cause:** The `python:3.11-slim` image had to `pip install torch` (~800MB CPU PyTorch) on every cold start. With `restart_policy: "never"`, SaladCloud re-allocates to a fresh node each time, so every restart = new node = fresh pip install from scratch. Combined with 12GB RAM cap, this was either OOM-killing during pip install or the training never completed before the node was recycled.

**Fix Applied:**
1. Deleted crash-looping container `engram-1774230203`
2. Switched image to `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (torch pre-installed)
3. Bumped memory from 12GB → 16GB
4. Container command now only installs chromadb+requests+tqdm (~50MB) instead of 800MB torch
5. Added `free -m` and `2>&1` to command for better debugging
6. Added non-interactive `delete` command to salad_train.py
7. Created cron instruction file `cron_instructions/salad_watcher.md`

**New Container:** `engram-1774233103` — deployed with fixed config, currently allocating a GPU node.

**Status:** Deploying (pytorch image is larger, ~6GB, so initial pull is slower than python:3.11-slim but torch is pre-installed — startup after image pull will be much faster).

**Cron:** Watcher cron `47383ffb-38a7-4a3b-93f2-af3a0463d621` — wrote instruction file but could not update the cron payload directly (no OpenClaw CLI auth from subagent context). Main agent should verify the cron message references the new group name `engram-1774233103`.
