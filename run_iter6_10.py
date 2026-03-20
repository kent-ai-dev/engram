"""
run_iter6_10.py -- Direct training runner for iterations 6-10.
Writes progress to progress.json after each config completes.
Designed to be killed and resumed safely.
"""
import subprocess
import sys
import os
import re
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).parent.resolve()
PROGRESS_FILE = REPO_DIR / "iter6_10_progress.json"
LOG_FILE = REPO_DIR / "training_log.jsonl"
MODELS_DIR = REPO_DIR / "models"
INGEST_PATH = REPO_DIR / "ingest.py"
EVAL_PATH = REPO_DIR / "eval_brain.py"

CONFIGS = [
    {"name": "baseline",     "embed_dim": 64,  "context_size": 8,  "n_layers": 3, "epochs": 1, "batch_size": 512},
    {"name": "medium",       "embed_dim": 96,  "context_size": 12, "n_layers": 4, "epochs": 1, "batch_size": 512},
    {"name": "large",        "embed_dim": 128, "context_size": 16, "n_layers": 5, "epochs": 1, "batch_size": 512},
    {"name": "target_small", "embed_dim": 256, "context_size": 8,  "n_layers": 6, "epochs": 1, "batch_size": 512},
]

ITERATIONS = list(range(6, 11))  # 6, 7, 8, 9, 10


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(REPO_DIR / "iter6_10.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_progress():
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed": []}  # list of "iter:config" strings


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def get_utf8_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def update_hyperparams(embed_dim, context_size, n_layers, epochs, batch_size=256):
    with open(INGEST_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"^EMBED_DIM\s*=\s*\d+", f"EMBED_DIM = {embed_dim}", content, flags=re.MULTILINE)
    content = re.sub(r"^CONTEXT_SIZE\s*=\s*\d+", f"CONTEXT_SIZE = {context_size}", content, flags=re.MULTILINE)
    content = re.sub(r"^N_LAYERS\s*=\s*\d+", f"N_LAYERS = {n_layers}", content, flags=re.MULTILINE)
    content = re.sub(r"^EPOCHS\s*=\s*\d+", f"EPOCHS = {epochs}", content, flags=re.MULTILINE)
    content = re.sub(r"^BATCH_SIZE\s*=\s*\d+", f"BATCH_SIZE = {batch_size}", content, flags=re.MULTILINE)
    with open(INGEST_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_content = f.read()
    eval_content = re.sub(r"^EMBED_DIM\s*=\s*\d+", f"EMBED_DIM = {embed_dim}", eval_content, flags=re.MULTILINE)
    eval_content = re.sub(r"^CONTEXT_SIZE\s*=\s*\d+", f"CONTEXT_SIZE = {context_size}", eval_content, flags=re.MULTILINE)
    eval_content = re.sub(r"^N_LAYERS\s*=\s*\d+", f"N_LAYERS = {n_layers}", eval_content, flags=re.MULTILINE)
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        f.write(eval_content)


def run_ingest_with_live_output(attempt_id):
    """Run ingest.py writing progress to a temp file so we can monitor it."""
    tmp_stdout = REPO_DIR / f"ingest_out_{attempt_id}.txt"
    tmp_stderr = REPO_DIR / f"ingest_err_{attempt_id}.txt"

    proc = subprocess.Popen(
        [sys.executable, "-u", str(INGEST_PATH)],
        cwd=REPO_DIR,
        stdout=open(tmp_stdout, "w", encoding="utf-8"),
        stderr=open(tmp_stderr, "w", encoding="utf-8"),
        env=get_utf8_env(),
    )

    log(f"    Ingest PID: {proc.pid} | monitoring...")
    last_report = time.time()
    last_size = 0

    try:
        while proc.poll() is None:
            time.sleep(30)
            try:
                size = tmp_stdout.stat().st_size
                if size != last_size:
                    # Read last few lines for progress
                    with open(tmp_stdout, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    progress_lines = [l.strip() for l in lines if "%" in l or "Loss" in l or "Epoch" in l]
                    if progress_lines:
                        log(f"    Progress: {progress_lines[-1]}")
                    last_size = size
                else:
                    elapsed = time.time() - last_report
                    if elapsed > 120:
                        log(f"    Still running (no new output for {elapsed:.0f}s)...")
                        last_report = time.time()
            except Exception:
                pass
    except Exception as e:
        log(f"    Monitor error: {e}")
        proc.kill()

    rc = proc.wait()

    # Read output
    stdout = ""
    stderr = ""
    try:
        stdout = tmp_stdout.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        stderr = tmp_stderr.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # Cleanup temp files
    try:
        tmp_stdout.unlink()
        tmp_stderr.unlink()
    except Exception:
        pass

    return rc, stdout, stderr


def parse_training_output(stdout):
    loss = None
    ponder = None
    for line in stdout.splitlines():
        m = re.search(r"Avg Loss:\s*([\d.]+).*Avg Ponder Steps:\s*([\d.]+)", line)
        if m:
            loss = float(m.group(1))
            ponder = float(m.group(2))
    return loss, ponder


def parse_eval_output(stdout):
    for line in stdout.splitlines():
        if line.startswith("EVAL_RESULT:"):
            try:
                return json.loads(line[len("EVAL_RESULT:"):].strip())
            except json.JSONDecodeError:
                pass
    return None


def save_checkpoint(config_name, iteration):
    dest = MODELS_DIR / f"{config_name}_iter{iteration}"
    dest.mkdir(parents=True, exist_ok=True)
    weights_src = REPO_DIR / "engram_weights.pth"
    if weights_src.exists():
        shutil.copy2(weights_src, dest / "engram_weights.pth")
    chroma_src = REPO_DIR / "engram_memory"
    chroma_dest = dest / "engram_memory"
    if chroma_src.exists():
        if chroma_dest.exists():
            shutil.rmtree(chroma_dest)
        shutil.copytree(chroma_src, chroma_dest)
    log(f"    Checkpoint saved: {dest.name}")


def append_training_log(entry):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def notify(msg):
    notifications_file = REPO_DIR / "AGENT_NOTIFICATIONS.md"
    with open(notifications_file, "a", encoding="utf-8") as f:
        f.write(f"\n{msg}\n")


def run_config(iteration, config):
    """Train one config. Returns result dict or None on failure."""
    name = config["name"]
    log(f"  Config: {name} (embed={config['embed_dim']}, ctx={config['context_size']}, layers={config['n_layers']})")

    # Patch hyperparams
    update_hyperparams(
        config["embed_dim"], config["context_size"],
        config["n_layers"], config["epochs"],
        batch_size=config.get("batch_size", 512)
    )

    # Train with retries
    attempt_id = f"iter{iteration}_{name}_{int(time.time())}"
    t0 = time.time()
    rc, stdout, stderr = None, "", ""
    for attempt in range(3):
        log(f"    Training attempt {attempt+1}/3...")
        try:
            rc, stdout, stderr = run_ingest_with_live_output(attempt_id + f"_a{attempt}")
        except Exception as e:
            log(f"    Exception during training: {e}")
            rc = -1
            stderr = str(e)

        if rc == 0:
            break
        else:
            error_summary = (stderr[-500:] if stderr else stdout[-300:]).replace("\n", " ")
            log(f"    FAILED (rc={rc}): {error_summary[:200]}")
            notify(f"## [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CDT] ERROR - {name}/iter{iteration}\n"
                   f"**Error:** Training failed rc={rc}\n"
                   f"**Root cause:** {error_summary[:300]}\n"
                   f"**Fix applied:** Retrying (attempt {attempt+2}/3)\n"
                   f"**Status:** Retrying...\n")
            # Clean ChromaDB before retry
            time.sleep(5)
            chroma_dir = REPO_DIR / "engram_memory"
            if chroma_dir.exists():
                try:
                    shutil.rmtree(chroma_dir)
                    log(f"    Cleared engram_memory for retry")
                except Exception as e2:
                    log(f"    Could not clear engram_memory: {e2}")
            time.sleep(3)

    elapsed = time.time() - t0

    if rc != 0:
        error_summary = (stderr[-500:] if stderr else stdout[-300:]).replace("\n", " ")
        log(f"    FAILED after 3 attempts")
        append_training_log({
            "type": "train_error",
            "iteration": iteration,
            "config": name,
            "error": error_summary,
            "timestamp": datetime.now().isoformat(),
        })
        return None

    loss, ponder = parse_training_output(stdout)
    log(f"    Training done in {elapsed:.0f}s | loss={loss}, ponder={ponder}")

    # Show last training lines
    last_lines = [l.strip() for l in stdout.splitlines() if l.strip()][-5:]
    for line in last_lines:
        log(f"      {line}")

    # Evaluate
    log(f"    Evaluating...")
    try:
        eval_result = subprocess.run(
            [sys.executable, str(EVAL_PATH), name, str(iteration)],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=600,
            env=get_utf8_env(),
            encoding="utf-8",
        )
        eval_data = parse_eval_output(eval_result.stdout) if eval_result.returncode == 0 else None
        if eval_result.returncode != 0:
            log(f"    Eval failed (rc={eval_result.returncode}): {eval_result.stderr[:200]}")
    except Exception as e:
        log(f"    Eval exception: {e}")
        eval_data = None

    # Wait for ChromaDB lock release
    time.sleep(5)

    # Save checkpoint
    save_checkpoint(name, iteration)

    result = {
        "config": name,
        "embed_dim": config["embed_dim"],
        "final_loss": loss,
        "avg_ponder_steps": ponder,
        "train_time_sec": round(elapsed, 1),
        "coherence_score": eval_data.get("coherence_score") if eval_data else None,
        "vocab_size": eval_data.get("vocab_size") if eval_data else None,
        "avg_surprise": eval_data.get("avg_surprise") if eval_data else None,
        "avg_response_length": eval_data.get("avg_response_length") if eval_data else None,
    }
    return result


def main():
    log("=" * 60)
    log("ENGRAM TRAINING RUNNER: Iterations 6-10")
    log("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()
    completed = set(progress["completed"])
    log(f"Resuming from progress: {len(completed)} configs already done")

    all_results = {}

    for iteration in ITERATIONS:
        log(f"\n{'='*50}")
        log(f"ITERATION {iteration}")
        log(f"{'='*50}")

        iter_results = []

        for config in CONFIGS:
            key = f"{iteration}:{config['name']}"
            if key in completed:
                log(f"  SKIP (already done): {config['name']}")
                continue

            result = run_config(iteration, config)

            if result:
                iter_results.append(result)
                completed.add(key)
                progress["completed"] = list(completed)
                save_progress(progress)
                log(f"  SUCCESS: {config['name']} loss={result.get('final_loss')}")
            else:
                log(f"  FAILED: {config['name']} - continuing to next config")

        # Log iteration to training_log.jsonl
        from pathlib import Path
        books = sorted(f.name for f in (REPO_DIR / "corpus").glob("*.txt")) if (REPO_DIR / "corpus").exists() else []
        words = sum(
            len(re.findall(r"\b\w+\b", f.read_text(encoding="utf-8", errors="replace")))
            for f in (REPO_DIR / "corpus").glob("*.txt")
        )
        append_training_log({
            "type": "iteration_complete",
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "books_downloaded": [],
            "corpus_books": books,
            "corpus_words": words,
            "config_results": iter_results,
        })

        log(f"\nIter {iteration} complete: {len(iter_results)}/{len(CONFIGS)} configs succeeded")
        if iter_results:
            best = min(iter_results, key=lambda x: x.get("final_loss") or 999)
            log(f"Best config: {best['config']} (loss={best.get('final_loss', 'N/A')})")

        # Git commit
        try:
            subprocess.run(["git", "add", "."], cwd=REPO_DIR, capture_output=True)
            subprocess.run(["git", "commit", "-m", f"Iteration {iteration}: {len(iter_results)} configs trained"],
                           cwd=REPO_DIR, capture_output=True)
        except Exception as e:
            log(f"Git commit failed: {e}")

    # Final summary
    log("\n" + "=" * 60)
    log("TRAINING COMPLETE: Iterations 6-10")
    log("=" * 60)

    books = sorted(f.name for f in (REPO_DIR / "corpus").glob("*.txt")) if (REPO_DIR / "corpus").exists() else []
    words = sum(
        len(re.findall(r"\b\w+\b", f.read_text(encoding="utf-8", errors="replace")))
        for f in (REPO_DIR / "corpus").glob("*.txt")
    )

    log(f"Final corpus: {words:,} words, {len(books)} books")
    log(f"Total configs completed: {len(completed)}")

    notify(
        f"\n## TRAINING COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Iterations completed: 6-10\n"
        f"Final corpus: {words:,} words, {len(books)} books\n"
        f"Configs done: {sorted(completed)}\n"
        f"Weights: engram_weights.pth updated\n"
    )

    # Restore baseline
    update_hyperparams(64, 8, 3, 1, batch_size=512)
    log("Restored baseline hyperparams")

    # Final git commit
    try:
        subprocess.run(["git", "add", "."], cwd=REPO_DIR, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Training complete: iterations 6-10"],
                       cwd=REPO_DIR, capture_output=True)
        log("Final git commit done")
    except Exception as e:
        log(f"Final git commit failed: {e}")


if __name__ == "__main__":
    main()
