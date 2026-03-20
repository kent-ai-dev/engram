"""
train_runner.py -- Autonomous training orchestrator for Engram.
Manages config matrix, trains each configuration, evaluates, and logs results.
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

# Config matrix: gradual scaling
# Epochs=1 for CPU feasibility; corpus grows each iteration providing more data
# target configs use smaller context to be tractable on CPU
# NOTE: target_small was originally embed_dim=256 which exceeds 3600s CPU timeout (~4-7h).
# Replaced with embed_dim=128, ctx=8, layers=6 (estimated ~1900s, same embed as large but
# deeper/smaller-context) to stay within the 3600s training timeout on CPU.
CONFIGS = [
    {"name": "baseline",     "embed_dim": 64,  "context_size": 8,  "n_layers": 3, "epochs": 1, "batch_size": 512},
    {"name": "medium",       "embed_dim": 96,  "context_size": 12, "n_layers": 4, "epochs": 1, "batch_size": 512},
    {"name": "large",        "embed_dim": 128, "context_size": 16, "n_layers": 5, "epochs": 1, "batch_size": 512},
    {"name": "target_small", "embed_dim": 128, "context_size": 8,  "n_layers": 6, "epochs": 1, "batch_size": 512},
]

LOG_FILE = REPO_DIR / "training_log.jsonl"
MODELS_DIR = REPO_DIR / "models"
INGEST_PATH = REPO_DIR / "ingest.py"
EVAL_PATH = REPO_DIR / "eval_brain.py"


def get_utf8_env():
    """Return environment dict with PYTHONIOENCODING=utf-8."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_cmd(cmd, cwd=None, timeout=7200):
    """Run a command and return (returncode, stdout, stderr).
    Returns (-1, '', 'TIMEOUT: ...') on timeout instead of raising.
    """
    print(f"  [CMD] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or REPO_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=get_utf8_env(),
            encoding="utf-8",
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as te:
        err_msg = f"TIMEOUT: Command timed out after {timeout}s: {' '.join(cmd) if isinstance(cmd, list) else cmd}"
        print(f"  [TIMEOUT] {err_msg}")
        return -1, "", err_msg
    except Exception as e:
        err_msg = f"EXCEPTION in run_cmd: {type(e).__name__}: {e}"
        print(f"  [ERROR] {err_msg}")
        return -1, "", err_msg


def update_hyperparams(embed_dim, context_size, n_layers, epochs, batch_size=256):
    """Patch hyperparameters in ingest.py using regex."""
    with open(INGEST_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"^EMBED_DIM\s*=\s*\d+", f"EMBED_DIM = {embed_dim}", content, flags=re.MULTILINE)
    content = re.sub(r"^CONTEXT_SIZE\s*=\s*\d+", f"CONTEXT_SIZE = {context_size}", content, flags=re.MULTILINE)
    content = re.sub(r"^N_LAYERS\s*=\s*\d+", f"N_LAYERS = {n_layers}", content, flags=re.MULTILINE)
    content = re.sub(r"^EPOCHS\s*=\s*\d+", f"EPOCHS = {epochs}", content, flags=re.MULTILINE)
    content = re.sub(r"^BATCH_SIZE\s*=\s*\d+", f"BATCH_SIZE = {batch_size}", content, flags=re.MULTILINE)

    with open(INGEST_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    # Also patch eval_brain.py so it loads with the right architecture
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_content = f.read()

    eval_content = re.sub(r"^EMBED_DIM\s*=\s*\d+", f"EMBED_DIM = {embed_dim}", eval_content, flags=re.MULTILINE)
    eval_content = re.sub(r"^CONTEXT_SIZE\s*=\s*\d+", f"CONTEXT_SIZE = {context_size}", eval_content, flags=re.MULTILINE)
    eval_content = re.sub(r"^N_LAYERS\s*=\s*\d+", f"N_LAYERS = {n_layers}", eval_content, flags=re.MULTILINE)

    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        f.write(eval_content)

    print(f"  Patched hyperparams: embed_dim={embed_dim}, context_size={context_size}, n_layers={n_layers}, epochs={epochs}")


def parse_training_output(stdout):
    """Extract final loss and ponder steps from ingest.py stdout."""
    loss = None
    ponder = None
    for line in stdout.splitlines():
        m = re.search(r"Avg Loss:\s*([\d.]+).*Avg Ponder Steps:\s*([\d.]+)", line)
        if m:
            loss = float(m.group(1))
            ponder = float(m.group(2))
    return loss, ponder


def parse_eval_output(stdout):
    """Extract eval result JSON from eval_brain.py stdout."""
    for line in stdout.splitlines():
        if line.startswith("EVAL_RESULT:"):
            try:
                return json.loads(line[len("EVAL_RESULT:"):].strip())
            except json.JSONDecodeError:
                pass
    return None


def save_model_checkpoint(config_name, iteration):
    """Copy current weights and ChromaDB to models/{config_name}_iter{iteration}/."""
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

    print(f"  Checkpoint saved: {dest}")


def append_log(entry):
    """Append a JSON line to training_log.jsonl."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def git_commit(message):
    """Git add + commit."""
    try:
        run_cmd(["git", "add", "."], cwd=REPO_DIR)
        rc, out, err = run_cmd(["git", "commit", "-m", message], cwd=REPO_DIR)
        if rc == 0:
            print(f"  Git commit: {message}")
        else:
            print(f"  Git commit skipped (nothing to commit or no git identity configured)")
    except Exception as e:
        print(f"  Git commit failed: {e}")


def corpus_word_count():
    """Count total words in corpus/ and training_data.txt."""
    total = 0
    for fpath in [REPO_DIR / "training_data.txt"]:
        if fpath.exists():
            total += len(re.findall(r"\b\w+\b", fpath.read_text(encoding="utf-8", errors="replace")))
    corpus_dir = REPO_DIR / "corpus"
    if corpus_dir.exists():
        for f in corpus_dir.glob("*.txt"):
            total += len(re.findall(r"\b\w+\b", f.read_text(encoding="utf-8", errors="replace")))
    return total


def list_corpus_books():
    """Return list of book filenames in corpus/."""
    corpus_dir = REPO_DIR / "corpus"
    if not corpus_dir.exists():
        return []
    return sorted(f.name for f in corpus_dir.glob("*.txt"))


def download_books(book_ids):
    """Download books by ID using download_book.py (non-interactive mode via stdin)."""
    ids_str = " ".join(book_ids)
    print(f"  Downloading books: {ids_str}")
    result = subprocess.run(
        [sys.executable, str(REPO_DIR / "download_book.py")],
        input=ids_str + "\n",
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        timeout=300,
        env=get_utf8_env(),
        encoding="utf-8",
    )
    # Print ASCII-safe subset of output
    safe_out = result.stdout.encode("ascii", errors="replace").decode("ascii")
    print(safe_out[-2000:] if len(safe_out) > 2000 else safe_out)
    if result.returncode != 0:
        safe_err = result.stderr.encode("ascii", errors="replace").decode("ascii")
        print(f"  Download stderr: {safe_err[:500]}")
    return result.returncode == 0


# Books to download in order (spread across iterations)
BOOK_SCHEDULE = [
    ["84"],        # Iter 1: Frankenstein
    ["1342"],      # Iter 2: Pride & Prejudice
    ["345"],       # Iter 3: Dracula
    ["35"],        # Iter 4: Time Machine
    ["2701"],      # Iter 5: Moby Dick
    ["76"],        # Iter 6: Huck Finn
    ["98"],        # Iter 7: Tale of Two Cities
    ["844"],       # Iter 8: Being Earnest
    [],            # Iter 9: no new book
    [],            # Iter 10: no new book
]


def run_iteration(iteration, config, book_files=None):
    """Run a single training+eval iteration for one config.
    book_files: list of specific book paths to train on. If None, uses all corpus.
    """
    config_name = config["name"]
    print(f"\n  --- Config: {config_name} (embed={config['embed_dim']}, ctx={config['context_size']}, layers={config['n_layers']}, epochs={config['epochs']}) ---")

    # 1. Update hyperparams (use large batch for speed on CPU)
    batch_size = config.get("batch_size", 256)
    update_hyperparams(config["embed_dim"], config["context_size"], config["n_layers"], config["epochs"], batch_size=batch_size)

    # 2. Train
    print(f"  Training {config_name}...")
    t0 = time.time()
    cmd = [sys.executable, str(INGEST_PATH)]
    if book_files:
        cmd += ["--books"] + book_files
    # Timeout per config: large/target_small need more time on CPU (up to 2h)
    config_timeout = 7200 if config_name in ("large", "target_small") else 3600
    rc, stdout, stderr = run_cmd(cmd, timeout=config_timeout)
    elapsed = time.time() - t0

    if rc != 0:
        error_msg = stderr[-1000:] if stderr else stdout[-500:]
        print(f"  Training FAILED for {config_name}: {error_msg}")
        append_log({
            "type": "train_error",
            "iteration": iteration,
            "config": config_name,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        })
        return None

    loss, ponder = parse_training_output(stdout)
    print(f"  Training complete in {elapsed:.0f}s | loss={loss}, ponder={ponder}")

    # Print last few lines of training output for visibility
    last_lines = [l for l in stdout.splitlines() if l.strip()][-5:]
    for line in last_lines:
        print(f"    {line}")

    # 3. Evaluate
    print(f"  Evaluating {config_name}...")
    rc2, eval_stdout, eval_stderr = run_cmd(
        [sys.executable, str(EVAL_PATH), config_name, str(iteration)],
        timeout=600
    )

    eval_data = None
    if rc2 == 0:
        eval_data = parse_eval_output(eval_stdout)
    else:
        print(f"  Eval FAILED for {config_name}: {eval_stderr[:500]}")
        print(f"  Eval stdout: {eval_stdout[-500:]}")

    # Wait for ChromaDB SQLite to fully release its file lock on Windows
    time.sleep(5)

    # 4. Save checkpoint
    save_model_checkpoint(config_name, iteration)

    result = {
        "config": config_name,
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
    print("=" * 60)
    print("ENGRAM AUTONOMOUS TRAINING RUNNER")
    print("=" * 60)

    # Parse iteration override from args
    start_iteration = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Configure git identity if not set (needed for commits)
    try:
        rc, out, _ = run_cmd(["git", "config", "user.email"])
        if rc != 0 or not out.strip():
            run_cmd(["git", "config", "--global", "user.email", "engram-trainer@localhost"])
            run_cmd(["git", "config", "--global", "user.name", "Engram Trainer"])
    except Exception:
        pass

    # Load existing log to track loss history
    loss_history = {}  # config_name -> [losses]
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "iteration_complete":
                        for r in entry.get("config_results", []):
                            cn = r.get("config")
                            l = r.get("final_loss")
                            if cn and l is not None:
                                loss_history.setdefault(cn, []).append(l)
                except Exception:
                    pass

    consecutive_no_improve = {c["name"]: 0 for c in CONFIGS}

    for iteration in range(start_iteration, start_iteration + max_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration} / {start_iteration + max_iterations - 1}")
        print(f"{'='*60}")

        # Step 1: Download new book(s) if scheduled
        book_idx = iteration - 1
        new_books = []
        if book_idx < len(BOOK_SCHEDULE):
            scheduled = BOOK_SCHEDULE[book_idx]
            already_downloaded = list_corpus_books()
            for bid in scheduled:
                already = any(f.startswith(f"{bid}_") for f in already_downloaded)
                if not already:
                    new_books.append(bid)

        if new_books:
            print(f"\nStep 1: Downloading books {new_books}...")
            download_books(new_books)
        else:
            print(f"\nStep 1: No new books to download this iteration.")

        books_in_corpus = list_corpus_books()
        corpus_words = corpus_word_count()
        print(f"Corpus: {len(books_in_corpus)} books, ~{corpus_words:,} words")

        # Determine which books to use for this iteration's training
        # Strategy: train on the current iteration's new book + training_data.txt
        # This keeps each iteration fast (one book ~6-20 min) while sampling different text
        # On later iterations (>5), train on ALL accumulated books for final convergence
        corpus_dir = REPO_DIR / "corpus"
        if iteration > 5:
            # Final iterations: train on full corpus for convergence
            iter_book_files = None
            print("  Using full accumulated corpus for final convergence training.")
        else:
            # Early iterations: train on current book + base training data
            # Find the book downloaded for this iteration
            scheduled_ids = BOOK_SCHEDULE[book_idx] if book_idx < len(BOOK_SCHEDULE) else []
            iter_book_files = []
            if (REPO_DIR / "training_data.txt").exists():
                iter_book_files.append("training_data.txt")
            for bid in scheduled_ids:
                matching = [f for f in books_in_corpus if f.startswith(f"{bid}_")]
                if matching:
                    iter_book_files.append(str(corpus_dir / matching[0]))
            # If no specific books for this iteration, fall back to Frankenstein (first book)
            if not any(f.startswith(str(corpus_dir)) for f in iter_book_files):
                frankenstein = [f for f in books_in_corpus if "84_" in f]
                if frankenstein:
                    iter_book_files.append(str(corpus_dir / frankenstein[0]))
            print(f"  Training on: {iter_book_files}")

        # Step 2: Train all configs
        print(f"\nStep 2: Training {len(CONFIGS)} configurations...")
        iteration_results = []
        for config in CONFIGS:
            try:
                result = run_iteration(iteration, config, book_files=iter_book_files)
            except Exception as _exc:
                import traceback as _tb
                print(f"  EXCEPTION running config {config['name']}: {_exc}")
                _tb.print_exc()
                result = None
            if result:
                iteration_results.append(result)
                # Track loss history for stopping condition
                cn = config["name"]
                l = result.get("final_loss")
                if l is not None:
                    prev_losses = loss_history.get(cn, [])
                    if prev_losses and l >= min(prev_losses):
                        consecutive_no_improve[cn] = consecutive_no_improve.get(cn, 0) + 1
                    else:
                        consecutive_no_improve[cn] = 0
                    loss_history.setdefault(cn, []).append(l)

        # Step 3: Log iteration
        log_entry = {
            "type": "iteration_complete",
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "books_downloaded": new_books,
            "corpus_books": books_in_corpus,
            "corpus_words": corpus_words,
            "config_results": iteration_results,
        }
        append_log(log_entry)

        # Step 4: Print summary
        print(f"\n{'='*60}")
        print(f"=== ITERATION {iteration} COMPLETE ===")
        print(f"Books downloaded: {new_books if new_books else 'none'}")
        print(f"Corpus size: {corpus_words:,} words")
        print(f"\nConfig Results:")
        for r in iteration_results:
            loss_str = f"{r['final_loss']:.4f}" if r["final_loss"] is not None else "N/A"
            coh_str = f"{r['coherence_score']:.2f}" if r["coherence_score"] is not None else "N/A"
            vocab_str = f"{r['vocab_size']:,}" if r["vocab_size"] is not None else "N/A"
            print(f"  - {r['config']} ({r['embed_dim']}-dim): loss={loss_str}, coherence={coh_str}, vocab={vocab_str}")

        # Key insights
        if iteration_results:
            best = min(iteration_results, key=lambda x: x["final_loss"] if x["final_loss"] else 999)
            print(f"\nKey Insights:")
            print(f"  Best config this iteration: {best['config']} (loss={best.get('final_loss', 'N/A'):.4f})")
            best_coh = max(iteration_results, key=lambda x: x["coherence_score"] if x["coherence_score"] is not None else 0)
            print(f"  Best coherence: {best_coh['config']} ({best_coh.get('coherence_score', 0):.2f})")
        else:
            print("\nKey Insights: No configs succeeded this iteration.")

        # Next step guidance
        all_books = len(list_corpus_books())
        all_configs_done = len(iteration_results) == len(CONFIGS)
        if iteration >= 5:
            print(f"\nNext Step: Analyzing trends - {iteration} iterations complete, assessing convergence.")
        else:
            next_scheduled = BOOK_SCHEDULE[iteration] if iteration < len(BOOK_SCHEDULE) else []
            print(f"\nNext Step: Iteration {iteration+1} - downloading {next_scheduled or 'no new book'}, training all configs.")

        print(f"Progress: {iteration}/{start_iteration + max_iterations - 1} iterations")

        # Stopping condition check
        plateau_configs = [cn for cn, count in consecutive_no_improve.items() if count >= 3]
        if plateau_configs:
            print(f"\nWARNING: Loss plateau detected for: {plateau_configs} (3+ iterations no improvement)")

        # Git commit
        git_commit(f"Iteration {iteration}: corpus={corpus_words} words, {len(iteration_results)} configs trained")

        # Notify web server to reload the latest model weights
        try:
            import urllib.request
            req = urllib.request.Request(
                "http://localhost:5000/reload",
                method="POST",
                headers={"Content-Type": "application/json"},
                data=b"{}",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                reload_data = json.loads(resp.read())
                print(f"  Web server reloaded: vocab={reload_data.get('vocab_size')}, iters={reload_data.get('iterations')}")
        except Exception as e:
            print(f"  Web server reload skipped (server may not be running): {e}")

        # Check stopping conditions
        if all_books >= 10 and all_configs_done and iteration >= 5:
            print(f"\nSTOPPING: All 10 books downloaded AND all configs tested AND >=5 iterations complete.")
            break

        if iteration >= start_iteration + max_iterations - 1:
            print(f"\nSTOPPING: Reached max iterations ({max_iterations}).")
            break

    # Final analysis
    print(f"\n{'='*60}")
    print("FINAL ANALYSIS")
    print(f"{'='*60}")
    if LOG_FILE.exists():
        all_entries = []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_entries.append(json.loads(line))
                except Exception:
                    pass
        iters = [e for e in all_entries if e.get("type") == "iteration_complete"]
        if iters:
            last = iters[-1]
            print(f"Total iterations logged: {len(iters)}")
            print(f"Final corpus: {last.get('corpus_words', 0):,} words, {len(last.get('corpus_books', []))} books")
            print(f"\nFinal config performance:")
            for r in last.get("config_results", []):
                loss_str = f"{r['final_loss']:.4f}" if r.get("final_loss") else "N/A"
                coh_str = f"{r['coherence_score']:.2f}" if r.get("coherence_score") is not None else "N/A"
                print(f"  {r['config']}: loss={loss_str}, coherence={coh_str}")

    # Restore ingest.py to baseline config
    update_hyperparams(64, 8, 3, 5, batch_size=64)
    print("\nRestored ingest.py to baseline config.")
    git_commit("Final: restored baseline hyperparams after training run")


if __name__ == "__main__":
    main()
