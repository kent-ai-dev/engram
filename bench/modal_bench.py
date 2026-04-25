"""
bench/modal_bench.py — Modal app for running bench/run.py on L4 GPU.

Usage (from repo root):
  python bench/run.py [args] --via-modal          # launch + wait + download result
  modal run bench/modal_bench.py -- '{"seed":42}'  # direct Modal CLI

The local --via-modal flow:
  1. Serialise all bench args to JSON
  2. Upload JSON to Modal volume as bench_config_<run_id>.json
  3. Call run_bench.remote() which clones the repo and executes bench/run.py
  4. Pull result JSON back to bench/history/<run_id>.json
"""

import json
import os
import sys
import io
from pathlib import Path

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

VOLUME_NAME = "engram-weights"
VOLUME_MOUNT = "/engram-weights"

if MODAL_AVAILABLE:
    app = modal.App("engram-bench")

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    image = (
        modal.Image.from_registry(
            "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
            add_python="3.12",
        )
        .pip_install("chromadb", "numpy")
        .apt_install("git")
        .run_commands("echo 'bench-image-v1'")
    )

    @app.function(
        image=image,
        gpu="L4",
        timeout=3600,
        volumes={VOLUME_MOUNT: volume},
        retries=0,
    )
    def run_bench(config_json: str) -> str:
        """
        Run bench/run.py inside a Modal L4 container.
        Returns the result JSON as a string.
        """
        import subprocess, shutil, sys, json, os, tempfile

        config = json.loads(config_json)
        run_id = config.get("run_id", "modal_bench")

        print(f"[modal_bench] run_id={run_id}")

        # Clone repo
        repo_dir = "/tmp/engram"
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        result = subprocess.run(
            ["git", "clone", "https://github.com/kent-ai-dev/engram.git", repo_dir],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr}")

        # Build bench/run.py CLI args from config
        args = [sys.executable, "bench/run.py"]
        args += ["--seed", str(config.get("seed", 42))]
        args += ["--epochs", str(config.get("epochs", 2))]
        args += ["--batch-size", str(config.get("batch_size", 128))]
        args += ["--context-size", str(config.get("context_size", 32))]
        args += ["--n-ponder-train", str(config.get("n_ponder_train", 3))]
        if config.get("n_ponder_eval"):
            args += ["--n-ponder-eval", str(config["n_ponder_eval"])]
        args += ["--run-id", run_id]
        args += ["--output-dir", "bench/history"]
        if config.get("corpus"):
            args += ["--corpus"] + config["corpus"]
        if config.get("holdout"):
            args += ["--holdout"] + config["holdout"]
        if config.get("use_lti"):
            args.append("--use-lti")
        if config.get("use_loop_idx"):
            args.append("--use-loop-idx")
        if config.get("use_lora"):
            args.append("--use-lora")
        if config.get("use_rope"):
            args.append("--use-rope")

        print(f"[modal_bench] Running: {' '.join(args)}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for deterministic CUDA

        result = subprocess.run(args, cwd=repo_dir, text=True, timeout=3500, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"bench/run.py failed rc={result.returncode}")

        # Read and return result JSON
        out_path = os.path.join(repo_dir, "bench", "history", f"{run_id}.json")
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Result not found at {out_path}")

        with open(out_path) as f:
            result_json = f.read()

        # Also copy to volume for persistence
        vol_path = os.path.join(VOLUME_MOUNT, "bench_results", f"{run_id}.json")
        os.makedirs(os.path.dirname(vol_path), exist_ok=True)
        with open(vol_path, "w") as f:
            f.write(result_json)
        volume.commit()
        print(f"[modal_bench] Result written to volume: {vol_path}")

        return result_json

    @app.local_entrypoint()
    def main(config_json: str = "{}"):
        result = run_bench.remote(config_json)
        print(result)


def launch_via_modal(args) -> str:
    """
    Called by bench/run.py when --via-modal is set.
    Runs bench/modal_bench.py via `modal run` subprocess (handles app hydration).
    Returns path to the local result JSON file.
    """
    import subprocess, tempfile

    config = {
        "run_id": args.run_id,
        "seed": args.seed,
        "corpus": args.corpus,
        "holdout": args.holdout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "context_size": args.context_size,
        "n_ponder_train": args.n_ponder_train,
        "n_ponder_eval": args.n_ponder_eval,
        "use_lti": args.use_lti,
        "use_loop_idx": args.use_loop_idx,
        "use_lora": args.use_lora,
        "use_rope": args.use_rope,
    }

    config_json = json.dumps(config)
    this_file = os.path.abspath(__file__)

    print(f"[bench] Launching run_id={args.run_id} on Modal L4 GPU...")
    print(f"[bench] Running: modal run {this_file} -- <config>")

    # `modal run` calls the @app.local_entrypoint() which calls run_bench.remote()
    # Modal local entrypoint params are passed as --param-name value (kebab-case).
    result = subprocess.run(
        [sys.executable, "-m", "modal", "run", this_file, "--config-json", config_json],
        capture_output=False,   # stream output so user sees progress
        text=True,
        timeout=4000,
    )
    if result.returncode != 0:
        raise RuntimeError(f"modal run failed with exit code {result.returncode}")

    # The Modal function also writes the result to the volume under bench_results/.
    # Pull it back via modal volume get.
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.run_id}.json")

    dl = subprocess.run(
        [sys.executable, "-m", "modal", "volume", "get",
         "engram-weights", f"bench_results/{args.run_id}.json", out_path],
        capture_output=True, text=True, timeout=60,
    )
    if dl.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"Could not download result from Modal volume: {dl.stderr}")

    with open(out_path) as f:
        data = json.load(f)
    print(f"[bench] Modal run complete. eval_cosine_top1={data.get('eval_cosine_top1', 'N/A'):.2f}%")
    print(f"[bench] wrote {out_path}")
    return out_path
