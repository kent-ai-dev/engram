#!/usr/bin/env python3
"""Download TinyStories dataset for Engram training.

TinyStories (Eldan & Li, 2023) is purpose-built for small language models.
~476M tokens of simple, coherent children's stories using ~3000-word vocabulary.

Downloads the validation split (smaller, good for initial testing) and
optionally the full training split.

Usage:
    python3 download_tinystories.py           # validation only (~27MB)
    python3 download_tinystories.py --full    # full training set (~2.2GB)
"""

import argparse
import os
import sys

def download():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Download full training set (2.2GB)")
    parser.add_argument("--output-dir", default="corpus", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    print("Downloading TinyStories from HuggingFace...")
    ds = load_dataset("roneneldan/TinyStories")

    # Always save validation split (small, good for testing)
    val_path = os.path.join(args.output_dir, "tinystories_val.txt")
    print(f"Writing validation split to {val_path}...")
    with open(val_path, "w", encoding="utf-8") as f:
        for item in ds["validation"]:
            text = item.get("text", "")
            if text.strip():
                f.write(text.strip() + "\n\n")
    val_size = os.path.getsize(val_path) / (1024 * 1024)
    print(f"  Validation: {val_size:.1f} MB")

    if args.full:
        train_path = os.path.join(args.output_dir, "tinystories_train.txt")
        print(f"Writing training split to {train_path}...")
        with open(train_path, "w", encoding="utf-8") as f:
            count = 0
            for item in ds["train"]:
                text = item.get("text", "")
                if text.strip():
                    f.write(text.strip() + "\n\n")
                    count += 1
                    if count % 100000 == 0:
                        print(f"  {count:,} stories written...")
        train_size = os.path.getsize(train_path) / (1024 * 1024)
        print(f"  Training: {train_size:.1f} MB ({count:,} stories)")

    print("\nDone! Files are in the corpus/ directory.")
    print("Run ingest.py to train on them.")


if __name__ == "__main__":
    download()
