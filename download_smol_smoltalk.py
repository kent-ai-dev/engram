"""
download_smol_smoltalk.py — fetch HuggingFaceTB/smol-smoltalk (the curated
SmolLM2 training subset, 485k rows / 971 MB), subsample, convert to
engram's User:/Bot: format, and apply the same cleanup as corpus_clean.py.

The full dataset is too big for engram's L4 budget. We subsample by
character budget (default ~10 MB) targeting short conversational turns
that match dailydialog's style. Long instruction-tuning chains are
filtered out — engram's context window is 32, so multi-turn chains over
~6 messages waste data anyway.

Run from repo root:
    python3 download_smol_smoltalk.py
"""

import os
import re
import random
from collections import Counter

OUTPUT = "corpus/smol_smoltalk_subset.txt"
TARGET_BYTES = 10 * 1024 * 1024  # ~10 MB total
MAX_TURNS_PER_CONVO = 6          # drop long instruction chains
MIN_FREQ = 3                     # rare-token <unk> threshold (matches corpus_clean.py)
SEED = 42


def normalize_line(text: str) -> str:
    """Same cleanup as corpus_clean.normalize_line: lowercase, <num>, drop garbage."""
    text = text.lower()
    text = re.sub(r"\d+", "<num>", text)
    out = []
    for tok in re.findall(r"<num>|[^\w\s<>]|[a-z]+", text):
        if tok == "<num>" or re.match(r"^[a-z]+$", tok) or tok in ".,!?;:'\"-":
            out.append(tok)
    return " ".join(out)


def main():
    from datasets import load_dataset

    random.seed(SEED)

    print("Loading HuggingFaceTB/smol-smoltalk (train split, streaming)…")
    # Stream to avoid 971 MB local download — we only consume until the byte budget.
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True)

    raw_lines: list[str] = []
    n_convos = 0
    n_dropped_long = 0
    bytes_written = 0

    for row in ds:
        messages = row.get("messages", [])
        if not messages or len(messages) > MAX_TURNS_PER_CONVO * 2:
            n_dropped_long += 1
            continue

        convo_lines: list[str] = []
        ok = True
        for msg in messages:
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if not content:
                ok = False
                break
            tag = "User:" if role == "user" else "Bot:"
            cleaned = normalize_line(content)
            if not cleaned or len(cleaned) > 600:
                ok = False
                break
            convo_lines.append(f"{tag} {cleaned}")

        if not ok or not convo_lines:
            continue

        block = "\n".join(convo_lines) + "\n\n"
        if bytes_written + len(block.encode("utf-8")) > TARGET_BYTES:
            break
        raw_lines.append(block)
        bytes_written += len(block.encode("utf-8"))
        n_convos += 1

    print(f"  collected {n_convos:,} conversations ({bytes_written / 1024 / 1024:.1f} MB)")
    print(f"  dropped {n_dropped_long:,} long/multi-turn conversations")

    # Pass 1: token-frequency count for rare-merge
    counter: Counter = Counter()
    for block in raw_lines:
        for line in block.splitlines():
            for tok in line.split():
                if tok in ("User:", "Bot:"):
                    continue
                counter[tok] += 1

    rare = {t for t, c in counter.items() if c < MIN_FREQ and t not in ("<num>",)}

    # Pass 2: emit with rare tokens merged into <unk>
    final_lines: list[str] = []
    for block in raw_lines:
        for line in block.splitlines():
            if not line.strip():
                final_lines.append("")
                continue
            toks = line.split()
            new_toks: list[str] = []
            for tok in toks:
                if tok in ("User:", "Bot:"):
                    new_toks.append(tok)
                elif tok in rare:
                    new_toks.append("<unk>")
                else:
                    new_toks.append(tok)
            final_lines.append(" ".join(new_toks))
        final_lines.append("")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(final_lines))

    out_size = os.path.getsize(OUTPUT)
    print(
        f"  wrote {OUTPUT}  ({out_size / 1024 / 1024:.1f} MB, "
        f"{n_convos:,} convos, {len(counter) - len(rare):,} unique tokens after rare-merge, "
        f"{len(rare):,} merged into <unk>)"
    )


if __name__ == "__main__":
    main()
