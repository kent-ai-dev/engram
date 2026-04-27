"""
corpus_clean.py — clean dailydialog.txt of preprocessing artifacts that
contaminated v7's vocab (numeric IDs, mixed alphanumeric tokens, rare
proper nouns that show up only once or twice).

Strategy:
  1. Lowercase everything.
  2. Replace any run of digits with a single <NUM> placeholder. (Keep
     punctuation around it.)
  3. Strip tokens that contain digits OR underscores after step 2 (these
     are placeholders like "yw132", "100RMB" that survive normalization).
  4. Replace single-occurrence rare proper nouns with <UNK> after a vocab
     pass (token frequency < 3).
  5. Write to corpus/dailydialog_clean.txt.

Run from repo root:
    uv run corpus_clean.py
"""

import os
import re
from collections import Counter

INPUT = "corpus/dailydialog.txt"
OUTPUT = "corpus/dailydialog_clean.txt"
MIN_FREQ = 3   # tokens appearing fewer than this become <UNK>


def normalize_line(line: str) -> str:
    """Lowercase, replace digit runs with <num>, drop alphanum-mixed garbage."""
    line = line.lower()
    # Replace any digit-run with a single token <num>
    line = re.sub(r'\d+', '<num>', line)
    # Tokenize and drop any token that has digits or underscores left,
    # OR that contains non-ascii (eg Chinese chars from the corpus).
    out = []
    for tok in re.findall(r'<num>|[^\w\s<>]|[a-z]+', line):
        if tok in ('<num>',):
            out.append(tok)
        elif re.match(r'^[a-z]+$', tok):
            out.append(tok)
        elif tok in '.,!?;:\'"-':
            out.append(tok)
        # else: drop
    return ' '.join(out)


def main():
    if not os.path.exists(INPUT):
        raise SystemExit(f"missing {INPUT}")

    # Pass 1: read + normalize, count token frequencies
    cleaned_lines = []
    counter = Counter()
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            # Preserve structural blank lines
            if not stripped:
                cleaned_lines.append("")
                continue
            # Preserve speaker-tag prefixes from dailydialog ("User:" / "Bot:")
            m = re.match(r'^(User|Bot):\s*(.*)$', stripped, re.IGNORECASE)
            if m:
                tag = m.group(1).capitalize() + ":"
                body = normalize_line(m.group(2))
                line_clean = f"{tag} {body}"
            else:
                line_clean = normalize_line(stripped)
            cleaned_lines.append(line_clean)
            for tok in line_clean.split():
                counter[tok] += 1

    # Pass 2: replace very-rare tokens with <unk>
    rare = {tok for tok, c in counter.items() if c < MIN_FREQ and tok not in ('User:', 'Bot:', '<num>')}
    final_lines = []
    for line in cleaned_lines:
        if not line:
            final_lines.append("")
            continue
        toks = []
        for tok in line.split():
            toks.append("<unk>" if tok in rare else tok)
        final_lines.append(" ".join(toks))

    # Stats
    total_tokens = sum(counter.values())
    unique_before = len(counter)
    unique_after = len(counter) - len(rare)
    print(f"  total tokens (before rare-merge): {total_tokens:,}")
    print(f"  unique tokens (before rare-merge): {unique_before:,}")
    print(f"  unique tokens (after merging rare<{MIN_FREQ}): {unique_after:,}")
    print(f"  rare tokens merged into <unk>: {len(rare):,}")
    # Sample of rare tokens for visual inspection
    rare_sample = list(rare)[:20]
    print(f"  rare-token sample: {rare_sample}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(final_lines))
    out_size = os.path.getsize(OUTPUT)
    print(f"\n  wrote {OUTPUT} ({out_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
