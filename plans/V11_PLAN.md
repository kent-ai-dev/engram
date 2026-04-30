# v11 Plan — The Corpus-Expansion Test v10 Was Meant To Run

**Status:** active. Launching now.

**Date queued:** 2026-04-28
**Date revised:** 2026-04-30 (v11 redirected after the v10 corpus-bug finding and rejected smol-smoltalk/MultivexAI alternatives)

## Hypothesis being tested

> Coherence at this scale is a function of conversational corpus volume, not model capacity.

v9 (21.5M params, dailydialog_clean only) refuted the capacity hypothesis — bigger model on the same corpus produced the same word-salad. v10 was supposed to test the corpus side but trained on dailydialog only because `corpus/everyday_conversations.txt` was uncommitted at launch and `ingest.py` silently dropped it (see V10 lesson below). **v11 is the experiment v10 was meant to run** — same v10 architecture, two committed conversational corpora, clean test of the corpus-volume hypothesis.

## What changed since the original v11 plan

The original v11 plan (2026-04-28) added a `HuggingFaceTB/smol-smoltalk` subsample as a third corpus. We rejected that after inspecting the data:

- **smol-smoltalk is instruction-tuning data, not natural dialog.** Sample turns: *"your response should contain less than `<num>` words"*, *"create a python function that accepts a list of integers"*, *"rewrite the input text to make it more professional"*. ~50% of "Bot:" turns are actually system prompts (mapped wrong by my downloader). Wrong distribution for conversational coherence.
- **MultivexAI/Everyday-Language-Corpus** was the alternative — turned out to be 8,788 single-sentence utterances (`[S]I'm going to make a cup of tea.[E]`), not multi-turn dialog. After parsing into User/Bot pairs, only ~30 KB usable. Too small to move the needle.

Conclusion: scaling corpus *quantity* before validating that the *quality-controlled* expansion (dailydialog + everyday-conversations) actually moves coherence is premature. v11 keeps the experiment small and clean.

## Lessons from v1 → v10 baked into this plan

| Source | Lesson | How v11 honors it |
|---|---|---|
| v1-v4 | Gutenberg novels dominate output vocab → archaic gibberish. | No fiction corpus. Two conversational sources only. |
| v5 → v5b | Post-LN + AdamW collapses; gradient explodes mid-epoch. | Pre-LN architecture (locked in v6). |
| v5/Phase 5 | RoPE halves grad-norm p99 vs sinusoidal positional embeddings. | RoPE locked default; head_dim=32 (embed_dim=384, n_heads=12). |
| v6 | Architecture fix alone doesn't fix data quality. | Don't touch architecture in v11; only corpus changes. |
| v7 | Even good HF datasets need cleanup (numeric IDs, alphanumeric garbage like `yw132`, `100RMB`). | Both corpus files are pre-cleaned by `corpus_clean.py` / `download_everyday_conversations.py`. |
| v8 | Cleaned dailydialog → final loss 1.0044 (best yet); all real English. | Reuse cleaned dailydialog (`corpus/dailydialog_clean.txt`). |
| v8 | Vocab shrinks dramatically post-cleanup (38k → 9.5k). Smaller vocab = sharper gradients. | Combined two-corpus vocab estimated 10-12k after rare-merge — modest growth. |
| v9 | Capacity is fine at 21.5M params — output forms fragments (`i am`, `the bot`, `i have a really`) but no syntax. | Hold v9's exact arch (384D / 12L / 12H / RoPE / Pre-LN). Don't change brain shape. |
| v9 | `avg_ponder=3.0` constant — halt gate not learning to terminate early. | Untouched in v11; halt gate behavior is its own follow-up. |
| v10 | New corpus files must be in `origin/main` BEFORE Modal launch — Modal clones from GitHub and `ingest.py` silently drops missing files. v10 was launched with `everyday_conversations.txt` only present locally; it got dropped and v10 trained on dailydialog only (≡ v9 with a different seed). | Pre-flight step 4 below explicitly verifies `git ls-tree HEAD corpus/` includes every file in the BOOKS list. **Both files confirmed in HEAD as of commits 7f5ae9c (dailydialog_clean) and 99f6443 (everyday_conversations).** |
| v10 | Different SmolLM-style training datasets aren't drop-in for conversational coherence. | Reject smol-smoltalk-style instruction-tuning sources. Only natural dialog. |

## Architecture (frozen from v9/v10)

```
embed_dim:       384
n_layers:        12
n_heads:         12       (head_dim = 384 / 12 = 32)
context_size:    32
max_ponder:      3
positional:      RoPE (use_rope=True, max_seq_len=128)
norm:            Pre-LN
optimizer:       AdamW, weight_decay=0.01
brain_params:    21,509,761
```

## Corpus

Two conversational sources, both already in `origin/main`:

| File | Source | Cleaned size | In HEAD |
|---|---|---|---|
| `corpus/dailydialog_clean.txt` | DailyDialog (HF) | ~3.6 MB | ✅ since 922a6fc |
| `corpus/everyday_conversations.txt` | `HuggingFaceTB/everyday-conversations-llama3.1-2k` | ~1.7 MB | ✅ since 99f6443 |

**Combined: ~5.3 MB.** That's 50% bigger than v9's corpus. Estimated combined vocab after rare-merge: 10,000-12,000 tokens.

## Training config (`configs/engram_v11_config.json`)

```json
{
  "EMBED_DIM": 384, "N_LAYERS": 12, "CONTEXT_SIZE": 32, "N_HEADS": 12,
  "BATCH_SIZE": 96, "BRAIN_LR": 1e-3, "EMBED_LR": 5e-4,
  "EPOCHS": 5,
  "NGRAM_TABLE_SIZE": 100003,
  "BOOKS": ["corpus/dailydialog_clean.txt", "corpus/everyday_conversations.txt"]
}
```

**Why these dial values differ from v9:**
- `EPOCHS: 5` — same as v9. Combined corpus ~5.3 MB still fits 5h L4 timeout comfortably.
- `NGRAM_TABLE_SIZE: 50021 → 100003` — vocab will exceed 9,509. The N-gram memory module hashes triples into the table; 50k buckets at 12k vocab = high collision rate. 100k buckets restores headroom. (Both values are primes; required by the modular hash in `EngramModule`.)

## Wall-time math

```
~5.3 MB corpus → ~950k sequences (vs v9's 0.65M)
At BATCH_SIZE=96 → ~9.9k batches/epoch
5 epochs → ~49.5k batches
At v9's measured ~14 batches/sec on L4 → ~3500s ≈ 1h
```

Comfortable under the 18000s (5h) Modal timeout. **Estimated cost ~$1-2.**

## Pre-flight checklist (do before launching)

1. ✅ **Both BOOKS files in HEAD on origin/main.** Verify with:
   ```
   git ls-tree HEAD corpus/dailydialog_clean.txt corpus/everyday_conversations.txt
   ```
   Both blobs must list. (Confirmed 2026-04-30.)
2. ✅ **Origin is up to date.** `git log origin/main..HEAD` must be empty (Confirmed 2026-04-30.)
3. Upload config: `modal volume put engram-weights configs/engram_v11_config.json /engram_v11_config.json --force`
4. Launch: `python3 -m modal run --detach ~/.openclaw/workspace/scripts/modal_train.py --config-path configs/engram_v11_config.json`

## Eval criteria (unchanged from the loop)

1. **Different replies for different prompts** (16/16) — ✅ already passing since v6.
2. **≥80% real-English tokens** — should pass given cleaned corpora.
3. **≥3 of 5 chitchat prompts produce coherent dialog-shaped replies** — the only criterion that's failed v6 → v10. Coherent = recognizable subject-verb-object or greeting/response, not perfect grammar.

If v11 passes all three: emit `<promise>ENGRAM_COHERENT</promise>`, ship to live server, archive the autonomous loop.

If v11 fails criterion 3: the corpus-volume hypothesis is questionable. Next investigation should question:
- The loss function (MSE on embeddings may be the wrong objective for grammar)
- The generation strategy (top-k=10 with cosine penalty may flatten the distribution)
- The N-gram memory module (might be smearing predictions toward token-frequency mean)

This plan does not pre-commit to v12 corpus expansion. Diagnose first.

## Cost ceiling

Loop's hard kill is cumulative spend > $50. Current spend across v5-v10 ≈ $25 (v5 $3.60, v6 $3, v7 $3, v8 $3, v9 $4, v10 $4). v11 at $1-2 keeps us under $30 — well below the ceiling.

## Files involved

- `corpus/dailydialog_clean.txt` (committed since 922a6fc)
- `corpus/everyday_conversations.txt` (committed since 99f6443)
- `configs/engram_v11_config.json` (this plan revision)
- `plans/V11_PLAN.md` (this file)

## What's intentionally NOT changed

- Architecture (per the v9-capacity-refuted lesson — don't fix what isn't broken).
- Generation parameters (`TEMPERATURE=0.9`, `TOP_K=10`, `GEN_STEPS=20`).
- The N-gram memory module (`EngramModule`) — keeping it on for continuity with v9/v10.
- Adaptive pondering (`max_ponder=3`) — known stuck at 3.0 but orthogonal to coherence.
- Vocabulary cleanup pipeline (`corpus_clean.py`-style normalize_line).
