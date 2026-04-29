# v11 Plan — Corpus Expansion via smol-smoltalk

**Status:** queued. Stage everything, fire only if v10 fails to crack coherence.

**Date queued:** 2026-04-28

## Hypothesis being tested

> Coherence at this scale is a function of conversational corpus volume, not model capacity.

v9 (21.5M params, dailydialog_clean only) refuted the capacity hypothesis — bigger model on the same corpus produced the same word-salad. v10 (training now) tests whether 25% more corpus (everyday-conversations) helps. v11 escalates to ~3× more corpus by adding a smol-smoltalk subsample. If v11 still fails, the bottleneck is something we haven't identified — at that point stop scaling and re-examine the architecture.

## Lessons from v1 → v9 baked into this plan

| Source | Lesson | How v11 honors it |
|---|---|---|
| v1-v4 | Gutenberg novels dominate output vocab → archaic gibberish. | No fiction corpus. Three conversational sources only. |
| v5 → v5b | Post-LN + AdamW collapses; gradient explodes mid-epoch. | Pre-LN architecture (locked in v6). Already in `engram_model.py`. |
| v5/Phase 5 | RoPE halves grad-norm p99 vs sinusoidal positional embeddings. | RoPE locked default; head_dim=32 (embed_dim=384, n_heads=12). |
| v6 | Architecture fix alone doesn't fix data quality. | Don't touch architecture in v11; only corpus changes. |
| v7 | Even good HF datasets need cleanup (numeric IDs, alphanumeric garbage like `yw132`, `100RMB`). | `download_smol_smoltalk.py` applies the same `corpus_clean.normalize_line` regex pipeline. |
| v7 | Long instruction chains over the 32-token context waste data. | Filter to ≤6-turn conversations on download (smol-smoltalk has many ChatGPT-style instruction chains). |
| v8 | Cleaned dailydialog → final loss 1.0044 (best yet); all real English. | Keep cleaned dailydialog as one of three corpora. |
| v8 | Vocab shrinks dramatically post-cleanup (38k → 9.5k). Smaller vocab = sharper gradients. | Apply rare-token < 3 → `<unk>` merging on the smol-smoltalk subset before concat. |
| v9 | Capacity is fine at 21.5M params — output forms fragments (`i am`, `the bot`, `i have a really`) but no syntax. | Hold v9's exact arch (384D / 12L / 12H / RoPE / Pre-LN). Don't change brain shape. |
| v9 | `avg_ponder=3.0` constant — halt gate not learning to terminate early. | Untouched in v11; halt gate behavior is its own follow-up, not a coherence blocker. |
| v10 | New corpus files must be in `origin/main` BEFORE Modal launch — Modal clones from GitHub and `ingest.py` silently drops missing files. v10 was launched with `everyday_conversations.txt` only present locally; it got dropped and v10 trained on dailydialog only (≡ v9 with a different seed). | Pre-flight step 4 below explicitly verifies `git ls-tree HEAD corpus/` includes every file in the BOOKS list. |

## Architecture (frozen from v9)

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

Three conversational sources, blended:

| File | Source | Cleaned size | Origin |
|---|---|---|---|
| `corpus/dailydialog_clean.txt` | DailyDialog (HF) | ~3.6 MB | Already cleaned via `corpus_clean.py` |
| `corpus/everyday_conversations.txt` | `HuggingFaceTB/everyday-conversations-llama3.1-2k` | ~1.7 MB | Llama-3.1-70B-generated greetings + assistant identity + 20 topics |
| `corpus/smol_smoltalk_subset.txt` | `HuggingFaceTB/smol-smoltalk` (subsampled) | ~10 MB | SmolLM2 training subset — proven sufficient for grammar at the 135M-param scale |

**Combined: ~15.3 MB.** Estimated combined vocab after rare-merge: 12,000-15,000 unique tokens.

## Training config (`configs/engram_v11_config.json`)

```json
{
  "EMBED_DIM": 384, "N_LAYERS": 12, "CONTEXT_SIZE": 32, "N_HEADS": 12,
  "BATCH_SIZE": 96, "BRAIN_LR": 1e-3, "EMBED_LR": 5e-4,
  "EPOCHS": 4,
  "NGRAM_TABLE_SIZE": 100003,
  "BOOKS": ["corpus/dailydialog_clean.txt",
            "corpus/everyday_conversations.txt",
            "corpus/smol_smoltalk_subset.txt"]
}
```

**Why these dial values differ from v9:**
- `EPOCHS: 5 → 4` — combined corpus is ~3× v9's, so 4 epochs at the new size is more total tokens than 5 epochs at v9's size. Keeps wall-time under L4's 5h timeout.
- `NGRAM_TABLE_SIZE: 50021 → 100003` — vocab will exceed 9,509. The N-gram memory module hashes triples into the table; 50k buckets at 12k vocab = high collision rate. 100k buckets restores headroom. (Both are primes; required by the modular hash in `EngramModule`.)

## Wall-time math

```
~15.3 MB corpus → ~2.7M sequences (vs v9's 0.65M)
At BATCH_SIZE=96 → ~28k batches/epoch (vs v9's ~6.5k)
4 epochs → ~112k batches
At v9's measured ~14 batches/sec on L4 → ~8000s ≈ 2.2h
```

Comfortable under the 18000s (5h) Modal timeout. **Estimated cost ~$2-3.**

## Pre-flight checklist (do before launching)

1. **Prerequisite:** v10 has finished and its eval transcript is recorded. If v10 passes coherence (3/5 chitchat replies coherent) → STOP, ship v10, archive this plan.
2. Run `python3 download_smol_smoltalk.py` to materialize `corpus/smol_smoltalk_subset.txt` (~5-10 min, streams from HF — no full 971 MB local download).
3. Quick visual sanity-check: `head -20 corpus/smol_smoltalk_subset.txt` should show `User:` / `Bot:` lines with only lowercase letters, `<num>`, `<unk>`, and basic punctuation. No mixed-alphanumeric tokens.
4. **CRITICAL — commit corpus files to origin/main BEFORE launching.** Modal training clones the repo from GitHub; any corpus file that's only local will be silently dropped by `ingest.py`'s existence check (`corpus_files = [b for b in args.books if os.path.exists(b)]`). v10 hit this bug — it was launched with `BOOKS=[dailydialog_clean.txt, everyday_conversations.txt]` but the second file wasn't in HEAD, so v10 trained on dailydialog only and was effectively a v9 re-run with a different seed. Verify with `git ls-tree HEAD corpus/` before launching.
5. Upload config: `modal volume put engram-weights configs/engram_v11_config.json /engram_v11_config.json --force`
6. Launch: `python3 -m modal run --detach ~/.openclaw/workspace/scripts/modal_train.py --config-path configs/engram_v11_config.json`

## Eval criteria (unchanged from the loop)

A v11 run is considered successful and ships only if `eval_chat.py` shows all three:
1. **Different replies for different prompts** (16/16) — ✅ already passing since v6.
2. **≥80% real-English tokens** — should pass given cleaned corpora.
3. **≥3 of 5 chitchat prompts produce coherent dialog-shaped replies** — the only criterion that's failed v6 → v9. Coherent here means recognizable subject-verb-object or greeting/response pattern, not perfect grammar.

If v11 passes all three: emit `<promise>ENGRAM_COHERENT</promise>`, ship to live server, archive the autonomous loop.

If v11 fails criterion 3: **stop scaling corpus**. The next investigation should question the loss function (MSE on embeddings may be the wrong objective for grammar) or the generation strategy (top-k=10 with cosine penalty may be flattening the distribution). This plan does not pre-commit to a v12.

## Cost ceiling

Loop's hard kill is cumulative spend > $50. Current spend across v5-v10 ≈ $25 (v5 $3.60, v6 $3, v7 $3, v8 $3, v9 $4, v10 $4). v11 at $2-3 keeps us under $30 — well below the ceiling.

## Files staged

- `download_smol_smoltalk.py` — streaming downloader with cleanup, ready to run.
- `configs/engram_v11_config.json` — Modal training config, not yet uploaded.
- `plans/V11_PLAN.md` — this file.

## What's intentionally NOT changed

- Architecture (per the v9-capacity-refuted lesson — don't fix what isn't broken).
- Generation parameters (`TEMPERATURE=0.9`, `TOP_K=10`, `GEN_STEPS=20`).
- The N-gram memory module (`EngramModule`) — keeping it on for continuity with v9; ablating it deserves a separate experiment.
- Adaptive pondering (`max_ponder=3`) — the halt gate's `avg_ponder=3.0` constant behavior is a known issue but orthogonal to coherence.
