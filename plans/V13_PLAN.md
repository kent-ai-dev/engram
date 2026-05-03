# v13 Plan — Sharpen the Softmax (INV_TEMP 10 → 30)

**Status:** staged in code, awaiting Modal launch.
**Date queued:** 2026-05-03 (immediately after v12 deploy)

## Hypothesis

> v12's cross-entropy loss is functioning correctly, but at INV_TEMPERATURE=10 the softmax over the vocab is too flat for the model to commit to a single token — leaving 5.6 nats of headroom unused. Sharpening the temperature gives the same loss function strictly more gradient bite per wrong prediction.

## Evidence motivating this hypothesis (from v12)

| Epoch | xent loss |
|-------|-----------|
| 1 | 8.16 |
| 2 | 8.00 |
| 3 | 7.74 |
| 4 | 7.54 |
| 5 | 7.38 |

- Loss decreases cleanly ~−0.2 nats/epoch — no instability, no collapse, no NaN. Cross-entropy training is working.
- **Plateau at 7.38 vs floor of ~1.77 at INV_TEMP=10** = 5.6 nats unused. The model is not converging because the per-token gradient is too small once predictions land in the right neighborhood.
- Eval transcript (`eval_runs/chat_20260503_165046.json`): replies are distinct per prompt and contain dialog scaffolding (USER/BOT, "i'm not going", "i ll go to") but no coherent sentences. Suggests the model has learned distributional shape but can't discriminate within it.

## Math: why 30 (and what the floor becomes)

For an L2-normalized prediction perfectly aligned with the correct token (`cos = 1.0`) and the next-best wrong token at `cos ≈ 0.7`:

| INV_TEMP | logit gap | softmax(correct) | xent loss |
|----------|-----------|------------------|-----------|
| 10 | 3.0 | ~0.95 | ~0.05 |
| 30 | 9.0 | ~0.9999 | ~0.0001 |

But the *measured* floor is dominated by the entropy of the actual cosine distribution across V=14,704 tokens (most are between 0 and 0.5). Empirically:

- INV_TEMP=10 floor ≈ 1.77 nats (per the v12 model_card, derived from a uniform-ish softmax over the cosine distribution)
- INV_TEMP=30 floor ≈ 0.59 nats (linear scaling of the dominant entropy term)

So v13 has roughly **10x the headroom** v12 had to find.

## Risks

1. **Gradient explosion.** Sharper softmax → larger gradients on wrong predictions → potential instability. Mitigation: existing `clip_grad_norm_(all_params, 1.0)` should handle it; watch for NaN in epoch 1.
2. **Sharpness without information.** If the cosine distances themselves are non-discriminative (i.e. true and wrong tokens both ~0.5 cosine), no temperature will fix it. v13 result will diagnose this: if loss plateaus near the new floor (~0.59) while eval is still incoherent, the embedding geometry itself is the bottleneck and we'd need learnable token embeddings (the v12-plan-fallback option).
3. **Loss-progression pattern.** v13 should drop faster per epoch than v12 (~−0.5 nats/epoch instead of −0.2). If it doesn't, the temperature wasn't the bottleneck.

## What's frozen from v12

- Architecture (12L/384D/12H/RoPE/Pre-LN/AdamW, ~21.5M brain + ~38.7M engram params)
- Corpus (`dailydialog_clean.txt` + `everyday_conversations.txt`, 5.3 MB combined, vocab 14,704)
- BATCH_SIZE=96, BRAIN_LR=1e-3, EMBED_LR=5e-4, EPOCHS=5, NGRAM_TABLE_SIZE=100003
- Per-epoch refresh of `vocab_matrix_global` from drifting `embed_cache`
- N-gram memory module (EngramModule)
- Modal function timeout 12h, subprocess timeout 11h40m (assumed unchanged)

**Single-variable test.** Only `INV_TEMPERATURE` changes (10.0 → 30.0).

## Eval criteria (unchanged from v12)

1. Different replies for different prompts
2. ≥80% real-English tokens
3. ≥3 of 5 chitchat prompts produce coherent dialog-shaped replies

If v13 passes all three: ship to live frontend, archive the loop, the loss-temperature was the missing piece.

## What if v13 also fails

Two single-variable hypotheses (loss function, loss temperature) will both be exhausted. The next escalation is **not** another single-variable swap — it's an admission that 5.3 MB of corpus is too thin for a ~21M brain to find a coherent manifold under any loss. The recommended v14 is then **corpus expansion at 10-15x scale**:

- `HuggingFaceTB/smol-smoltalk` (971 MB total) subsampled to ~50-80 MB
- Same architecture, INV_TEMP=30
- Trim epochs to 3 to fit Modal's 12h budget
- Estimated cost: ~$8-10 (longer per-epoch, fewer epochs)

That requires explicit budget extension beyond the original $50 ceiling.

## Cost

Estimated **$5-6** on Modal L4, same wall time as v12 (loss-temperature change has zero throughput impact). After v13 the cumulative spend would be ~$50-51 — at or just over the original ceiling. **Requires user approval to launch.**

## Files involved

- `ingest.py:321` — `INV_TEMPERATURE = 30.0` (was 10.0)
- `configs/engram_v13_config.json` — new
- `plans/V13_PLAN.md` — this file

## What's intentionally NOT changed

- Loss function shape (still temperature-scaled cosine cross-entropy via tied output projection)
- Architecture
- Corpus
- Generation strategy
- N-gram memory module
- Optimizer, learning rates, batch size, epochs
- Per-epoch checkpoint logic
- Modal timeouts

## How to launch

The repo doesn't currently contain a `modal_train.py` wrapper — v12 was launched outside the committed tree. To fire v13:

1. Confirm v12 cumulative spend and approve $5-6 for v13.
2. Launch the Modal training job pointing at `configs/engram_v13_config.json` (or with the staged `ingest.py` directly).
3. After ~12 hours: download artifacts to `models/v13_xent_temp30/`, write `model_card.json`, run `eval_chat.py`.
4. Per the deploy-after-training rule: update `server.py` `ACTIVE_MODEL = "v13_xent_temp30"`, restart, verify `/status`, update both public status pages.
