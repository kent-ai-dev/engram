# Plan: Autonomous Training Subagent for Engram using OpenClaw

## Context

Engram is a 137k-parameter agentic language model with adaptive pondering, surprise-gated learning, and episodic memory. It currently trains on ~50K words (training_data.txt + 2 Project Gutenberg books) with basic hyperparameters (64-dim embeddings, 8-word context, 3 layers).

**Problem:** To compete with frontier models, Engram needs systematic scaling through:
- More training data (expand corpus to ~500K words)
- Larger model architectures (256-1024 dim, 128+ context, 6-12 layers)
- Hyperparameter optimization (learning rates, epochs, batch sizes)

**Solution:** Build an autonomous OpenClaw subagent that continuously trains, evaluates, and improves Engram without manual intervention. The subagent will explore multiple model scales in parallel (gradual 64→96→128 vs immediate jump to 256-dim) over 5-10 training iterations.

**Why OpenClaw Subagents:** Leverages persistent session capability for multi-iteration workflows, integrated monitoring via `/subagents log/info`, and repo access for automated file operations.

**Constraint:** No external LLM API calls (no Anthropic/OpenAI for synthetic data generation). Will use Project Gutenberg corpus expansion instead.

---

## Implementation Approach

### 1. Create Automated Evaluation Script

**File:** `/Users/kennethchambers/Documents/GitHub/engram/eval_brain.py`

**Purpose:** Non-interactive version of test_brain.py that:
- Loads trained model weights
- Runs predefined test prompts
- Extracts quantitative metrics
- Outputs structured JSON results

**Test Prompts:**
```python
[
    "what is the capital of france",
    "tell me a story about adventure",
    "how do you make coffee",
    "what do you think about friendship",
    "can you help me understand mathematics"
]
```

**Metrics to Extract:**
- Average surprise score (MSE loss per token)
- Average ponder steps (1-3, indicates reasoning depth)
- Response coherence (consecutive valid words / total words)
- Response length (words before stopping)
- Vocabulary size (from ChromaDB)
- Episode count (episodic memory accumulation)

**Output Format:**
```json
{
  "timestamp": "2026-03-19T10:30:00",
  "model_config": {"embed_dim": 64, "context_size": 8, "n_layers": 3},
  "metrics": {
    "avg_surprise": 0.845,
    "avg_ponder_steps": 1.2,
    "coherence_score": 0.65,
    "avg_response_length": 12.4,
    "vocab_size": 8900,
    "episode_count": 0
  },
  "test_results": [
    {"prompt": "what is the capital of france", "response": "...", "surprise": 0.82}
  ]
}
```

**Implementation Notes:**
- Base on test_brain.py structure (lines 143-299)
- Remove interactive input loop, replace with test_prompts array
- Capture generation output instead of printing to console
- Calculate coherence: count real words from vocab vs nonsense
- Save results to `eval_results_<timestamp>.json`

---

### 2. Create Training Orchestration Script

**File:** `/Users/kennethchambers/Documents/GitHub/engram/train_runner.py`

**Purpose:** Wrapper that handles:
- Hyperparameter configuration updates
- Training execution (calls ingest.py)
- Evaluation execution (calls eval_brain.py)
- Results logging
- Git commits for checkpointing

**Configuration Matrix:**
```python
CONFIGS = [
    # Gradual scaling experiments
    {"name": "baseline", "embed_dim": 64, "context_size": 8, "n_layers": 3, "epochs": 5},
    {"name": "medium", "embed_dim": 96, "context_size": 12, "n_layers": 4, "epochs": 7},
    {"name": "large", "embed_dim": 128, "context_size": 16, "n_layers": 5, "epochs": 10},

    # Jump-to-scale experiments
    {"name": "target_small", "embed_dim": 256, "context_size": 128, "n_layers": 6, "epochs": 5},
    {"name": "target_medium", "embed_dim": 256, "context_size": 128, "n_layers": 6, "epochs": 10},
]
```

**Workflow per Iteration:**
1. Download 1-2 new books via `download_book.py`
2. For each config:
   - Update hyperparameters in ingest.py (lines 19-27)
   - Run `uv run ingest.py` and capture output
   - Parse training loss and ponder steps from stdout
   - Run `uv run eval_brain.py` and load JSON results
   - Save checkpoint: `models/engram_weights_{config_name}_{iteration}.pth`
3. Log all results to `training_log.jsonl` (one JSON object per line)
4. Commit changes: `git add . && git commit -m "Iteration {N}: {summary}"`
5. Compare configs, recommend best performer

**Hyperparameter Update Strategy:**
```python
def update_ingest_params(config):
    with open('ingest.py', 'r') as f:
        content = f.read()

    content = re.sub(r'EMBED_DIM = \d+', f'EMBED_DIM = {config["embed_dim"]}', content)
    content = re.sub(r'CONTEXT_SIZE = \d+', f'CONTEXT_SIZE = {config["context_size"]}', content)
    content = re.sub(r'N_LAYERS = \d+', f'N_LAYERS = {config["n_layers"]}', content)
    content = re.sub(r'EPOCHS = \d+', f'EPOCHS = {config["epochs"]}', content)

    with open('ingest.py', 'w') as f:
        f.write(content)
```

---

### 3. Spawn OpenClaw Subagent

**Command:**
```
/subagents spawn training-agent "Autonomous Engram Training" --mode session
```

**Task Instructions for Subagent:**

```
You are an autonomous training orchestrator for the Engram language model. Execute 5-10 training iterations to systematically improve the model through corpus expansion and architecture scaling.

REPOSITORY: /Users/kennethchambers/Documents/GitHub/engram

YOUR MISSION:
Explore model scaling in parallel by training multiple configurations per iteration:
- Gradual scaling: 64→96→128 dimensions
- Jump-to-scale: 256 dimensions with 128 context
- Data expansion: Add 1-2 Project Gutenberg books per iteration

ITERATION WORKFLOW:
1. DATA ACQUISITION
   - Check which books are already downloaded (ls corpus/)
   - Download 1-2 new books: python download_book.py
   - Available books: Frankenstein (84), Pride & Prejudice (1342), Dracula (345),
     Time Machine (35), Moby Dick (2701), Huck Finn (76), Tale of Two Cities (98),
     Being Earnest (844)

2. PARALLEL TRAINING
   - Run python train_runner.py to train all configs
   - Monitor training loss, ponder steps, vocab growth
   - Training takes 5-30 minutes depending on config

3. EVALUATION
   - Each config automatically evaluated via eval_brain.py
   - Results saved to eval_results_<config>_<iteration>.json

4. ANALYSIS
   - Compare configs: Which improved most? Which metrics matter?
   - Log insights to training_log.jsonl

5. DECISION
   - If iteration < 5: Continue with next book download
   - If iteration >= 5: Analyze trends, recommend best config
   - If loss plateaus for 3 iterations: Consider increasing epochs or dimensions

CRITICAL FILES:
- ingest.py: Training entry point (hyperparameters at lines 19-27)
- eval_brain.py: Evaluation script (you will create this based on test_brain.py)
- train_runner.py: Orchestration script (you will create this)
- download_book.py: Data acquisition (call this to add books)
- training_log.jsonl: Your progress log (append after each iteration)

OUTPUT FORMAT (after each iteration):
```
=== ITERATION {N} COMPLETE ===
Books downloaded: {list}
Corpus size: {X} words

Config Results:
- baseline (64-dim): loss=1.024, coherence=0.62, vocab=9.2K
- medium (96-dim): loss=0.987, coherence=0.68, vocab=9.5K ⭐ BEST
- large (128-dim): loss=1.045, coherence=0.61, vocab=9.8K
- target_small (256-dim): loss=0.892, coherence=0.71, vocab=10.1K ⭐⭐ BEST OVERALL
- target_medium (256-dim, 10 epochs): loss=0.856, coherence=0.75, vocab=10.3K ⭐⭐⭐ WINNER

Key Insights: {what you learned}
Next Step: {what you'll do in next iteration}

Progress: {N}/10 iterations | {plateau count}/3 until stop
```

STOPPING CONDITIONS:
- 10 iterations completed
- Loss stops improving for 3 consecutive iterations
- All 8 books downloaded AND all configs tested
- User sends "stop" command via /subagents send

CONSTRAINTS:
- NO external API calls (no Anthropic, OpenAI, etc.)
- Only use Project Gutenberg for new data
- Keep git commits after each iteration
- If training fails, log error and skip that config

START ACTION:
1. Verify current state: ls corpus/, ls *.pth, check training_log.jsonl
2. Create eval_brain.py based on test_brain.py
3. Create train_runner.py with config matrix
4. Begin iteration 1 with book download

BEGIN NOW. Report status after each major step.
```

---

## Monitoring & Control

### User Commands

**Monitor progress:**
```bash
/subagents list                    # View active subagent
/subagents log training-agent      # See full execution log
/subagents info training-agent     # Check current status
cat training_log.jsonl             # Review iteration results
ls models/                         # See saved checkpoints
```

**Check intermediate results:**
```bash
cat eval_results_baseline_1.json  # View specific config evaluation
git log --oneline                  # See iteration commits
```

**Control execution:**
```bash
/subagents send training-agent "status"           # Request current iteration
/subagents send training-agent "stop"             # Finish current iteration then halt
/subagents send training-agent "skip-evaluation"  # Skip eval, just train
/subagents kill training-agent                    # Emergency stop
```

**Manual testing:**
```bash
# After any iteration, manually test the model
uv run test_brain.py

# Compare two model checkpoints
cp models/engram_weights_baseline_5.pth engram_weights.pth
uv run test_brain.py  # Test baseline

cp models/engram_weights_target_small_5.pth engram_weights.pth
uv run test_brain.py  # Test target config
```

---

## File Changes

### New Files Created by Subagent

1. **`/Users/kennethchambers/Documents/GitHub/engram/eval_brain.py`**
   - Automated evaluation script
   - ~150 lines, based on test_brain.py structure
   - Returns JSON metrics instead of interactive chat

2. **`/Users/kennethchambers/Documents/GitHub/engram/train_runner.py`**
   - Training orchestration script
   - ~200 lines, manages config matrix
   - Calls ingest.py and eval_brain.py for each config

3. **`/Users/kennethchambers/Documents/GitHub/engram/training_log.jsonl`**
   - Progress log (JSON Lines format)
   - One entry per iteration
   - Tracks all configs and metrics

4. **`/Users/kennethchambers/Documents/GitHub/engram/models/`** (directory)
   - Stores model checkpoints
   - Format: `engram_weights_{config_name}_{iteration}.pth`
   - Allows comparing models across iterations

5. **`/Users/kennethchambers/Documents/GitHub/engram/eval_results_*.json`**
   - Evaluation results per config per iteration
   - Format: `eval_results_{config}_{iteration}.json`

### Modified Files

1. **`/Users/kennethchambers/Documents/GitHub/engram/ingest.py`**
   - Lines 19-27: Hyperparameters updated by train_runner.py
   - No permanent changes (reverts per config)

2. **`/Users/kennethchambers/Documents/GitHub/engram/corpus/`** (directory)
   - New .txt files added via download_book.py
   - Start: 2 books (~740KB)
   - End: 8-10 books (~2-3MB)

---

## Expected Outcomes

**After 5-10 iterations:**

1. **Expanded Corpus**
   - Start: 2 books (~50K words of training sequences)
   - End: 8-10 books (~300-500K words)
   - Diversity: Victorian literature, sci-fi, mysteries, drama

2. **Optimized Architecture**
   - Evidence-based hyperparameter choices
   - Clear winner among: 64/96/128 gradual vs 256 jump-to-scale
   - Recommended config for production use

3. **Performance Metrics**
   - Training loss: Expect 20-40% reduction
   - Coherence score: Expect 0.60 → 0.75+
   - Vocabulary: Expect 8.9K → 15-20K tokens
   - Ponder steps: More varied (not stuck at 1)
   - Episode count: 100-500 memories accumulated

4. **Comprehensive Log**
   - `training_log.jsonl`: 50-100 data points (5-10 iterations × 5 configs)
   - Graphs showing loss curves, coherence trends, vocab growth
   - Clear recommendation: "Use config X for best balance of speed/quality"

5. **Model Checkpoints**
   - 50+ saved models in `models/` directory
   - Ability to revert to any iteration
   - Best model promoted to `engram_weights.pth`

**Success Criteria:**
- ✅ Loss decreases consistently across iterations
- ✅ 256-dim model outperforms 64-dim baseline by 30%+
- ✅ Coherence score above 0.70 for best config
- ✅ No training failures or crashes
- ✅ Clear recommendation for next scaling step (1024-dim? 512 context?)

---

## Risk Mitigation

**Risk 1: Training takes too long (>1 hour per config)**
- **Mitigation:** Start with low epoch counts (5), scale up gradually
- **Fallback:** Reduce config matrix to 3 configs instead of 5
- **Detection:** Monitor first iteration timing

**Risk 2: Large models cause OOM errors**
- **Mitigation:** Test 256-dim config first before going larger
- **Fallback:** Reduce BATCH_SIZE from 64 to 32 or 16
- **Detection:** Watch for "CUDA out of memory" or system crashes

**Risk 3: Corpus too large (slow loading)**
- **Mitigation:** Monitor file sizes, stay under 5MB total
- **Fallback:** Remove some books, focus on quality over quantity
- **Detection:** If loading takes >30 seconds

**Risk 4: Subagent gets stuck or produces no output**
- **Mitigation:** Explicit logging at each step in task instructions
- **Fallback:** Kill and restart with simpler instructions
- **Detection:** No log activity for 10+ minutes

**Risk 5: Hyperparameter changes break training**
- **Mitigation:** Keep backup of working model before each iteration
- **Fallback:** Revert to baseline config (64-dim) if issues arise
- **Detection:** Training loss increases or NaN values

**Risk 6: Evaluation metrics don't reflect quality**
- **Mitigation:** Manual spot-checks via test_brain.py every 2-3 iterations
- **Fallback:** Add human evaluation prompts to eval_brain.py
- **Detection:** Metrics improve but chat output still incoherent

---

## Verification Steps

**After subagent completes:**

1. **Check Training Log**
   ```bash
   cat training_log.jsonl | jq '.iteration, .best_config, .metrics'
   ```
   - Verify 5-10 iterations logged
   - Confirm metrics improve over time
   - Identify best performing config

2. **Review Model Checkpoints**
   ```bash
   ls -lh models/
   ```
   - Should have 50+ .pth files
   - File sizes should increase for larger configs (256-dim > 64-dim)

3. **Test Best Model**
   ```bash
   cp models/engram_weights_target_medium_10.pth engram_weights.pth
   uv run test_brain.py
   ```
   - Chat with model manually
   - Verify improved coherence vs baseline
   - Check surprise scores are lower (better learned)

4. **Verify Corpus Expansion**
   ```bash
   ls corpus/ | wc -l    # Should be 8-10 files
   wc -w corpus/*.txt    # Should be 300K-500K words total
   ```

5. **Check Git History**
   ```bash
   git log --oneline --graph | head -20
   ```
   - Should have commit per iteration
   - Messages should describe what was trained

6. **Analyze Metrics**
   ```python
   # Load and plot training_log.jsonl
   import json, matplotlib.pyplot as plt

   with open('training_log.jsonl') as f:
       logs = [json.loads(line) for line in f]

   # Plot loss curves for each config
   for config in ['baseline', 'medium', 'large', 'target_small', 'target_medium']:
       losses = [log['metrics'][config]['loss'] for log in logs if config in log['metrics']]
       plt.plot(losses, label=config)

   plt.legend()
   plt.xlabel('Iteration')
   plt.ylabel('Training Loss')
   plt.title('Engram Training Progress')
   plt.savefig('training_curves.png')
   ```

**End-to-End Test:**
```bash
# Full cycle validation
uv run ingest.py              # Baseline training works
uv run test_brain.py          # Manual testing works
uv run eval_brain.py          # Automated eval works
python train_runner.py        # Orchestration works
python download_book.py       # Data acquisition works
```

---

## Next Steps After This Plan

1. **Immediate:** Spawn the OpenClaw subagent with provided instructions
2. **Monitor:** Check logs every 30-60 minutes for first 2 iterations
3. **Validate:** After iteration 1, verify eval_brain.py and train_runner.py were created correctly
4. **Adjust:** Send guidance if subagent diverges from plan
5. **Review:** After 5-10 iterations, analyze results and plan Phase 2 scaling (1024-dim, MoE, etc.)

**Timeline Estimate:**
- Iteration 1-2: ~1 hour (subagent creates scripts, first training)
- Iteration 3-5: ~2-3 hours (parallel configs take longer)
- Iteration 6-10: ~3-4 hours (larger models train slower)
- **Total: 6-8 hours of autonomous operation**
