---
title: Engram
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# engram

A miniature agentic language model built from scratch using PyTorch — no pre-trained weights, no APIs. Engram departs from traditional LLMs by embedding **agentic reasoning** and **persistent memory** directly into the architecture.

## Architecture

Engram separates **reasoning** (PyTorch) from **vocabulary** (ChromaDB) and adds three agentic capabilities that make it fundamentally different from standard next-token predictors.

```
[ChromaDB: infinite learnable word vectors, normalized]
           ↓  look up last N words
[PyTorch AttentionBrain: fixed-size, adaptive pondering]
           ↓  predict next concept vector
[ChromaDB Episodic Memory: retrieve similar brain states]
           ↓  blend memories with prediction
[ChromaDB: nearest-neighbor search → word]
```

**PyTorch brain** (`AttentionBrain`):
- ~137k parameters — fixed size regardless of vocabulary
- 3 stacked attention layers with causal masking
- **Adaptive pondering**: loops through layers up to 3 times with learned halt gate
- **Allocates more compute to difficult inputs** (like PonderNet/TRM)
- Knows HOW to think in concept space, not WHAT words mean

**ChromaDB vocabulary** (concept space):
- Each word = coordinate in 64D concept space
- Vectors are **learnable** via gradient descent
- **L2-normalized** for semantic similarity (not magnitude-based)
- New words can be added anytime without touching brain architecture

**ChromaDB episodic memory** (hippocampus):
- Separate collection storing specific interaction moments
- Indexed by brain's internal state, not word identity
- Retrieved during generation and blended with predictions
- **Dynamic topic retrieval emerges from embedding geometry** — no explicit topic management needed

**Three agentic capabilities**:
1. **Surprise-gated learning** (dopamine signal): High prediction error = learn more aggressively (up to 3x gradient). Low error = learn gently. Physical allocation of neural change to novel moments.
2. **Episodic memory** (hippocampus): Remembers specific interactions in a searchable brain-state-indexed collection. Blends retrieved memories during generation.
3. **Recurrent pondering** (adaptive compute): Loops through attention blocks 1-3 times based on learned halt gate. More "thinking" for novel inputs.

**What makes this different from GPT**:
Standard LLMs treat every token identically, compress everything into weights, and use fixed compute per token. Engram physically allocates more neural change to surprises, remembers specific moments in episodic memory, and adaptively allocates reasoning depth. The vocabulary is an external, persistent, continuously-updatable semantic space.

## Files

| File | Purpose |
|------|---------|
| `ingest.py` | Train on `training_data.txt` + `corpus/*.txt`. Auto-detects Q&A pairs, trains with pondering, saves normalized embeddings |
| `test_brain.py` | Interactive chat. Shows surprise scores, episodic memory stores, pondering steps, and subconscious thoughts |
| `training_data.txt` | Training corpus — add text here or drop files in `corpus/` folder |
| `corpus/` | Additional .txt files for training (optional) |
| `engram_weights.pth` | Saved PyTorch brain weights |
| `engram_memory/` | ChromaDB persistence: `engram_vocab` (words) + `engram_episodes` (memories) |

## Quickstart

```bash
# 1. Train on the corpus
uv run ingest.py

# 2. Chat with the trained brain
uv run test_brain.py
```

## Tuning knobs

In `ingest.py`:
- `EMBED_DIM` (default 64): embedding size — higher = more capacity, slower training
- `CONTEXT_SIZE` (default 8): how many past words to attend over
- `N_LAYERS` (default 3): attention block depth
- `EPOCHS` (default 5): training passes — increase for better convergence
- `max_ponder` in `AttentionBrain` (default 3): maximum pondering loops

In `test_brain.py`:
- `TEMPERATURE` (default 0.9): higher = more creative/random, lower = more conservative
- `TOP_K` (default 10): sample from the top K candidate words at each step
- `surprise_threshold` for episode storage (default 1.5x average): lower = more memories stored
- `episode_blend_weight` (default 0.3): how much to blend retrieved episodes with prediction

## Evolution roadmap

| Capability | Status | What it enables |
|------------|--------|-----------------|
| Adaptive pondering | ✅ Working | Variable compute allocation (1-3 reasoning loops) |
| Surprise-gated learning | ✅ Working | Up to 3x gradient for novel inputs |
| Episodic memory | ✅ Working | Persistent memory of specific interactions |
| Q&A auto-detection | ✅ Working | Learns conversational turn-taking automatically |
| Paragraph boundaries | ✅ Working | No cross-topic garbage transitions |
| Normalized embeddings | ✅ Working | Semantic similarity (not magnitude-based) |
| Context window + attention | ✅ Working | 8-word memory span |
| Diverse word generation | ✅ Working | Temperature + top-k sampling |
| Coherent short phrases | Partial | More training data (10k+ words) |
| Long-range coherence | Not yet | Larger model, more data, more epochs |

**Key features**:
- Training separates paragraphs (blank lines) to avoid cross-topic noise
- Auto-detects Q&A patterns and injects `<USER>`/`<BOT>` markers
- Episodes persist across sessions (not wiped by re-training)
- Pondering depth varies: common words = 1 step, novel concepts = 2-3 steps

The biggest lever for quality improvement is **more training data**. Drop `.txt` files into `corpus/` folder and re-run `ingest.py`.
