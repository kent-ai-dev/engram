---
title: Engram
emoji: "🧠"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
---

![Engram](https://media2.dev.to/dynamic/image/width=1000,height=420,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Ftc7rra34vaay4gczhw78.png)

# Engram

> 📖 **Read the full story**: [Engram: A New Type of AI](https://dev.to/nodefiend/engram-a-new-type-of-ai-1jbd) on DEV Community

A miniature agentic language model built from scratch using PyTorch — no pre-trained weights, no APIs. Engram departs from traditional LLMs by embedding **agentic reasoning** and **persistent memory** directly into the architecture.

## Architecture

Engram separates **reasoning** (PyTorch) from **vocabulary** (ChromaDB), adds **conditional memory** via hash-indexed N-gram tables, and includes three agentic capabilities that make it fundamentally different from standard next-token predictors.

```
[ChromaDB: infinite learnable word vectors, normalized]
           ↓  look up last N words
[PyTorch AttentionBrain: Layer 0]
           ↓
  + [N-gram Memory via Learned Gate]  ← hash-indexed bigram/trigram tables
           ↓
[PyTorch AttentionBrain: Layers 1-N, adaptive pondering]
           ↓  predict next concept vector
  + [Episodic Memory via Learned Gate] ← ChromaDB brain-state-indexed episodes
           ↓
[ChromaDB: nearest-neighbor search → word]
```

### Conditional Memory (N-gram Tables)

Based on the insight from *"Conditional Memory via Scalable Lookup"* (Cheng et al., 2026) — language models waste neural depth reconstructing static multi-word patterns that should just be looked up. The `EngramModule` adds:

- **Hash-indexed embedding tables** for bigrams and trigrams (O(1) lookup)
- **Learned gating** that controls how much memory flows into the hidden state
- **Between-layer injection**: memory is added after Layer 0, freeing later layers for compositional reasoning
- The same gate is reused for episodic memory, replacing the old fixed blend ratio

This is functionally equivalent to doubling the model's effective depth — Layer 5 with memory matches Layer 12 without it (per the paper's benchmarks).

**PyTorch brain** (`AttentionBrain`):
- ~137k parameters — fixed size regardless of vocabulary
- 4 stacked attention layers with causal masking
- **Adaptive pondering**: loops through layers up to 3 times with learned halt gate
- **Allocates more compute to difficult inputs** (like PonderNet/TRM)
- Knows HOW to think in concept space, not WHAT words mean

**Engram memory module** (`EngramModule`):
- ~480k parameters — bigram + trigram embedding tables
- Hash-indexed for O(1) lookup of multi-word patterns
- Learned gating via `W_K`/`W_V` projections (shared for N-gram and episodic memory)
- ~25% of total params dedicated to memory (matching the paper's optimal ratio)

**ChromaDB vocabulary** (concept space):
- Each word = coordinate in 96D concept space
- Vectors are **learnable** via gradient descent
- **L2-normalized** for semantic similarity (not magnitude-based)
- New words can be added anytime without touching brain architecture

**ChromaDB episodic memory** (hippocampus):
- Separate collection storing specific interaction moments
- Indexed by brain's internal state, not word identity
- Retrieved during generation and blended via **learned gate** (replaces fixed 0.3 weight)
- **Dynamic topic retrieval emerges from embedding geometry** — no explicit topic management needed

**Four agentic capabilities**:
1. **Conditional memory** (N-gram lookup): Hash-indexed phrase-level patterns injected between layers via learned gate. Frees later layers for reasoning.
2. **Surprise-gated learning** (dopamine signal): High prediction error = learn more aggressively (up to 3x gradient). Low error = learn gently.
3. **Episodic memory** (hippocampus): Remembers specific interactions in a searchable brain-state-indexed collection. Blended via learned gate during generation.
4. **Recurrent pondering** (adaptive compute): Loops through attention blocks 1-3 times based on learned halt gate. More "thinking" for novel inputs.

**What makes this different from GPT**:
Standard LLMs treat every token identically, compress everything into weights, and use fixed compute per token. Engram physically allocates more neural change to surprises, remembers specific moments in episodic memory, looks up phrase-level patterns via hash tables, and adaptively allocates reasoning depth. The vocabulary is an external, persistent, continuously-updatable semantic space.

## Relationship to DeepSeek's Engram Paper

This project's name and core architecture (separating reasoning from memory) predates the DeepSeek paper. In January 2026, Cheng et al. published *"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"*, which independently validated the same architectural philosophy and named their module "Engram." Their paper provides rigorous benchmarks proving that memory lookup is complementary to neural computation — reasoning gains exceed knowledge gains (BBH +5.0 vs MMLU +3.4), and memory injection effectively doubles model depth.

We've incorporated three techniques from their paper: N-gram embedding tables, learned gating, and between-layer memory injection.

## Files

| File | Purpose |
|------|---------|
| `ingest.py` | Train on `training_data.txt` + `corpus/*.txt`. Auto-detects Q&A pairs, trains brain + N-gram tables, saves normalized embeddings |
| `test_brain.py` | Interactive chat. Shows surprise scores, episodic memory stores, pondering steps, and subconscious thoughts |
| `eval_brain.py` | Non-interactive evaluation with test prompts, coherence scoring, and JSON results |
| `app.py` | Gradio chat interface for Hugging Face Spaces |
| `training_data.txt` | Training corpus — add text here or drop files in `corpus/` folder |
| `corpus/` | Additional .txt files for training (optional) |
| `engram_weights.pth` | Saved PyTorch brain weights |
| `engram_memory_module.pth` | Saved N-gram embedding tables + gating weights |
| `engram_word_to_id.pth` | Word-to-ID mapping for N-gram hashing |
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
- `EMBED_DIM` (default 96): embedding size — higher = more capacity, slower training
- `CONTEXT_SIZE` (default 32): how many past words to attend over
- `N_LAYERS` (default 4): attention block depth
- `EPOCHS` (default 1): training passes — increase for better convergence
- `NGRAM_TABLE_SIZE` (default 4999): size of bigram/trigram hash tables (prime number)
- `max_ponder` in `AttentionBrain` (default 3): maximum pondering loops

In `test_brain.py`:
- `TEMPERATURE` (default 0.9): higher = more creative/random, lower = more conservative
- `TOP_K` (default 10): sample from the top K candidate words at each step
- `surprise_threshold` for episode storage (default 1.5x average): lower = more memories stored
- Gate behavior is learned — the `W_K`/`W_V` projections in `EngramModule` control how much episodic and N-gram memory influences generation

## Evolution roadmap

| Capability | Status | What it enables |
|------------|--------|-----------------|
| Adaptive pondering | ✅ Working | Variable compute allocation (1-3 reasoning loops) |
| Surprise-gated learning | ✅ Working | Up to 3x gradient for novel inputs |
| Episodic memory | ✅ Working | Persistent memory of specific interactions |
| Conditional memory (N-gram) | ✅ Working | Hash-indexed phrase-level pattern lookup |
| Learned gating | ✅ Working | Context-aware memory blending (replaces fixed ratio) |
| Between-layer injection | ✅ Working | Memory frees later layers for reasoning |
| Q&A auto-detection | ✅ Working | Learns conversational turn-taking automatically |
| Paragraph boundaries | ✅ Working | No cross-topic garbage transitions |
| Normalized embeddings | ✅ Working | Semantic similarity (not magnitude-based) |
| Context window + attention | ✅ Working | 32-word memory span |
| Diverse word generation | ✅ Working | Temperature + top-k sampling |
| Coherent short phrases | Partial | More training data (10k+ words) |
| Long-range coherence | Not yet | Larger model, more data, more epochs |

**Key features**:
- Training separates paragraphs (blank lines) to avoid cross-topic noise
- Auto-detects Q&A patterns and injects `<USER>`/`<BOT>` markers
- Episodes persist across sessions (not wiped by re-training)
- Pondering depth varies: common words = 1 step, novel concepts = 2-3 steps
- N-gram tables give the brain direct access to phrase-level patterns without using attention layers

The biggest lever for quality improvement is **more training data**. Drop `.txt` files into `corpus/` folder and re-run `ingest.py`.
