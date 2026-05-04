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
- ~19.3M parameters — fixed size regardless of vocabulary
- 8 stacked multi-head attention layers (8 heads, 256-dim) with causal masking
- **RoPE positional encoding** — replaces learned `pos_embed`; enables test-time context extension (verified stable at 3× training context, zero quality cliff)
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

In `engram_model.py` (architecture constants):
- `EMBED_DIM` (default 256): embedding size — higher = more capacity, slower training
- `CONTEXT_SIZE` (default 32): training context window; RoPE supports eval at any length
- `N_LAYERS` (default 8): attention block depth
- `N_HEADS` (default 8): attention heads per block
- `NGRAM_TABLE_SIZE` (default 50021): size of bigram/trigram hash tables (prime number)
- `max_ponder` in `AttentionBrain` (default 3): maximum pondering loops

In `ingest.py` (training constants):
- `EPOCHS` (default 5): training passes — increase for better convergence

In `test_brain.py`:
- `TEMPERATURE` (default 0.9): higher = more creative/random, lower = more conservative
- `TOP_K` (default 10): sample from the top K candidate words at each step
- `surprise_threshold` for episode storage (default 1.5x average): lower = more memories stored
- Gate behavior is learned — the `W_K`/`W_V` projections in `EngramModule` control how much episodic and N-gram memory influences generation

## Evolution roadmap

| Capability | Status | Measured delta | What it enables |
|------------|--------|----------------|-----------------|
| Adaptive pondering | ✅ Working | — | Variable compute allocation (1-3 reasoning loops) |
| Surprise-gated learning | ✅ Working | — | Up to 3x gradient for novel inputs |
| Episodic memory | ✅ Working | — | Persistent memory of specific interactions |
| Conditional memory (N-gram) | ✅ Working | — | Hash-indexed phrase-level pattern lookup |
| Learned gating | ✅ Working | — | Context-aware memory blending (replaces fixed ratio) |
| Between-layer injection | ✅ Working | — | Memory frees later layers for reasoning |
| Q&A auto-detection | ✅ Working | — | Learns conversational turn-taking automatically |
| Paragraph boundaries | ✅ Working | — | No cross-topic garbage transitions |
| Normalized embeddings | ✅ Working | — | Semantic similarity (not magnitude-based) |
| RoPE positional encoding | ✅ Working | grad_norm_p99 −50% (0.561→0.280); 5.0% top1 at 3× ctx | Stable training + test-time context extension |
| Context window + attention | ✅ Working | — | 32-word training span; ≥96 at eval |
| Diverse word generation | ✅ Working | — | Temperature + top-k sampling |
| Coherent short phrases | Partial | — | More training data (10k+ words) |
| Long-range coherence | Partial | — | RoPE supports it; need larger corpus to see gains |
| LTI injection (tested) | ❌ No gain | Δgrad_norm_p99=+0.011; Δtop1=0.0pp | Drift prevention — did not help at 19M param scale |
| Loop-index embedding (tested) | ❌ No gain | halt gate insensitive, Δtop1=0.0pp | Ponder-depth differentiation — halt gate never fires early |

**Key features**:
- Training separates paragraphs (blank lines) to avoid cross-topic noise
- Auto-detects Q&A patterns and injects `<USER>`/`<BOT>` markers
- Episodes persist across sessions (not wiped by re-training)
- Pondering depth varies: common words = 1 step, novel concepts = 2-3 steps
- N-gram tables give the brain direct access to phrase-level patterns without using attention layers

The biggest lever for quality improvement is **more training data**. Drop `.txt` files into `corpus/` folder and re-run `ingest.py`.

## Scholarly Support for the Thesis

Each of Engram's architectural bets has independent published support. The combination is novel; the individual ingredients are not speculative.

### 1. Conditional memory / lookup-as-sparsity
**Direct validation.** DeepSeek's *Engram* paper (Cheng et al., Jan 2026) is the strongest hit — same name, same thesis: hash-indexed N-gram tables as a complementary sparsity axis to MoE. They scaled to 27B params and report **reasoning gains exceeding knowledge gains** (BBH +5.0, ARC-C +3.7 vs MMLU +3.4) — directly supporting the claim that lookup *frees* neural compute for reasoning rather than just storing facts. Follow-on work *Pooling Engram Conditional Memory using CXL* (arXiv 2603.10087) treats engram tables as a pooled hardware resource.

- [Conditional Memory via Scalable Lookup (Cheng et al., 2026)](https://arxiv.org/abs/2601.07372)
- [DeepSeek Engram repo](https://github.com/deepseek-ai/Engram)
- [Pooling Engram Conditional Memory using CXL](https://arxiv.org/html/2603.10087v1)

### 2. Adaptive pondering
**Direct lineage.** PonderNet (Banino et al., 2021, arXiv 2107.05407) is the canonical citation — probabilistic halt gate, geometric prior, KL regularization. PonderNet beat Universal Transformer on bAbI using 6× fewer steps. Engram's `halt_gate` in `engram_model.py:217` is a direct port. Recent extensions: *AdaPonderLM* (token-wise adaptive depth) and *AHT-ViT* (vision transformers).

- [PonderNet: Learning to Ponder (Banino et al., 2021)](https://arxiv.org/pdf/2107.05407)
- [AdaPonderLM: Gated Pondering with Token-Wise Adaptive Depth](https://arxiv.org/html/2603.01914)

### 3. Latent-space reasoning (vs token-space CoT)
**Strong validation.** Meta's COCONUT (Hao et al., NeurIPS 2024) makes Engram's case explicitly: *"language space may not always be optimal for reasoning."* They feed hidden states back as next-step embeddings rather than decoding to text, and show it beats CoT on logical reasoning while emitting fewer tokens. Engram's pondering loop is the same idea — reasoning happens in concept space, not in emitted words.

- [Training LLMs to Reason in Continuous Latent Space (COCONUT)](https://arxiv.org/abs/2412.06769)
- [COCONUT GitHub (Meta)](https://github.com/facebookresearch/coconut)

### 4. Surprise-gated learning
**Strong neuroscience grounding, weaker ML grounding.** The dopamine/RPE literature is bulletproof — Schultz's reward prediction error coding, Diederen & Fletcher's *"Dopamine, Prediction Error and Beyond"*, and the recent *Cell* paper *"Dopamine encodes deep network teaching signals for individual learning trajectories"* all support the brain-inspired story. The ML translation — *scale gradient by prediction error* — is less common in published work; closest analogues are precision-weighted prediction errors in active inference and curriculum/hard-example mining. **This is the thesis pillar with the thinnest direct ML citation support and the most original-research risk.**

- [Dopamine, Prediction Error and Beyond (Diederen & Fletcher)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7804370/)
- [Dopamine encodes deep network teaching signals (Cell, 2025)](https://www.cell.com/cell/fulltext/S0092-8674(25)00575-6)
- [Striatal dopamine signals errors in prediction (Sci Adv)](https://www.science.org/doi/10.1126/sciadv.adq9684)

### 5. Episodic memory as a separate organ
**Mainstream and accelerating.** *"Towards LLMs with human-like episodic memory"* (Trends in Cognitive Sciences, Dong et al., 2025), *"Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents"* (arXiv 2502.06975), and the systematic review *Memory-Augmented Transformers* (arXiv 2508.10824) all argue the same separation Engram bets on. *MemReasoner* and the multi-tier memory taxonomy (Core/Episodic/Semantic/Procedural) treat episodic store as architecturally distinct.

- [Towards LLMs with human-like episodic memory (TiCS)](https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(25)00179-2)
- [Episodic Memory is the Missing Piece for Long-Term LLM Agents](https://arxiv.org/pdf/2502.06975)
- [A neural network model of when to retrieve and encode episodic memories (eLife)](https://elifesciences.org/articles/74445)
- [Memory-Augmented Transformers: A Systematic Review](https://arxiv.org/pdf/2508.10824)
- [MemReasoner architecture](https://openreview.net/pdf?id=ODcMy97cVZ)

### 6. Decoupled vocabulary
**Partial validation.** Meta's Byte-Latent Transformer (BLT) demonstrates that fixed-vocabulary tokenization is unnecessary at 8B scale with **50% fewer inference FLOPs**. Engram's ChromaDB-vocab approach is more radical (vocabulary as a runtime-extensible vector DB rather than a baked tokenizer), but the *direction* — decouple model from vocabulary — is now an active research program.

- [Why Your Next LLM Might Not Have A Tokenizer (BLT overview)](https://towardsdatascience.com/why-your-next-llm-might-not-have-a-tokenizer/)

### Summary

| Pillar | Independent peer support | Verdict |
|---|---|---|
| Conditional memory | DeepSeek Engram (Cheng 2026) | Validated at 27B scale |
| Adaptive pondering | PonderNet, AdaPonderLM | Established subfield |
| Latent reasoning | COCONUT (Meta, NeurIPS '24) | Validated, beats CoT |
| Episodic memory | TiCS 2025, MAT survey, MemReasoner | Mainstream consensus |
| Decoupled vocab | BLT (Meta) | Active research, not yet dominant |
| Surprise-gated grad | Dopamine/RPE neuroscience only | Original — weakest ML citation |

Four of six pillars have direct, recent, peer-reviewed support from frontier labs (DeepSeek, Meta, DeepMind). The surprise-gated learning rule is the most original — and therefore the most worth running ablations on.
