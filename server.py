"""
Engram Chat Server — FastAPI backend for the Engram ML model.
Serves a chat web UI and provides inference via the trained AttentionBrain model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chromadb
import re
import os
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engram_model import AttentionBrain, EngramModule

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_DIM = 256
CONTEXT_SIZE = 32
N_LAYERS = 8
TEMPERATURE = 0.9
TOP_K = 10
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Active model: v4_rope (8L/256D/RoPE, trained 2026-04-25 on Modal L4).
# To roll back to large_iter4 (older, smaller, no RoPE), change ACTIVE_MODEL.
ACTIVE_MODEL = "v5_rope"
_model_dir = os.path.join(BASE_DIR, "models", ACTIVE_MODEL)
if os.path.exists(os.path.join(_model_dir, "engram_weights.pth")):
    WEIGHTS_PATH = os.path.join(_model_dir, "engram_weights.pth")
    CHROMA_PATH = os.path.join(_model_dir, "engram_memory")
    ENGRAM_PATH = os.path.join(_model_dir, "engram_memory_module.pth")
    W2ID_PATH = os.path.join(_model_dir, "engram_word_to_id.pth")
else:
    CHROMA_PATH = os.path.join(BASE_DIR, "engram_memory")
    WEIGHTS_PATH = os.path.join(BASE_DIR, "engram_weights.pth")
    ENGRAM_PATH = os.path.join(BASE_DIR, "engram_memory_module.pth")
    W2ID_PATH = os.path.join(BASE_DIR, "engram_word_to_id.pth")
GEN_STEPS = 20


# ── Inference helpers (from eval_brain.py) ─────────────────────────────────────

def nearest_words(predicted_t, word_list, vocab_matrix, word_to_idx, n=TOP_K, penalty=None):
    predicted_t = F.normalize(predicted_t, dim=0)
    dists = torch.norm(vocab_matrix - predicted_t.unsqueeze(0), dim=1).clone()
    if penalty:
        for w, pen in penalty.items():
            if w in word_to_idx:
                dists[word_to_idx[w]] += pen
    k = min(n, len(word_list))
    top_dists, top_idx = torch.topk(dists, k, largest=False)
    return [(word_list[i.item()], top_dists[j].item()) for j, i in enumerate(top_idx)]


def generate_response(brain, word_list, vocab_matrix, word_to_idx, prompt_words, context_size,
                      engram_module=None, word_to_id=None):
    SKIP = {"<START>", "<BOT>", "<USER>"}
    embed_dim = vocab_matrix.shape[1]
    context = ["<START>"] * context_size
    for w in (["<USER>"] + prompt_words + ["<BOT>"]):
        context.append(w)

    def get_embed(word):
        if word in word_to_idx:
            return vocab_matrix[word_to_idx[word]]
        return torch.zeros(embed_dim)

    reply = []
    recent = []
    ponder_steps_list = []
    surprise_list = []

    for step in range(GEN_STEPS):
        ctx_tensors = [get_embed(w) for w in context[-context_size:]]
        ctx_stack = torch.stack(ctx_tensors)

        with torch.no_grad():
            ngram_memory = None
            if engram_module is not None and word_to_id is not None:
                ids = [word_to_id.get(w, 0) for w in context[-3:]]
                ngram_memory = engram_module.lookup(ids).unsqueeze(0)
            predicted, n_steps = brain(ctx_stack.unsqueeze(0),
                                       ngram_memory=ngram_memory,
                                       engram_module=engram_module)
            predicted = predicted.squeeze(0)

        ponder_steps_list.append(n_steps)

        penalty = {w: 3.0 for w in set(recent[-4:])}
        for tok in SKIP:
            penalty[tok] = float("inf")

        candidates = nearest_words(predicted, word_list, vocab_matrix, word_to_idx, penalty=penalty)
        filtered = [(w, d) for w, d in candidates if d < float("inf")]
        if not filtered:
            break

        words_list_c = [w for w, _ in filtered]
        dists_t = torch.tensor([d for _, d in filtered])
        probs = F.softmax(-dists_t / TEMPERATURE, dim=-1)
        chosen_idx = torch.multinomial(probs, 1).item()
        chosen_word = words_list_c[chosen_idx]

        chosen_dist = filtered[chosen_idx][1]
        surprise_list.append(chosen_dist)

        if chosen_word == "<USER>":
            break

        reply.append(chosen_word)
        recent.append(chosen_word)
        context.append(chosen_word)

    avg_ponder = sum(ponder_steps_list) / max(len(ponder_steps_list), 1)
    avg_surprise = sum(surprise_list) / max(len(surprise_list), 1)
    return reply, avg_ponder, avg_surprise


# ── Global state ───────────────────────────────────────────────────────────────

model_state = {
    "loaded": False,
    "error": None,
    "brain": None,
    "engram_module": None,
    "word_to_id": None,
    "word_list": None,
    "vocab_matrix": None,
    "word_to_idx": None,
    "vocab_size": 0,
    "corpus_books": 0,
    "iterations": 0,
    "active_model": ACTIVE_MODEL,
    "use_rope": False,
}


def load_model():
    """Load model weights and ChromaDB vocab at startup."""
    try:
        if not os.path.exists(WEIGHTS_PATH):
            model_state["error"] = f"No weights found at {WEIGHTS_PATH}"
            return
        if not os.path.exists(CHROMA_PATH):
            model_state["error"] = f"No ChromaDB found at {CHROMA_PATH}"
            return

        # Load ChromaDB vocab
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        vocab_collection = chroma_client.get_collection("engram_vocab")
        all_data = vocab_collection.get(include=["embeddings"])
        embed_cache = {
            word: list(emb) for word, emb in zip(all_data["ids"], all_data["embeddings"])
        }

        # Release ChromaDB
        del vocab_collection, all_data
        try:
            chroma_client._producer = None
            chroma_client._consumer = None
        except Exception:
            pass
        del chroma_client

        # Build vocab matrix
        word_list = list(embed_cache.keys())
        word_to_idx = {w: i for i, w in enumerate(word_list)}
        vocab_matrix = torch.tensor(
            [embed_cache[w] for w in word_list], dtype=torch.float32
        )

        # Load brain — auto-detect hyperparams from checkpoint
        checkpoint = torch.load(WEIGHTS_PATH, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Infer dimensions from weight shapes
        embed_dim = state_dict["ln_final.weight"].shape[0]
        context_size = state_dict["pos_embed.weight"].shape[0]
        n_layers = max(
            int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")
        ) + 1
        use_rope = any(k.startswith("blocks.") and k.endswith(".freqs_cis") for k in state_dict)

        brain = AttentionBrain(embed_dim=embed_dim, context_size=context_size,
                               n_layers=n_layers, use_rope=use_rope)
        brain.load_state_dict(state_dict, strict=False)
        brain.eval()

        # Optional EngramModule (N-gram memory) — only if both files exist
        engram_module = None
        word_to_id = None
        if os.path.exists(ENGRAM_PATH) and os.path.exists(W2ID_PATH):
            try:
                engram_module = EngramModule(embed_dim)
                engram_module.load_state_dict(torch.load(ENGRAM_PATH, weights_only=True))
                engram_module.eval()
                word_to_id = torch.load(W2ID_PATH, weights_only=False)
                print(f"Loaded EngramModule ({len(word_to_id):,} word IDs)")
            except Exception as e:
                print(f"EngramModule load failed (continuing without N-gram memory): {e}")
                engram_module = None
                word_to_id = None

        # Update globals for inference
        model_state["embed_dim"] = embed_dim
        model_state["context_size"] = context_size
        model_state["use_rope"] = use_rope

        # Count corpus books
        corpus_books = 0
        corpus_dir = os.path.join(BASE_DIR, "corpus")
        if os.path.exists(corpus_dir):
            corpus_books = len([f for f in os.listdir(corpus_dir) if f.endswith(".txt")])

        # Count training iterations from log
        iterations = 0
        log_path = os.path.join(BASE_DIR, "training_log.jsonl")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "iteration_complete":
                            iterations = max(iterations, entry.get("iteration", 0))
                    except Exception:
                        pass

        model_state.update({
            "loaded": True,
            "error": None,
            "brain": brain,
            "engram_module": engram_module,
            "word_to_id": word_to_id,
            "word_list": word_list,
            "vocab_matrix": vocab_matrix,
            "word_to_idx": word_to_idx,
            "vocab_size": len(embed_cache),
            "corpus_books": corpus_books,
            "iterations": iterations,
        })
        print(f"Engram model loaded: model={ACTIVE_MODEL}, vocab={len(embed_cache):,}, "
              f"embed_dim={embed_dim}, n_layers={n_layers}, use_rope={use_rope}, "
              f"engram_module={'yes' if engram_module else 'no'}")

    except Exception as e:
        model_state["error"] = str(e)
        print(f"Failed to load model: {e}")


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(title="Engram Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static dir exists
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str


@app.on_event("startup")
def startup():
    load_model()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/status")
def status():
    return {
        "loaded": model_state["loaded"],
        "error": model_state["error"],
        "vocab_size": model_state["vocab_size"],
        "corpus_books": model_state["corpus_books"],
        "iterations": model_state["iterations"],
    }


@app.post("/reload")
def reload_model():
    """Reload model weights and vocab from disk without restarting the server."""
    load_model()
    return {
        "loaded": model_state["loaded"],
        "error": model_state["error"],
        "vocab_size": model_state["vocab_size"],
        "corpus_books": model_state["corpus_books"],
        "iterations": model_state["iterations"],
        "message": "Model reloaded from disk",
    }


@app.post("/chat")
def chat(req: ChatRequest):
    if not model_state["loaded"]:
        return JSONResponse(content={
            "response": "Model still training, check back soon!",
            "stats": {"vocab_size": 0, "surprise": 0, "ponder_steps": 0},
        })

    prompt_words = re.findall(r"\b\w+\b", req.message.lower())
    if not prompt_words:
        return {"response": "...", "stats": {"vocab_size": model_state["vocab_size"], "surprise": 0, "ponder_steps": 0}}

    reply, avg_ponder, avg_surprise = generate_response(
        model_state["brain"],
        model_state["word_list"],
        model_state["vocab_matrix"],
        model_state["word_to_idx"],
        prompt_words,
        model_state["context_size"],
        engram_module=model_state.get("engram_module"),
        word_to_id=model_state.get("word_to_id"),
    )

    response_text = " ".join(reply) if reply else "..."
    return {
        "response": response_text,
        "stats": {
            "vocab_size": model_state["vocab_size"],
            "surprise": round(avg_surprise, 4),
            "ponder_steps": round(avg_ponder, 2),
        },
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/chat/stream")
async def chat_stream(message: str = Query(...)):
    if not model_state["loaded"]:
        async def _not_loaded():
            yield _sse("done", {"response": "Model still training, check back soon!",
                                "stats": {"vocab_size": 0, "surprise": 0, "ponder_steps": 0}})
        return StreamingResponse(_not_loaded(), media_type="text/event-stream")

    prompt_words = re.findall(r"\b\w+\b", message.lower())
    if not prompt_words:
        async def _empty():
            yield _sse("done", {"response": "...",
                                "stats": {"vocab_size": model_state["vocab_size"], "surprise": 0, "ponder_steps": 0}})
        return StreamingResponse(_empty(), media_type="text/event-stream")

    async def _generate():
        brain = model_state["brain"]
        word_list = model_state["word_list"]
        vocab_matrix = model_state["vocab_matrix"]
        word_to_idx = model_state["word_to_idx"]
        context_size = model_state["context_size"]
        embed_dim = vocab_matrix.shape[1]

        # ── Thought: memory query ─────────────────────────────────────────
        yield _sse("thought", {"type": "memory_query",
                                "content": f"Querying ChromaDB for: {message}"})
        await asyncio.sleep(0.05)

        # Actually query ChromaDB for relevant passages
        chroma_snippet = ""
        n_results = 0
        similarity = 0.0
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            vocab_col = chroma_client.get_collection("engram_vocab")
            results = vocab_col.query(query_texts=[message], n_results=3)
            if results and results["ids"] and results["ids"][0]:
                n_results = len(results["ids"][0])
                if results["distances"] and results["distances"][0]:
                    similarity = round(1.0 / (1.0 + results["distances"][0][0]), 2)
                chroma_snippet = ", ".join(results["ids"][0])[:80]
            del vocab_col
            try:
                chroma_client._producer = None
                chroma_client._consumer = None
            except Exception:
                pass
            del chroma_client
        except Exception:
            pass

        yield _sse("thought", {"type": "context_found",
                                "content": f"Found {n_results} relevant passages (similarity: {similarity:.2f})"
                                           + (f" — {chroma_snippet}" if chroma_snippet else "")})
        await asyncio.sleep(0.05)

        # ── Prepare generation context ────────────────────────────────────
        SKIP = {"<START>", "<BOT>", "<USER>"}
        context = ["<START>"] * context_size
        for w in (["<USER>"] + prompt_words + ["<BOT>"]):
            context.append(w)

        def get_embed(word):
            if word in word_to_idx:
                return vocab_matrix[word_to_idx[word]]
            return torch.zeros(embed_dim)

        reply = []
        recent = []
        ponder_steps_list = []
        surprise_list = []

        for step in range(GEN_STEPS):
            ctx_tensors = [get_embed(w) for w in context[-context_size:]]
            ctx_stack = torch.stack(ctx_tensors)

            with torch.no_grad():
                predicted, n_steps = brain(ctx_stack.unsqueeze(0))
                predicted = predicted.squeeze(0)

            ponder_steps_list.append(n_steps)

            # ── Thought: ponder steps ─────────────────────────────────────
            if step == 0:
                yield _sse("thought", {"type": "ponder",
                                        "content": f"Ponder step {n_steps}/{brain.max_ponder}: refining..."})
                await asyncio.sleep(0.03)

            penalty = {w: 3.0 for w in set(recent[-4:])}
            for tok in SKIP:
                penalty[tok] = float("inf")

            candidates = nearest_words(predicted, word_list, vocab_matrix, word_to_idx, penalty=penalty)
            filtered = [(w, d) for w, d in candidates if d < float("inf")]
            if not filtered:
                break

            # ── Thought: attention (top vocab tokens by proximity) ────────
            if step == 0:
                top5 = [w for w, _ in filtered[:5]]
                yield _sse("thought", {"type": "attention",
                                        "content": f"Top attended tokens: {', '.join(top5)}"})
                await asyncio.sleep(0.03)

            words_list_c = [w for w, _ in filtered]
            dists_t = torch.tensor([d for _, d in filtered])
            probs = F.softmax(-dists_t / TEMPERATURE, dim=-1)

            # ── Thought: vocab sampling ───────────────────────────────────
            if step == 0:
                yield _sse("thought", {"type": "vocab_sample",
                                        "content": f"Sampling from vocab distribution (top-k={TOP_K}, temp={TEMPERATURE})"})
                await asyncio.sleep(0.03)

            chosen_idx = torch.multinomial(probs, 1).item()
            chosen_word = words_list_c[chosen_idx]

            chosen_dist = filtered[chosen_idx][1]
            surprise_list.append(chosen_dist)

            if chosen_word == "<USER>":
                break

            reply.append(chosen_word)
            recent.append(chosen_word)
            context.append(chosen_word)

            # ── Stream token ──────────────────────────────────────────────
            yield _sse("token", {"token": chosen_word})
            await asyncio.sleep(0.02)

        avg_ponder = sum(ponder_steps_list) / max(len(ponder_steps_list), 1)
        avg_surprise = sum(surprise_list) / max(len(surprise_list), 1)
        response_text = " ".join(reply) if reply else "..."

        yield _sse("done", {
            "response": response_text,
            "stats": {
                "vocab_size": model_state["vocab_size"],
                "surprise": round(avg_surprise, 4),
                "ponder_steps": round(avg_ponder, 2),
            },
        })

    return StreamingResponse(_generate(), media_type="text/event-stream")
