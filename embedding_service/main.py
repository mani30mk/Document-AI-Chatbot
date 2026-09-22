"""
Embedding Microservice — FastAPI + FastEmbed (ONNX)

Standalone service that runs BAAI/bge-small-en-v1.5 locally via ONNX runtime.
Outputs 384-dimensional embeddings, matching Supabase vector(384) column.

Designed for Render free tier (~195 MB RAM: FastAPI 45 MB + ONNX model 150 MB).

Endpoints:
  GET  /health  → 200 OK (wake-up target for warm-on-demand pattern)
  POST /embed   → accepts {"texts": [...]} → returns {"embeddings": [[...], ...]}
"""

import os
import time
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Document AI Embedding Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded embedding model ─────────────────────────────────────────────
_embeddings = None


def get_model():
    """Lazy-load FastEmbedEmbeddings on first use to avoid import-time RAM spike."""
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import FastEmbedEmbeddings
        _embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        print("FastEmbed model loaded (BAAI/bge-small-en-v1.5, 384-dim).")
    return _embeddings


# ── Request / Response models ────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    count: int


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """Lightweight wake-up endpoint. Returns 200 OK immediately."""
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """Embed a list of text strings using local FastEmbed ONNX model."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided.")

    if len(req.texts) > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Too many texts ({len(req.texts)}). Max 500 per request.",
        )

    start = time.monotonic()
    try:
        model = get_model()
        vectors = model.embed_documents(req.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    elapsed = time.monotonic() - start
    dims = len(vectors[0]) if vectors else 0
    print(f"Embedded {len(req.texts)} texts → {dims}-dim in {elapsed:.2f}s")

    return EmbedResponse(
        embeddings=vectors,
        dimensions=dims,
        count=len(vectors),
    )


# ── Startup ──────────────────────────────────────────────────────────────────
import threading


@app.on_event("startup")
def startup_prewarm():
    """Pre-warm the embedding model in a background thread on boot."""
    def _warm():
        try:
            get_model()
            print("Embedding model pre-warmed successfully.")
        except Exception as e:
            print(f"Notice: Pre-warm failed ({e})")
    threading.Thread(target=_warm, daemon=True).start()
