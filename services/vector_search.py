"""
Vector similarity search (cosine distance in-memory and Supabase pgvector RPC),
along with centroid-based representative chunk selection for document summarization.
"""

import ast
import math
import os
import re
from typing import Optional, List, Dict

from config import get_supabase_client, session_vectorstores


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_vectors(query_embedding: list[float], store: dict, k: int = 5) -> list[dict]:
    """Search in-memory vector store for top-k most similar documents."""
    if not store or "embeddings" not in store or not store["embeddings"]:
        return []
    scored = []
    for i, emb in enumerate(store["embeddings"]):
        sim = cosine_similarity(query_embedding, emb)
        scored.append((sim, store["chunks"][i]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:k]]


def search_supabase_vectors(query_embedding: list[float], session_id: str, k: int = 5) -> list[dict]:
    """Search Supabase pgvector using the match_documents RPC function."""
    supabase = get_supabase_client()
    if not supabase:
        return []
    try:
        result = supabase.rpc("match_documents", {
            "query_embedding": query_embedding,
            "match_count": k,
            "filter": {"session_id": session_id},
        }).execute()
        if result.data:
            return [{"content": r.get("content", ""), "metadata": r.get("metadata", {})} for r in result.data]
    except Exception as err:
        print(f"Supabase vector search error ({err})")
    return []


def normalize_doc_name(name: str) -> str:
    """Normalize filename for fuzzy matching (case, hyphens, underscores, spaces)."""
    return re.sub(r"[\s\-_]+", "", name.lower())


def _parse_embedding(raw) -> Optional[List[float]]:
    """Supabase/postgrest can return pgvector columns as a string like
    '[0.1,-0.2,...]' instead of a parsed list — normalize either form to floats."""
    if isinstance(raw, str):
        try:
            return [float(x) for x in ast.literal_eval(raw)]
        except Exception:
            try:
                return [float(x) for x in raw.strip("[]").split(",") if x.strip()]
            except Exception:
                return None
    if isinstance(raw, list):
        return [float(x) for x in raw]
    return None


def select_representative_chunks(session_id: str, filename: str, max_chunks: int = 10) -> Optional[List[str]]:
    """
    Pick representative chunks for a document using its already-computed embeddings,
    by choosing the chunks closest to the centroid of that document's chunk embeddings.
    Returns None if too few chunks exist (caller falls back to tiered text logic).
    """
    chunks_with_vecs: list[tuple[str, list[float]]] = []

    # 1. In-memory fallback store
    store = session_vectorstores.get(session_id)
    if store and store.get("chunks") and store.get("embeddings"):
        for chunk, raw_vec in zip(store["chunks"], store["embeddings"]):
            source = chunk.get("metadata", {}).get("source", "")
            if os.path.basename(source) == filename or normalize_doc_name(os.path.basename(source)) == normalize_doc_name(filename):
                vec = _parse_embedding(raw_vec)
                if vec:
                    chunks_with_vecs.append((chunk["content"], vec))

    # 2. Supabase pgvector, if nothing found in-memory
    supabase = get_supabase_client()
    if not chunks_with_vecs and supabase:
        try:
            res = (
                supabase.table("documents")
                .select("content, metadata, embedding")
                .eq("metadata->>session_id", session_id)
                .execute()
            )
            for row in (res.data or []):
                source = (row.get("metadata") or {}).get("source", "")
                if os.path.basename(source) == filename or normalize_doc_name(os.path.basename(source)) == normalize_doc_name(filename):
                    vec = _parse_embedding(row.get("embedding"))
                    if vec:
                        chunks_with_vecs.append((row["content"], vec))
        except Exception as e:
            print(f"Notice: Supabase chunk lookup for summarize failed ({e})")

    if len(chunks_with_vecs) <= max_chunks:
        return None  # too few chunks to bother — let existing tiered text logic handle it

    contents = [c for c, _ in chunks_with_vecs]
    vecs = [v for _, v in chunks_with_vecs]
    dim = len(vecs[0])
    centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
    distances = [
        math.sqrt(sum((v[i] - centroid[i]) ** 2 for i in range(dim)))
        for v in vecs
    ]
    top_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:max_chunks]
    top_indices_sorted = sorted(top_indices)  # preserve original document order for readability
    return [contents[i] for i in top_indices_sorted]
