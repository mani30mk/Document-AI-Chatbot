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


def find_keyword_chunks(query: str, chunks: list[dict], max_results: int = 4) -> list[dict]:
    """
    Find chunks containing exact keyword or phrase matches for terms in the query.
    Ensures acronyms (e.g. 'NI Score', 'GPT', 'API') and specific terms are never missed.
    """
    if not query or not chunks:
        return []

    stop_words = {
        "what", "is", "are", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "how", "why", "who", "which", "where", "can", "you",
        "tell", "me", "about", "explain", "describe", "define", "please", "does", "do"
    }
    words = [w for w in re.findall(r"\b[A-Za-z0-9_-]{2,}\b", query) if w.lower() not in stop_words]
    if not words:
        return []

    phrase = " ".join(words).lower()
    matched = []

    for c in chunks:
        content_lower = c.get("content", "").lower()
        score = 0
        if phrase and phrase in content_lower:
            score += 10
        for w in words:
            # Word boundary matching for exact terms and acronyms
            if re.search(r"\b" + re.escape(w.lower()) + r"\b", content_lower):
                score += (3 if w.isupper() else 1)
        if score > 0:
            matched.append((score, c))

    matched.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in matched[:max_results]]


def search_vectors(query_embedding: list[float], store: dict, k: int = 7, query_text: str = "") -> list[dict]:
    """Search in-memory vector store for top-k most similar documents with hybrid keyword boosting."""
    if not store or "embeddings" not in store or not store["embeddings"]:
        return []

    scored = []
    for i, emb in enumerate(store["embeddings"]):
        sim = cosine_similarity(query_embedding, emb)
        scored.append((sim, store["chunks"][i]))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_vec_chunks = [item[1] for item in scored[:k]]

    if query_text and store.get("chunks"):
        kw_chunks = find_keyword_chunks(query_text, store["chunks"], max_results=3)
        # Merge keyword chunks at front if not already present
        seen_contents = {c.get("content") for c in top_vec_chunks}
        for kc in kw_chunks:
            if kc.get("content") not in seen_contents:
                top_vec_chunks.insert(0, kc)
                seen_contents.add(kc.get("content"))

    return top_vec_chunks[:k]


def search_supabase_vectors(query_embedding: list[float], session_id: str, k: int = 7, query_text: str = "") -> list[dict]:
    """
    Search Supabase pgvector using the match_documents RPC function.
    Falls back to direct table query + python cosine similarity if RPC is not available.
    """
    supabase = get_supabase_client()
    if not supabase:
        return []

    retrieved = []
    # 1. Try match_documents stored procedure
    try:
        result = supabase.rpc("match_documents", {
            "query_embedding": query_embedding,
            "match_count": k,
            "filter": {"session_id": session_id},
        }).execute()
        if result.data:
            retrieved = [{"content": r.get("content", ""), "metadata": r.get("metadata", {})} for r in result.data]
    except Exception as err:
        print(f"Supabase RPC match_documents error ({err}), trying direct table query...")

    # 2. Fallback: Query documents table directly and compute cosine similarity in Python
    if not retrieved:
        try:
            res = (
                supabase.table("documents")
                .select("content, metadata, embedding")
                .eq("metadata->>session_id", session_id)
                .limit(200)
                .execute()
            )
            if res.data:
                scored = []
                for row in res.data:
                    vec = _parse_embedding(row.get("embedding"))
                    if vec:
                        sim = cosine_similarity(query_embedding, vec)
                        scored.append((sim, {"content": row.get("content", ""), "metadata": row.get("metadata", {})}))
                scored.sort(key=lambda x: x[0], reverse=True)
                retrieved = [item[1] for item in scored[:k]]
        except Exception as table_err:
            print(f"Supabase documents table fallback error ({table_err})")

    # 3. Hybrid keyword boosting
    if query_text and retrieved:
        kw_chunks = find_keyword_chunks(query_text, retrieved, max_results=3)
        seen_contents = {c.get("content") for c in retrieved}
        for kc in kw_chunks:
            if kc.get("content") not in seen_contents:
                retrieved.insert(0, kc)
                seen_contents.add(kc.get("content"))

    return retrieved[:k]


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
