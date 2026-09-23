"""
RAG Study Assistant — FastAPI Backend (Modular Lightweight Edition)
Entry point for Render: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── Import configuration and state ──────────────────────────────────────────
import config
from config import (
    UPLOAD_DIR,
    SESSIONS_FILE,
    MAX_CACHED_VECTORSTORES,
    MAX_CACHED_SESSIONS,
    EMBEDDING_SERVICE_URL,
    ADMIN_CLEANUP_KEY,
    supabase_client,
    session_vectorstores,
    session_histories,
    session_files,
    session_docs,
    session_slides,
    session_api_keys,
    session_device_ids,
    session_updated_at,
    session_access_order,
    touch_session,
    evict_old_sessions,
    get_gemini_api_keys,
    get_current_api_key,
    rotate_api_key,
    is_rate_limit_error,
    get_available_models,
    RAG_SYSTEM_PROMPT,
    SUMMARIZE_PROMPT,
)

# ── Import services ────────────────────────────────────────────────────────
from services.document_loaders import (
    split_text,
    load_docx,
    load_pdf,
    load_txt,
    load_pptx,
    load_file,
    extract_shape_images,
)
from services.embeddings import (
    RemoteEmbeddings,
    GeminiEmbeddings,
    UnifiedEmbeddings,
    get_embeddings,
    reset_embeddings,
)
from services.vector_search import (
    cosine_similarity,
    search_vectors,
    search_supabase_vectors,
    select_representative_chunks,
    normalize_doc_name,
)
from services.session_store import (
    load_sessions_from_disk,
    save_sessions_to_disk,
    load_session,
    save_session,
)
from services.llm import generate_chat, generate_text
from services.youtube import search_youtube

# ── Import routers ─────────────────────────────────────────────────────────
from routers import sessions_router, documents_router, chat_router

# ── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Study Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS & JS)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include modular routers
app.include_router(sessions_router)
app.include_router(documents_router)
app.include_router(chat_router)


@app.api_route("/", methods=["GET", "HEAD"])
def serve_index():
    """Serve the frontend single page app."""
    return FileResponse("index.html")


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """Health check endpoint for Render."""
    return {"status": "ok"}


@app.on_event("startup")
def startup_prewarm():
    """Load sessions on startup."""
    load_sessions_from_disk()
    embed_mode = "remote service" if EMBEDDING_SERVICE_URL else ("cloud API" if get_gemini_api_keys() else "none")
    print(f"Startup complete. Embeddings: {embed_mode}.")


# ── Re-exports for backward compatibility with scripts and unit tests ───────
__all__ = [
    "app",
    "UPLOAD_DIR",
    "SESSIONS_FILE",
    "MAX_CACHED_VECTORSTORES",
    "MAX_CACHED_SESSIONS",
    "EMBEDDING_SERVICE_URL",
    "ADMIN_CLEANUP_KEY",
    "supabase_client",
    "session_vectorstores",
    "session_histories",
    "session_files",
    "session_docs",
    "session_slides",
    "session_api_keys",
    "session_device_ids",
    "session_updated_at",
    "session_access_order",
    "touch_session",
    "evict_old_sessions",
    "get_gemini_api_keys",
    "get_current_api_key",
    "rotate_api_key",
    "is_rate_limit_error",
    "get_available_models",
    "RAG_SYSTEM_PROMPT",
    "SUMMARIZE_PROMPT",
    "split_text",
    "load_docx",
    "load_pdf",
    "load_txt",
    "load_pptx",
    "load_file",
    "extract_shape_images",
    "RemoteEmbeddings",
    "GeminiEmbeddings",
    "UnifiedEmbeddings",
    "get_embeddings",
    "reset_embeddings",
    "cosine_similarity",
    "search_vectors",
    "search_supabase_vectors",
    "select_representative_chunks",
    "normalize_doc_name",
    "load_sessions_from_disk",
    "save_sessions_to_disk",
    "load_session",
    "save_session",
    "generate_chat",
    "generate_text",
    "search_youtube",
]
