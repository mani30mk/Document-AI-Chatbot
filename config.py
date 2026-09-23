"""
Configuration, environment variables, Supabase client initialization,
and in-memory session caches for Document AI.
"""

import gc
import json
import os
import socket
import urllib.request
from typing import Optional
from supabase.client import Client, create_client

# Force IPv4 socket resolution to prevent Render container IPv6 handshake/read timeouts to Google APIs
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0 or family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_getaddrinfo

# ── File paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
SESSIONS_FILE = os.path.join(UPLOAD_DIR, "sessions.json")

# ── In-memory caches ───────────────────────────────────────────────────────
# Capped to 1 active vectorstore to conserve Render 512MB RAM
MAX_CACHED_VECTORSTORES = 1
session_vectorstores: dict[str, dict] = {}  # sid -> {"chunks": [...], "embeddings": [...]}
session_histories: dict[str, list] = {}
session_files: dict[str, list[str]] = {}
session_docs: dict[str, dict[str, str]] = {}
session_slides: dict[str, dict[str, list]] = {}
session_api_keys: dict[str, str] = {}
session_device_ids: dict[str, str] = {}
session_updated_at: dict[str, str] = {}

# Bounded session cache — evict oldest when exceeding limit
MAX_CACHED_SESSIONS = 5
session_access_order: list[str] = []  # most-recent at end


def touch_session(session_id: str):
    """Mark a session as recently used (move to end of access order)."""
    if session_id in session_access_order:
        session_access_order.remove(session_id)
    session_access_order.append(session_id)


def evict_old_sessions():
    """Evict oldest sessions from in-memory dicts when cache exceeds MAX_CACHED_SESSIONS."""
    while len(session_access_order) > MAX_CACHED_SESSIONS:
        oldest = session_access_order.pop(0)
        session_histories.pop(oldest, None)
        session_files.pop(oldest, None)
        session_docs.pop(oldest, None)
        session_slides.pop(oldest, None)
        session_api_keys.pop(oldest, None)
        session_vectorstores.pop(oldest, None)
        session_device_ids.pop(oldest, None)
        session_updated_at.pop(oldest, None)
        gc.collect()
        print(f"Evicted session {oldest[:8]}... from in-memory cache (will re-hydrate from Supabase on demand).")


# ── Remote embedding service URL ───────────────────────────────────────────
_env_emb_url = os.getenv("EMBEDDING_SERVICE_URL", "").rstrip("/")
if not _env_emb_url or "document-ai-embeddings" in _env_emb_url:
    EMBEDDING_SERVICE_URL = "https://document-ai-chatbot-7ch2.onrender.com"
else:
    EMBEDDING_SERVICE_URL = _env_emb_url

# ── Supabase (persistent cloud storage & pgvector) ─────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
supabase_client: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase for persistent cloud storage & pgvector.")
    except Exception as e:
        print(f"Warning: Could not connect to Supabase: {e}")


def get_supabase_client() -> Optional[Client]:
    """
    Get current Supabase client.
    Checks main.py if dynamically mocked in unit tests, falling back to module client.
    """
    import sys
    main_mod = sys.modules.get("main")
    if main_mod and hasattr(main_mod, "supabase_client"):
        return getattr(main_mod, "supabase_client")
    return supabase_client


# ── Admin cleanup key ───────────────────────────────────────────────────────
ADMIN_CLEANUP_KEY = os.getenv("ADMIN_CLEANUP_KEY", "")


def get_admin_cleanup_key() -> str:
    import sys
    main_mod = sys.modules.get("main")
    if main_mod and hasattr(main_mod, "ADMIN_CLEANUP_KEY"):
        val = getattr(main_mod, "ADMIN_CLEANUP_KEY")
        if val:
            return val
    return os.getenv("ADMIN_CLEANUP_KEY", "") or ADMIN_CLEANUP_KEY


# ── Prompts ────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a helpful study assistant. Answer questions using the document context below when possible.
Be concise, clear, and structure your answer with headings and bullet points where appropriate.
If the answer isn't in the context, say so honestly.

At the very end of your response, on a new line, include a tag with the 2 to 4 word specific academic/technical subject for searching educational lecture videos:
<!-- yt_search: <academic subject keywords> -->

Context:
{context}"""

SUMMARIZE_PROMPT = "You are an expert summarizer. Please provide a comprehensive and concise summary of the following document:\n\n{text}"


# ── Gemini API Keys Management & Rotation ──────────────────────────────────
_current_key_index: int = 0
AVAILABLE_GEMINI_MODELS: list[str] = []


def get_gemini_api_keys() -> list[str]:
    """
    Retrieve Gemini API keys from GEMINI_API_KEY environment variable.
    Supports comma-separated, semicolon-separated, or newline-separated multiple keys.
    """
    raw = os.getenv("GEMINI_API_KEY", "")
    if not raw:
        return []
    keys = [
        k.strip()
        for k in raw.replace("\n", ",").replace(";", ",").split(",")
        if k.strip()
    ]
    return keys


def get_current_api_key() -> Optional[str]:
    """Get the currently active API key."""
    keys = get_gemini_api_keys()
    if not keys:
        return None
    global _current_key_index
    return keys[_current_key_index % len(keys)]


def rotate_api_key() -> Optional[str]:
    """Rotate to the next API key in round-robin fashion. Resets cached embeddings."""
    global _current_key_index
    keys = get_gemini_api_keys()
    if not keys:
        return None
    _current_key_index = (_current_key_index + 1) % len(keys)

    # Invalidate cached embeddings
    try:
        from services.embeddings import reset_embeddings
        reset_embeddings()
    except ImportError:
        pass

    masked = keys[_current_key_index][:6] + "..." + keys[_current_key_index][-4:] if len(keys[_current_key_index]) > 10 else "***"
    print(f"Rotated to API key #{_current_key_index + 1}/{len(keys)} ({masked})")
    return keys[_current_key_index]


def is_rate_limit_error(exc: Exception) -> bool:
    """Detect if an exception is due to quota exhaustion, high demand, or transient network timeouts."""
    msg = str(exc).lower()
    return any(
        s in msg
        for s in [
            "429",
            "resource_exhausted",
            "resourceexhausted",
            "quota",
            "rate limit",
            "ratelimit",
            "too many requests",
            "503",
            "high demand",
            "overloaded",
            "temporarily unavailable",
            "service unavailable",
            "capacity",
            "timed out",
            "timeout",
            "handshake",
            "ssl",
            "connection error",
            "connection reset",
        ]
    )


def get_available_models(api_key: Optional[str] = None) -> list[str]:
    """Query Google API for active models, prioritizing reliable production models first."""
    global AVAILABLE_GEMINI_MODELS
    if AVAILABLE_GEMINI_MODELS:
        return AVAILABLE_GEMINI_MODELS

    candidates = [
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.0-flash",
    ]

    keys = [api_key] if api_key else get_gemini_api_keys()
    if not keys:
        non_pro = [m for m in candidates if "pro" not in m.lower()]
        pro = [m for m in candidates if "pro" in m.lower()]
        return non_pro + pro

    for key in keys:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
            req = urllib.request.Request(url, headers={"User-Agent": "DocumentAI/1.0"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                discovered = []
                for m in models:
                    methods = m.get("supportedGenerationMethods", [])
                    name = m.get("name", "").replace("models/", "")
                    name_lower = name.lower()
                    if (
                        "generateContent" in methods
                        and "gemini" in name_lower
                        and not any(x in name_lower for x in ["tts", "embed", "imagen", "robotics", "computer-use", "thinking", "preview", "exp"])
                    ):
                        discovered.append(name)

                # Prioritize: fast flash models first, then other stable models
                preferred_order = [
                    "gemini-2.5-flash",
                    "gemini-flash-latest",
                    "gemini-flash-lite-latest",
                    "gemini-2.0-flash",
                ]
                sorted_models = []
                for p in preferred_order:
                    if p in discovered and p not in sorted_models:
                        sorted_models.append(p)
                for d in discovered:
                    if d not in sorted_models:
                        sorted_models.append(d)
                for c in candidates:
                    if c not in sorted_models:
                        sorted_models.append(c)

                # Push any "pro" models to the end
                non_pro = [m for m in sorted_models if "pro" not in m.lower()]
                pro = [m for m in sorted_models if "pro" in m.lower()]
                sorted_models = non_pro + pro

                if sorted_models:
                    AVAILABLE_GEMINI_MODELS = sorted_models
                    print(f"Discovered {len(AVAILABLE_GEMINI_MODELS)} available Gemini models: {AVAILABLE_GEMINI_MODELS[:5]}")
                    return AVAILABLE_GEMINI_MODELS
        except Exception as e:
            print(f"Notice: Model discovery via API key ({e}), trying next candidate/defaults.")
            continue

    non_pro = [m for m in candidates if "pro" not in m.lower()]
    pro = [m for m in candidates if "pro" in m.lower()]
    AVAILABLE_GEMINI_MODELS = non_pro + pro
    return AVAILABLE_GEMINI_MODELS
