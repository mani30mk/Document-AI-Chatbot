"""
Embedding services: Remote FastEmbed microservice (384-dim, 0 MB local RAM)
with Google Gemini embeddings fallback.
"""

import json
import time
import urllib.request
from typing import Optional, List

from google import genai
from config import EMBEDDING_SERVICE_URL, get_current_api_key


class RemoteEmbeddings:
    """Embeddings wrapper that calls the remote embedding service via HTTP."""

    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip("/")

    def embed_documents(self, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        """Embed a list of document texts via the remote service with warm-up retry and batching."""
        if not texts:
            return []

        all_embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = json.dumps({"texts": batch}).encode("utf-8")

            last_err = None
            for attempt in range(max_retries):
                try:
                    req = urllib.request.Request(
                        f"{self.service_url}/embed",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=40) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    all_embeddings.extend(data["embeddings"])
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    print(f"Remote embedding batch {i//batch_size + 1} attempt {attempt + 1}/{max_retries} failed ({e}). Waking up / retrying in 3s...")
                    time.sleep(3)

            if last_err is not None:
                raise last_err

        return all_embeddings

    def embed_query(self, text: str, max_retries: int = 4) -> list[float]:
        """Embed a single query text via the remote service with warm-up retry."""
        payload = json.dumps({"texts": [text]}).encode("utf-8")
        last_err = None
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    f"{self.service_url}/embed",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["embeddings"][0]
            except Exception as e:
                last_err = e
                print(f"Remote query embedding attempt {attempt + 1}/{max_retries} failed ({e}). Retrying in 4s...")
                time.sleep(4)
        raise last_err


class GeminiEmbeddings:
    """Lightweight embeddings using google-genai SDK directly (no LangChain)."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using Gemini embedding API with candidate model fallback."""
        candidate_models = [self.model, "gemini-embedding-001", "gemini-embedding-2"]
        unique_models = []
        for m in candidate_models:
            if m and m not in unique_models:
                unique_models.append(m)

        last_err = None
        for m in unique_models:
            try:
                all_embeddings = []
                for i in range(0, len(texts), 100):
                    batch = texts[i:i + 100]
                    result = self.client.models.embed_content(
                        model=m,
                        contents=batch,
                    )
                    all_embeddings.extend([list(e.values) for e in result.embeddings])
                self.model = m
                return all_embeddings
            except Exception as e:
                last_err = e
                print(f"Notice: Model {m} failed for embed_content ({e}), trying next candidate...")
                continue
        raise last_err

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.embed_documents([text])[0]


class UnifiedEmbeddings:
    """
    Dedicated FastEmbed microservice (384-dim, 0 MB local RAM, matches Supabase vector(384)).
    """

    def __init__(self, remote_url: Optional[str] = None, api_key: Optional[str] = None):
        self.primary_url = (remote_url or "https://document-ai-chatbot-7ch2.onrender.com").rstrip("/")
        self.verified_url = "https://document-ai-chatbot-7ch2.onrender.com"
        self._remote = RemoteEmbeddings(self.primary_url)
        self._gemini = GeminiEmbeddings(api_key) if api_key else None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._remote.embed_documents(texts)
        except Exception as e:
            print(f"Embedding service ({self.primary_url}) error: {e}")

        if self.primary_url != self.verified_url:
            try:
                print(f"Retrying with verified 2nd microservice: {self.verified_url}")
                return RemoteEmbeddings(self.verified_url).embed_documents(texts)
            except Exception as e:
                print(f"Verified 2nd microservice error: {e}")

        if self._gemini:
            try:
                return self._gemini.embed_documents(texts)
            except Exception as e:
                print(f"Gemini fallback embed error: {e}")

        raise ValueError("Could not embed documents. Please verify https://document-ai-chatbot-7ch2.onrender.com is reachable.")

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._remote.embed_query(text)
        except Exception as e:
            print(f"Embedding service query error ({self.primary_url}): {e}")

        if self.primary_url != self.verified_url:
            try:
                return RemoteEmbeddings(self.verified_url).embed_query(text)
            except Exception as e:
                print(f"Verified 2nd microservice query error: {e}")

        if self._gemini:
            try:
                return self._gemini.embed_query(text)
            except Exception as e:
                print(f"Gemini fallback query error: {e}")

        raise ValueError("Could not embed query. Please verify https://document-ai-chatbot-7ch2.onrender.com is reachable.")


_EMBEDDINGS: Optional[UnifiedEmbeddings] = None


def reset_embeddings():
    """Reset cached embeddings instance (e.g. after key rotation)."""
    global _EMBEDDINGS
    _EMBEDDINGS = None


def _get_embeddings_impl():
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    remote_url = EMBEDDING_SERVICE_URL or "https://document-ai-chatbot-7ch2.onrender.com"
    key = get_current_api_key()

    _EMBEDDINGS = UnifiedEmbeddings(remote_url=remote_url, api_key=key)
    return _EMBEDDINGS


def get_embeddings():
    """
    Returns an embeddings instance with fallback priority.
    Honors test mocks on main.get_embeddings if active.
    """
    import sys
    main_mod = sys.modules.get("main")
    if main_mod and hasattr(main_mod, "get_embeddings"):
        target = getattr(main_mod, "get_embeddings")
        if target is not get_embeddings:
            return target()
    return _get_embeddings_impl()
