"""
Chat and summarization router:
  - /ask: Question answering with RAG retrieval, multi-turn history, and YouTube recommendations
  - /summarize: Document summarization with 7-tier text fallback and centroid chunk selection
"""

import os
import re
import time
from typing import Optional, List

from fastapi import APIRouter, HTTPException

from models.chat import AskRequest, AskResponse, SummarizeRequest, YouTubeVideo
from config import (
    UPLOAD_DIR,
    RAG_SYSTEM_PROMPT,
    SUMMARIZE_PROMPT,
    get_supabase_client,
    get_gemini_api_keys,
    get_current_api_key,
    rotate_api_key,
    is_rate_limit_error,
    get_available_models,
    session_histories,
    session_files,
    session_docs,
    session_slides,
    session_vectorstores,
)
from services.embeddings import get_embeddings
from services.vector_search import (
    search_supabase_vectors,
    search_vectors,
    normalize_doc_name,
    select_representative_chunks,
)
from services.session_store import load_session, save_session
from services.llm import generate_chat, generate_text
from services.youtube import search_youtube
from services.document_loaders import load_file

router = APIRouter(tags=["Chat"])


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Ask a question. Uses RAG + multi-turn conversation history.
    """
    sid = req.session_id

    if sid not in session_histories:
        load_session(sid)
    if sid not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Retrieve relevant context via hybrid vector + keyword search
    retrieved_docs = []
    try:
        embeddings = get_embeddings()
        query_vec = embeddings.embed_query(req.question)

        # A. Try in-memory vector store first (fastest, 0ms)
        if sid in session_vectorstores:
            retrieved_docs = search_vectors(query_vec, session_vectorstores[sid], k=7, query_text=req.question)

        # B. If not in memory or insufficient results, query Supabase pgvector
        if not retrieved_docs or len(retrieved_docs) < 3:
            supabase = get_supabase_client()
            if supabase:
                sb_docs = search_supabase_vectors(query_vec, sid, k=7, query_text=req.question)
                seen = {d.get("content") for d in retrieved_docs}
                for d in sb_docs:
                    if d.get("content") not in seen:
                        retrieved_docs.append(d)
                        seen.add(d.get("content"))
    except Exception as r_err:
        print(f"Vector search error ({r_err}), using raw document text fallback.")

    context = (
        "\n\n".join(
            f"[{os.path.basename(d.get('metadata', {}).get('source', 'document'))}]: {d['content']}"
            for d in retrieved_docs[:7]
        )
        if retrieved_docs
        else ""
    )

    # Hybrid Keyword & Full-Text Augmentation
    # Ensures exact acronyms/keywords (e.g. 'NI Score') from session_docs are always in context
    if sid in session_docs and session_docs[sid]:
        extracted_sections = []
        for fname, full_text in session_docs[sid].items():
            if not full_text:
                continue
            # If the entire document is concise (<= 25,000 chars / ~15 pages), pass full document if context is thin
            if len(full_text) <= 25000 and len(context) < 3000:
                extracted_sections.append(f"[{fname} full text]:\n{full_text}")
            else:
                # Search for specific acronyms/phrases from the question in full_text
                keywords = [
                    w for w in re.findall(r"\b[A-Za-z0-9_-]{2,}\b", req.question)
                    if w.lower() not in {"what", "is", "are", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "how", "why", "who", "which", "where", "can", "you", "tell", "about", "explain", "describe", "define"}
                ]
                windows = []
                for kw in keywords:
                    for m in re.finditer(r"\b" + re.escape(kw) + r"\b", full_text, re.IGNORECASE):
                        start_pos = max(0, m.start() - 800)
                        end_pos = min(len(full_text), m.end() + 1200)
                        window_text = full_text[start_pos:end_pos].strip()
                        if not any(window_text in w or w in window_text for w in windows):
                            windows.append(window_text)
                        if len(windows) >= 3:
                            break
                    if len(windows) >= 3:
                        break
                if windows:
                    extracted_sections.append(f"[{fname} relevant sections]:\n" + "\n\n---\n\n".join(windows))

        if extracted_sections:
            extra_context = "\n\n".join(extracted_sections)
            if context:
                context += "\n\n" + extra_context
            else:
                context = extra_context

    # Final fallback if context is still empty
    if not context and sid in session_docs:
        context = "\n\n".join(
            f"[{fname}]: {text[:8000]}"
            for fname, text in session_docs[sid].items()
        )[:20000]

    models_to_try = get_available_models()[:5]
    keys = get_gemini_api_keys()
    if not keys:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in Render or in your .env file.",
        )

    # Build system prompt with context
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
    history = session_histories[sid]

    answer = None
    last_error = None
    start_time = time.monotonic()
    MAX_BUDGET_SECONDS = 35.0
    MIN_DEADLINE_S = 11.0  # Google's floor is 10s; small safety margin
    MAX_KEY_ATTEMPTS_PER_MODEL = 2

    for m_name in models_to_try:
        remaining = MAX_BUDGET_SECONDS - (time.monotonic() - start_time)
        if remaining < MIN_DEADLINE_S:
            print(f"Time budget remaining ({remaining:.1f}s < {MIN_DEADLINE_S:.1f}s floor), breaking model loop in /ask...")
            break
        for _ in range(min(MAX_KEY_ATTEMPTS_PER_MODEL, len(keys))):
            remaining = MAX_BUDGET_SECONDS - (time.monotonic() - start_time)
            if remaining < MIN_DEADLINE_S:
                print(f"Time budget remaining ({remaining:.1f}s < {MIN_DEADLINE_S:.1f}s floor), breaking key loop in /ask...")
                break
            current_key = get_current_api_key()
            try:
                answer = generate_chat(m_name, current_key, system_prompt, history, req.question, timeout_s=min(remaining, 15.0))
                if answer:
                    break
            except Exception as llm_err:
                last_error = llm_err
                if is_rate_limit_error(llm_err):
                    print(f"Transient error (quota/high demand) on model {m_name}: {llm_err}. Trying next...")
                    if len(keys) > 1:
                        rotate_api_key()
                        continue
                    else:
                        break  # Immediately try next candidate model
                else:
                    print(f"Model {m_name} failed ({llm_err}), falling back to next model candidate...")
                    break
        if answer:
            break

    if not answer:
        err_msg = f"AI model error: {str(last_error)}"
        if time.monotonic() - start_time >= MAX_BUDGET_SECONDS:
            err_msg = f"AI request timed out after {MAX_BUDGET_SECONDS:.0f}s: {str(last_error)}"
        raise HTTPException(status_code=500, detail=err_msg)

    yt_results = []
    try:
        yt_query = req.question
        if answer:
            yt_match = re.search(r"<!--\s*yt_search:\s*(.*?)\s*-->", answer, re.IGNORECASE)
            if yt_match:
                extracted = yt_match.group(1).strip()
                if extracted:
                    yt_query = extracted
                answer = re.sub(r"<!--\s*yt_search:\s*.*?\s*-->", "", answer).strip()
            else:
                clean_q = re.sub(r"^(what is|what are|explain|how to|describe|define)\s+", "", req.question, flags=re.I).strip(" ?.")
                doc_title = ""
                if sid in session_files and session_files[sid]:
                    doc_title = os.path.splitext(session_files[sid][0])[0].replace("_", " ").replace("-", " ")
                    doc_title = re.sub(r"\b(unit|chapter|notes|doc|file|pdf|part|presentation)\s*\d*\b", "", doc_title, flags=re.I).strip()
                if doc_title and len(clean_q) < 30:
                    yt_query = f"{doc_title} {clean_q}"
                else:
                    yt_query = clean_q

        yt_results = search_youtube(yt_query, max_results=3)
    except Exception as yt_err:
        print(f"YouTube search notice: {yt_err}")

    # Persist turn to history
    session_histories[sid].append({"role": "user", "content": req.question})
    session_histories[sid].append({"role": "bot", "content": answer})

    # Keep last 20 messages (10 turns) to avoid token bloat
    if len(session_histories[sid]) > 20:
        session_histories[sid] = session_histories[sid][-20:]

    save_session(sid)

    return AskResponse(
        answer=answer,
        session_id=sid,
        sources_used=len(retrieved_docs),
        youtube_sources=[YouTubeVideo(**v) for v in yt_results],
    )


@router.post("/summarize")
async def summarize(req: SummarizeRequest):
    """Summarize a specific uploaded file with multi-tier fallback."""
    sid = req.session_id
    fname = req.filename
    text = (req.text or "").strip()

    # Ensure session state is loaded from Supabase if not in memory
    if sid not in session_docs and sid not in session_slides:
        load_session(sid)

    # Tier 1: Client provided extracted text directly
    if not text:
        # Tier 2: Exact match in session_docs
        if sid in session_docs and fname in session_docs[sid]:
            text = session_docs[sid][fname]

    if not text:
        # Tier 3: Normalized filename match in session_docs[sid]
        target_norm = normalize_doc_name(fname)
        if sid in session_docs:
            for k, doc_text in session_docs[sid].items():
                if normalize_doc_name(k) == target_norm:
                    text = doc_text
                    break

    if not text:
        # Tier 4: Search across all sessions in session_docs
        target_norm = normalize_doc_name(fname)
        for s_id, doc_dict in session_docs.items():
            for k, doc_text in doc_dict.items():
                if normalize_doc_name(k) == target_norm:
                    text = doc_text
                    break
            if text:
                break

    if not text:
        # Tier 5: Check disk in UPLOAD_DIR for the file
        target_norm = normalize_doc_name(fname)
        found_file = None
        # Check session dir first
        session_dir = os.path.join(UPLOAD_DIR, sid)
        if os.path.isdir(session_dir):
            for disk_file in os.listdir(session_dir):
                if disk_file == fname or normalize_doc_name(disk_file) == target_norm:
                    found_file = os.path.join(session_dir, disk_file)
                    break

        # Check entire UPLOAD_DIR recursively if not found in session dir
        if not found_file and os.path.isdir(UPLOAD_DIR):
            for root, _, disk_files in os.walk(UPLOAD_DIR):
                for disk_file in disk_files:
                    if disk_file == fname or normalize_doc_name(disk_file) == target_norm:
                        found_file = os.path.join(root, disk_file)
                        break
                if found_file:
                    break

        if found_file:
            try:
                ext = os.path.splitext(found_file)[1].lower()
                docs, _ = load_file(found_file, ext, session_id=sid)
                if docs:
                    text = "\n\n".join(d["content"] for d in docs)
                    session_docs.setdefault(sid, {})[fname] = text
                    save_session(sid)
            except Exception as load_err:
                print(f"Notice: Could not load file from disk for summarize ({load_err})")

    if not text:
        # Tier 6: Single document fallback in session
        if sid in session_docs and len(session_docs[sid]) == 1:
            text = list(session_docs[sid].values())[0]

    if not text:
        # Tier 7: Check slides if PPT
        if sid in session_slides:
            slides_data = session_slides[sid].get(fname)
            if not slides_data:
                target_norm = normalize_doc_name(fname)
                for k, s_list in session_slides[sid].items():
                    if normalize_doc_name(k) == target_norm:
                        slides_data = s_list
                        break
            if slides_data:
                slide_parts = []
                for s in slides_data:
                    title = s.get("title", "")
                    raw = s.get("raw_text", "")
                    bullets = s.get("bullets", [])
                    bullet_text = "\n".join(
                        b if isinstance(b, str) else b.get("text", "") for b in bullets
                    )
                    part = f"Slide {s.get('slide_number', '')}: {title}\n{raw}\n{bullet_text}".strip()
                    if part:
                        slide_parts.append(part)
                if slide_parts:
                    text = "\n\n".join(slide_parts)

    if not text:
        raise HTTPException(
            status_code=404,
            detail=f"File '{fname}' not found in session and no text could be extracted.",
        )

    rep_chunks = select_representative_chunks(sid, fname)
    if rep_chunks:
        text = "\n\n---\n\n".join(rep_chunks)
        print(f"Summarize: using {len(rep_chunks)} representative chunks instead of full {len(text)}-char text.")

    prompt = SUMMARIZE_PROMPT.format(text=text[:20000])

    models_to_try = get_available_models()[:5]
    keys = get_gemini_api_keys()
    if not keys:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in Render or in your .env file.",
        )

    summary = None
    last_error = None
    start_time = time.monotonic()
    MAX_BUDGET_SECONDS = 35.0
    MIN_DEADLINE_S = 11.0  # Google's floor is 10s; small safety margin
    MAX_KEY_ATTEMPTS_PER_MODEL = 2

    for m_name in models_to_try:
        remaining = MAX_BUDGET_SECONDS - (time.monotonic() - start_time)
        if remaining < MIN_DEADLINE_S:
            print(f"Time budget remaining ({remaining:.1f}s < {MIN_DEADLINE_S:.1f}s floor), breaking model loop in /summarize...")
            break
        for _ in range(min(MAX_KEY_ATTEMPTS_PER_MODEL, len(keys))):
            remaining = MAX_BUDGET_SECONDS - (time.monotonic() - start_time)
            if remaining < MIN_DEADLINE_S:
                print(f"Time budget remaining ({remaining:.1f}s < {MIN_DEADLINE_S:.1f}s floor), breaking key loop in /summarize...")
                break
            current_key = get_current_api_key()
            try:
                summary = generate_text(m_name, current_key, prompt, timeout_s=min(remaining, 15.0))
                if summary:
                    break
            except Exception as e:
                last_error = e
                if is_rate_limit_error(e):
                    print(f"Summarize transient error (quota/high demand) on model {m_name}: {e}. Trying next...")
                    if len(keys) > 1:
                        rotate_api_key()
                        continue
                    else:
                        break  # Try next model candidate
                else:
                    print(f"Summarize model {m_name} failed ({e}), falling back to next model candidate...")
                    break
        if summary:
            break

    if not summary:
        err_msg = f"AI model error: {str(last_error)}"
        if time.monotonic() - start_time >= MAX_BUDGET_SECONDS:
            err_msg = f"AI summarization timed out after {MAX_BUDGET_SECONDS:.0f}s: {str(last_error)}"
        raise HTTPException(status_code=500, detail=err_msg)

    return {"summary": summary}
