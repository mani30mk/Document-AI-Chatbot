"""
Session state persistence across Supabase chat_sessions (production)
and local JSON disk storage (dev fallback).
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

from config import (
    SESSIONS_FILE,
    get_supabase_client,
    touch_session,
    evict_old_sessions,
    session_histories,
    session_files,
    session_docs,
    session_slides,
    session_device_ids,
    session_updated_at,
)


def load_sessions_from_disk():
    """Dev fallback: Load sessions from local JSON file (not used in production with Supabase)."""
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                session_files.update(data.get("files", {}))
                session_docs.update(data.get("docs", {}))
                session_slides.update(data.get("slides", {}))
                for sid, msgs in data.get("histories", {}).items():
                    session_histories[sid] = [
                        {"role": m["role"], "content": m["content"]}
                        for m in msgs
                    ]
            print(f"Loaded {len(session_files)} sessions from disk.")
        except Exception as e:
            print(f"Notice: Could not load sessions from disk ({e})")


def save_sessions_to_disk():
    """Dev fallback: Save sessions to local JSON file (not used in production with Supabase)."""
    try:
        data = {
            "files": session_files,
            "docs": session_docs,
            "slides": session_slides,
            "histories": {
                sid: [
                    {"role": m["role"], "content": m["content"]}
                    for m in msgs
                ]
                for sid, msgs in session_histories.items()
            }
        }
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Notice: Could not save sessions to disk ({e})")


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a session by session_id from Supabase (production) or local JSON (dev fallback).
    Hydrates in-memory dicts and returns a dict with the session state, or None if not found.
    """
    has_mem = (
        session_id in session_histories
        or session_id in session_files
        or session_id in session_docs
        or session_id in session_slides
    )

    supabase = get_supabase_client()

    # 1. If Supabase is connected, fetch from chat_sessions table
    if supabase:
        try:
            res = (
                supabase.table("chat_sessions")
                .select("*")
                .eq("session_id", session_id)
                .execute()
            )
            if res.data and len(res.data) > 0:
                row = res.data[0]
                files = row.get("files") or []
                docs = row.get("docs") or {}
                slides = row.get("slides") or {}
                raw_history = row.get("history") or []

                # Hydrate in-memory request-scoped cache
                session_files[session_id] = files
                session_docs[session_id] = docs
                session_slides[session_id] = slides
                if row.get("device_id"):
                    session_device_ids[session_id] = row["device_id"]
                if row.get("updated_at"):
                    session_updated_at[session_id] = row["updated_at"]
                session_histories[session_id] = [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in raw_history
                    if isinstance(m, dict)
                ]
                return {
                    "session_id": session_id,
                    "files": files,
                    "docs": docs,
                    "slides": slides,
                    "history": raw_history,
                }
            elif has_mem:
                raw_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in session_histories.get(session_id, [])
                ]
                return {
                    "session_id": session_id,
                    "files": session_files.get(session_id, []),
                    "docs": session_docs.get(session_id, {}),
                    "slides": session_slides.get(session_id, {}),
                    "history": raw_history,
                }
            return None
        except Exception as err:
            print(f"Notice: Supabase load_session error for {session_id} ({err}), trying dev fallback...")

    # 2. Dev fallback: check in-memory or load from local JSON disk file
    if not has_mem:
        load_sessions_from_disk()
        has_mem = (
            session_id in session_histories
            or session_id in session_files
            or session_id in session_docs
            or session_id in session_slides
        )

    if has_mem:
        raw_history = [
            {"role": m["role"], "content": m["content"]}
            for m in session_histories.get(session_id, [])
        ]
        return {
            "session_id": session_id,
            "files": session_files.get(session_id, []),
            "docs": session_docs.get(session_id, {}),
            "slides": session_slides.get(session_id, {}),
            "history": raw_history,
        }

    return None


def save_session(
    session_id: str,
    files: Optional[List[str]] = None,
    docs: Optional[Dict[str, str]] = None,
    slides: Optional[Dict[str, list]] = None,
    history: Optional[List] = None,
    device_id: Optional[str] = None,
):
    """
    Persist session state to Supabase (production) or local JSON (dev fallback).
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    session_updated_at[session_id] = now_iso

    # Update in-memory dicts
    if files is not None:
        session_files[session_id] = files
    if docs is not None:
        session_docs[session_id] = docs
    if slides is not None:
        session_slides[session_id] = slides
    if history is not None:
        session_histories[session_id] = history
    if device_id:
        session_device_ids[session_id] = device_id

    cur_files = session_files.get(session_id, [])
    cur_docs = session_docs.get(session_id, {})
    cur_slides = session_slides.get(session_id, {})
    cur_msgs = session_histories.get(session_id, [])
    cur_device_id = device_id or session_device_ids.get(session_id)

    history_json = [
        {"role": m["role"], "content": m["content"]}
        for m in cur_msgs
    ]

    supabase = get_supabase_client()

    # 1. Primary: Persist to Supabase chat_sessions table
    if supabase:
        try:
            payload = {
                "session_id": session_id,
                "files": cur_files,
                "docs": cur_docs,
                "slides": cur_slides,
                "history": history_json,
                "updated_at": now_iso,
            }
            if cur_device_id:
                payload["device_id"] = cur_device_id
            supabase.table("chat_sessions").upsert(payload).execute()
            # Evict old sessions after successful Supabase persist
            touch_session(session_id)
            evict_old_sessions()
            return
        except Exception as err:
            print(f"Notice: Supabase save_session error ({err}), falling back to disk...")

    # 2. Dev fallback: Save to local JSON disk file (not used in production)
    touch_session(session_id)
    save_sessions_to_disk()
