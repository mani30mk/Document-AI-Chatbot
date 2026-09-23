"""
Session management router:
  - Create new sessions
  - List session history for a device
  - Retrieve session state
  - Delete individual sessions (manual delete button)
  - Clear chat history
  - Cascading admin session cleanup
"""

import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List

from fastapi import APIRouter, HTTPException

from models.session import NewSessionRequest, SessionResponse, SessionSummary
from config import (
    get_supabase_client,
    get_admin_cleanup_key,
    session_device_ids,
    session_files,
    session_histories,
    session_updated_at,
    session_docs,
    session_slides,
    session_vectorstores,
    session_access_order,
)
from services.session_store import save_session, load_session

router = APIRouter(tags=["Sessions"])


@router.post("/session/new", response_model=SessionResponse)
def new_session(req: Optional[NewSessionRequest] = None):
    """Create a new chat/upload session."""
    sid = str(uuid.uuid4())
    dev_id = req.device_id if req else None
    if dev_id:
        session_device_ids[sid] = dev_id
    save_session(sid, files=[], docs={}, slides={}, history=[], device_id=dev_id)
    return SessionResponse(session_id=sid, files=[])


@router.get("/sessions", response_model=List[SessionSummary])
def list_sessions(device_id: str):
    """List past sessions for a given device_id, newest-active first."""
    supabase = get_supabase_client()
    if not supabase:
        summaries = []
        for sid, dev_id in session_device_ids.items():
            if dev_id == device_id:
                files = session_files.get(sid, [])
                history = session_histories.get(sid, [])
                if files:
                    title = files[0]
                elif history:
                    first_user_msg = next((h.get("content", "") for h in history if h.get("role") == "user"), "")
                    title = (first_user_msg[:60] + "…") if len(first_user_msg) > 60 else (first_user_msg or "New chat")
                else:
                    title = "New chat"
                summaries.append(SessionSummary(
                    session_id=sid,
                    title=title,
                    updated_at=session_updated_at.get(sid, datetime.now(timezone.utc).isoformat()),
                ))
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries

    try:
        res = (
            supabase.table("chat_sessions")
            .select("session_id, files, history, updated_at")
            .eq("device_id", device_id)
            .order("updated_at", desc=True)
            .limit(50)
            .execute()
        )
    except Exception as e:
        print(f"Notice: list_sessions query failed ({e})")
        return []

    summaries = []
    for row in (res.data or []):
        files = row.get("files") or []
        history = row.get("history") or []
        if files:
            title = files[0]
        elif history:
            first_user_msg = next((h.get("content", "") for h in history if h.get("role") == "user"), "")
            title = (first_user_msg[:60] + "…") if len(first_user_msg) > 60 else (first_user_msg or "New chat")
        else:
            title = "New chat"
        summaries.append(SessionSummary(
            session_id=row["session_id"],
            title=title,
            updated_at=row.get("updated_at") or "",
        ))
    return summaries


@router.delete("/admin/sessions/cleanup")
def cleanup_old_sessions(older_than_days: int = 5, key: str = ""):
    """Delete sessions inactive for more than `older_than_days`, cascading to
    their Supabase Storage files and vector rows. Protected by ADMIN_CLEANUP_KEY."""
    admin_key = get_admin_cleanup_key()
    if not admin_key or key != admin_key:
        raise HTTPException(status_code=403, detail="Invalid or missing admin key.")
    
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")

    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()

    try:
        old_sessions = (
            supabase.table("chat_sessions")
            .select("session_id")
            .lt("updated_at", cutoff)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query old sessions: {e}")

    deleted_count = 0
    errors = []
    for row in (old_sessions.data or []):
        sid = row["session_id"]
        try:
            # 1. Delete vector/document rows for this session
            supabase.table("documents").delete().eq("metadata->>session_id", sid).execute()
            # 2. Delete any Storage files under this session's folder
            try:
                files_in_bucket = supabase.storage.from_("documents").list(sid)
                if files_in_bucket:
                    paths = [f"{sid}/{f['name']}" for f in files_in_bucket if isinstance(f, dict) and f.get('name')]
                    if paths:
                        supabase.storage.from_("documents").remove(paths)
            except Exception as storage_err:
                print(f"Notice: storage cleanup for session {sid} failed ({storage_err})")
            # 3. Delete the session row itself
            supabase.table("chat_sessions").delete().eq("session_id", sid).execute()
            # 4. Evict from in-memory caches too, if present
            for d in (session_histories, session_files, session_docs, session_slides, session_vectorstores, session_device_ids, session_updated_at):
                d.pop(sid, None)
            if sid in session_access_order:
                session_access_order.remove(sid)
            deleted_count += 1
        except Exception as e:
            errors.append(f"{sid}: {e}")

    print(f"Session cleanup: deleted {deleted_count} sessions older than {older_than_days} days.")
    return {"deleted": deleted_count, "errors": errors}


@router.get("/session/{session_id}")
def get_session_state(session_id: str):
    """Retrieve full session state (files, slides, docs, chat history) for page reload."""
    sess = load_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")

    history_list = sess.get("history", [])

    return {
        "session_id": session_id,
        "files": sess.get("files", []),
        "slides": sess.get("slides", {}),
        "docs": sess.get("docs", {}),
        "history": history_list,
        "history_turns": len(history_list) // 2,
    }


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a single session: its chat_sessions row, vector/document rows,
    and any Supabase Storage files, plus evict it from in-memory caches."""
    deleted_from_db = False
    supabase = get_supabase_client()

    if supabase:
        try:
            supabase.table("documents").delete().eq("metadata->>session_id", session_id).execute()
        except Exception as e:
            print(f"Notice: documents cleanup for session {session_id} failed ({e})")

        try:
            files_in_bucket = supabase.storage.from_("documents").list(session_id)
            if files_in_bucket:
                paths = [f"{session_id}/{f['name']}" for f in files_in_bucket if isinstance(f, dict) and f.get('name')]
                if paths:
                    supabase.storage.from_("documents").remove(paths)
        except Exception as e:
            print(f"Notice: storage cleanup for session {session_id} failed ({e})")

        try:
            res = supabase.table("chat_sessions").delete().eq("session_id", session_id).execute()
            deleted_from_db = bool(res.data)
        except Exception as e:
            print(f"Notice: chat_sessions delete for session {session_id} failed ({e})")

    for d in (session_histories, session_files, session_docs, session_slides, session_vectorstores, session_device_ids, session_updated_at):
        d.pop(session_id, None)
    if session_id in session_access_order:
        session_access_order.remove(session_id)

    if not deleted_from_db and not supabase:
        # Dev fallback: no Supabase configured, already evicted from in-memory above
        pass

    return {"deleted": session_id}


@router.delete("/session/{session_id}/history")
def clear_history(session_id: str):
    """Clear conversation history for a session (keeps files/vectorstore)."""
    if session_id not in session_histories:
        load_session(session_id)
    if session_id not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found.")
    session_histories[session_id] = []
    save_session(session_id)
    return {"message": "Conversation history cleared."}
