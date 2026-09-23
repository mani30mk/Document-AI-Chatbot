"""
Document handling router:
  - File upload and parsing (PDF, DOCX, PPTX, TXT)
  - Raw document retrieval / download / preview
  - Slide image retrieval
  - Slide data retrieval
  - Document text retrieval
"""

import gc
import os
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from config import (
    UPLOAD_DIR,
    MAX_CACHED_VECTORSTORES,
    get_supabase_client,
    get_gemini_api_keys,
    rotate_api_key,
    is_rate_limit_error,
    session_histories,
    session_files,
    session_docs,
    session_slides,
    session_vectorstores,
)
from services.document_loaders import load_file, split_text
from services.embeddings import get_embeddings
from services.session_store import load_session, save_session

router = APIRouter(tags=["Documents"])


@router.get("/raw/{session_id}/{filename}")
def get_raw_file(session_id: str, filename: str):
    """Serve original uploaded file directly for Office Viewer or download."""
    file_path = os.path.join(UPLOAD_DIR, session_id, filename)
    if os.path.isfile(file_path):
        media_type = None
        lower_name = filename.lower()
        if lower_name.endswith((".pptx", ".ppt")):
            media_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        elif lower_name.endswith(".pdf"):
            media_type = "application/pdf"
        elif lower_name.endswith((".docx", ".doc")):
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif lower_name.endswith((".xlsx", ".xls")):
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif lower_name.endswith((".txt", ".csv", ".md", ".py", ".json", ".log")):
            media_type = "text/plain; charset=utf-8"
        return FileResponse(
            file_path,
            filename=filename,
            media_type=media_type,
            content_disposition_type="inline",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "public, max-age=3600",
            },
        )

    supabase = get_supabase_client()
    if supabase:
        try:
            pub_url = supabase.storage.from_("documents").get_public_url(f"{session_id}/{filename}")
            if pub_url:
                return RedirectResponse(pub_url)
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="File not found.")


@router.get("/slide_image/{session_id}/{filename}")
def get_slide_image(session_id: str, filename: str):
    """Serve slide images extracted from PPTX presentations."""
    clean_session_id = os.path.basename(session_id)
    clean_filename = os.path.basename(filename)

    expected_dir = os.path.abspath(os.path.join(UPLOAD_DIR, clean_session_id, "slide_images"))
    img_path = os.path.abspath(os.path.join(expected_dir, clean_filename))

    if not img_path.startswith(expected_dir) or not os.path.isfile(img_path):
        raise HTTPException(status_code=404, detail="Slide image not found.")

    ext = os.path.splitext(clean_filename)[1].lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    return FileResponse(
        img_path,
        media_type=media_type,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=86400",
        },
    )


@router.post("/upload/{session_id}")
async def upload_files(
    session_id: str,
    files: List[UploadFile] = File(...),
):
    """
    Upload one or more study files into the session's vector store.
    Supports PDF, DOCX, PPTX, TXT.
    """
    load_session(session_id)
    if session_id not in session_histories:
        session_histories[session_id] = []
        session_files[session_id] = []
        session_docs[session_id] = {}
        session_slides[session_id] = {}

    all_docs = []
    uploaded_names = []
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    supabase = get_supabase_client()
    ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".ppt", ".docx", ".doc", ".txt"}
    for uf in files:
        ext = os.path.splitext(uf.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}' for file '{uf.filename}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )
        file_bytes = await uf.read()

        saved_path = os.path.join(session_dir, uf.filename)
        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        try:
            docs, slides_info = load_file(saved_path, ext, session_id=session_id)
            all_docs.extend(docs)
            uploaded_names.append(uf.filename)
            if slides_info:
                session_slides.setdefault(session_id, {})[uf.filename] = slides_info

            # Store full text for summarization
            full_text = "\n\n".join(d["content"] for d in docs)
            session_docs.setdefault(session_id, {})[uf.filename] = full_text

            # If Supabase is connected, store the raw file in Supabase Storage
            if supabase:
                try:
                    supabase.storage.from_("documents").upload(
                        path=f"{session_id}/{uf.filename}",
                        file=file_bytes,
                        file_options={"upsert": "true"},
                    )
                except Exception as s_err:
                    print(f"Supabase storage upload note: {s_err}")
        except Exception as e:
            return {"error": f"Failed to parse {uf.filename}: {str(e)}", "skipped": uf.filename}

    if not all_docs:
        raise HTTPException(
            status_code=400,
            detail="No readable text could be extracted from the uploaded document(s). Please verify the file contains readable text.",
        )

    # Split all documents into chunks
    all_texts = []
    all_metadata = []
    for d in all_docs:
        chunks = split_text(d["content"])
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadata.append({**d.get("metadata", {}), "session_id": session_id})

    # Embedding & Indexing with Key Rotation on Rate Limit (429)
    keys = get_gemini_api_keys()
    max_attempts = max(len(keys), 1) if keys else 1
    stored_in_supabase = False
    indexing_success = False
    last_emb_error = None

    for attempt in range(max_attempts):
        try:
            curr_embeddings = get_embeddings()
            vectors = curr_embeddings.embed_documents(all_texts)

            # Persist in Supabase pgvector if configured
            if supabase:
                try:
                    # Insert documents with embeddings into Supabase
                    rows = []
                    for i, (text, meta, vec) in enumerate(zip(all_texts, all_metadata, vectors)):
                        rows.append({
                            "content": text,
                            "metadata": meta,
                            "embedding": vec,
                        })
                    # Batch insert in chunks of 50 to avoid request payload size limits
                    for b_idx in range(0, len(rows), 50):
                        supabase.table("documents").insert(rows[b_idx:b_idx + 50]).execute()
                    stored_in_supabase = True
                except Exception as err:
                    if is_rate_limit_error(err):
                        raise  # Let outer retry loop handle 429
                    print(f"Warning: Supabase vector insert failed ({err}), falling back to in-memory.")
                    stored_in_supabase = False

            # Always cache current session vector store in memory for instant 0ms retrieval and offline fallback
            if len(session_vectorstores) >= MAX_CACHED_VECTORSTORES:
                oldest_sid = next(iter(session_vectorstores))
                del session_vectorstores[oldest_sid]
            if session_id in session_vectorstores:
                session_vectorstores[session_id]["chunks"].extend(
                    [{"content": t, "metadata": m} for t, m in zip(all_texts, all_metadata)]
                )
                session_vectorstores[session_id]["embeddings"].extend(vectors)
            else:
                session_vectorstores[session_id] = {
                    "chunks": [{"content": t, "metadata": m} for t, m in zip(all_texts, all_metadata)],
                    "embeddings": vectors,
                }

            indexing_success = True
            break
        except Exception as emb_err:
            last_emb_error = emb_err
            if is_rate_limit_error(emb_err) and keys:
                print(f"Embedding rate limit / 429 with current key ({emb_err}). Rotating key...")
                rotate_api_key()
                continue
            else:
                print(f"Embedding / indexing error ({emb_err})")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to embed/index documents: {str(emb_err)}",
                )

    if not indexing_success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to embed/index after {max_attempts} key rotations: {str(last_emb_error)}",
        )

    # Clean up upload buffers and trigger garbage collection immediately
    chunks_count = len(all_texts)
    del all_docs
    del all_texts
    del vectors
    gc.collect()

    session_files[session_id].extend(uploaded_names)
    save_session(session_id)

    return {
        "message": f"Uploaded {len(uploaded_names)} file(s), indexed {chunks_count} chunks.",
        "files": session_files[session_id],
        "chunks": chunks_count,
        "slides": session_slides.get(session_id, {}),
        "docs": session_docs.get(session_id, {}),
        "storage": "supabase" if stored_in_supabase else "in-memory",
    }


@router.get("/slides/{session_id}/{filename}")
def get_slides(session_id: str, filename: str):
    """Retrieve structured slides for PPT viewer."""
    if session_id not in session_slides:
        load_session(session_id)
    if session_id not in session_slides or filename not in session_slides[session_id]:
        raise HTTPException(status_code=404, detail="Slides not found.")
    return {
        "filename": filename,
        "slides": session_slides[session_id][filename],
    }


@router.get("/document/{session_id}/{filename}")
def get_document_text(session_id: str, filename: str):
    """Retrieve extracted document text for DOCX/TXT viewer."""
    if session_id not in session_docs:
        load_session(session_id)
    if session_id not in session_docs or filename not in session_docs[session_id]:
        raise HTTPException(status_code=404, detail="Document text not found.")
    return {
        "filename": filename,
        "text": session_docs[session_id][filename],
    }
