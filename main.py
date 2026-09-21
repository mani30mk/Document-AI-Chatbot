"""
RAG Study Assistant — FastAPI Backend
Wraps your LangChain RAG chain with:
  - File upload + multi-format loading (PDF, DOCX, PPTX, TXT)
  - Per-session ChromaDB vector stores
  - Multi-turn conversation memory
  - /ask endpoint consumed by the frontend chatbot

Run:
    pip install fastapi uvicorn langchain langchain-community langchain-huggingface
                langchain-chroma langchain-google-genai python-multipart pypdf
                unstructured python-docx python-pptx
    uvicorn main:app --reload --port 8000
"""

import os
import uuid
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── LangChain imports ───────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import docx
from pptx import Presentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

import tempfile
import base64
import io
import gc
from PIL import Image
import urllib.request
import urllib.parse
import re
import json

# ── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Study Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/", methods=["GET", "HEAD"])
def serve_index():
    """Serve the frontend single page app."""
    return FileResponse("index.html")


# ── Global state & Cloud storage setup ─────────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploaded_docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
SESSIONS_FILE = os.path.join(UPLOAD_DIR, "sessions.json")

# In-memory session fallbacks (capped to 3 active vectorstores to conserve Render 512MB RAM)
MAX_CACHED_VECTORSTORES = 3
session_vectorstores: dict[str, Chroma] = {}
session_histories: dict[str, list] = {}
session_files: dict[str, list[str]] = {}
session_docs: dict[str, dict[str, str]] = {}
session_slides: dict[str, dict[str, list]] = {}
session_api_keys: dict[str, str] = {}


def get_or_load_vectorstore(session_id: str) -> Chroma | None:
    """Retrieve in-memory Chroma or restore from disk-persisted directory."""
    if session_id in session_vectorstores:
        return session_vectorstores[session_id]

    vector_dir = os.path.join(UPLOAD_DIR, session_id, "chroma_db")
    if os.path.exists(vector_dir):
        try:
            embeddings = get_embeddings()
            vs = Chroma(persist_directory=vector_dir, embedding_function=embeddings)
            # Evict oldest vectorstore if exceeding cache limit
            if len(session_vectorstores) >= MAX_CACHED_VECTORSTORES:
                oldest_sid = next(iter(session_vectorstores))
                del session_vectorstores[oldest_sid]
                gc.collect()
            session_vectorstores[session_id] = vs
            return vs
        except Exception as e:
            print(f"Notice: Could not load Chroma from disk for {session_id} ({e})")
            # If dimension mismatch or corrupted collection, clean it up
            try:
                import shutil
                shutil.rmtree(vector_dir, ignore_errors=True)
            except Exception:
                pass
    return None


def load_sessions_from_disk():
    global session_histories, session_files, session_docs, session_slides
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                session_files.update(data.get("files", {}))
                session_docs.update(data.get("docs", {}))
                session_slides.update(data.get("slides", {}))
                for sid, msgs in data.get("histories", {}).items():
                    session_histories[sid] = [
                        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                        for m in msgs
                    ]
            print(f"Loaded {len(session_files)} sessions from disk.")
        except Exception as e:
            print(f"Notice: Could not load sessions from disk ({e})")


def save_sessions_to_disk():
    try:
        data = {
            "files": session_files,
            "docs": session_docs,
            "slides": session_slides,
            "histories": {
                sid: [
                    {"role": "user" if isinstance(m, HumanMessage) else "bot", "content": m.content}
                    for m in msgs
                ]
                for sid, msgs in session_histories.items()
            }
        }
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Notice: Could not save sessions to disk ({e})")


# Supabase (persistent cloud storage & pgvector)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
supabase_client: Client | None = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase for persistent cloud storage & pgvector.")
    except Exception as e:
        print(f"Warning: Could not connect to Supabase: {e}")


# ── Shared components ────────────────────────────────────────────────────────
EMBEDDINGS = None
EMBEDDINGS_ERROR = None

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful study assistant. Answer questions using ONLY the context below.
If the answer isn't in the context, say so honestly.

Context:
{context}""",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ── Gemini API Keys Management & Rotation ──────────────────────────────────
_current_key_index: int = 0


def get_gemini_api_keys() -> list[str]:
    """
    Retrieve Gemini API keys from GEMINI_API_KEY environment variable.
    Supports comma-separated, semicolon-separated, or newline-separated multiple keys
    for quota failover (e.g. GEMINI_API_KEY="key1,key2,key3").
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


def get_current_api_key() -> str | None:
    """Get the currently active API key."""
    keys = get_gemini_api_keys()
    if not keys:
        return None
    global _current_key_index
    return keys[_current_key_index % len(keys)]


def rotate_api_key() -> str | None:
    """Rotate to the next API key in round-robin fashion."""
    keys = get_gemini_api_keys()
    if not keys:
        return None
    global _current_key_index
    _current_key_index = (_current_key_index + 1) % len(keys)
    masked = keys[_current_key_index][:6] + "..." + keys[_current_key_index][-4:] if len(keys[_current_key_index]) > 10 else "***"
    print(f"Rotated to API key #{_current_key_index + 1}/{len(keys)} ({masked})")
    return keys[_current_key_index]


def is_rate_limit_error(exc: Exception) -> bool:
    """Detect if an exception is due to 429 quota exhaustion or rate limiting."""
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
        ]
    )


AVAILABLE_GEMINI_MODELS: list[str] = []


def get_available_models(api_key: str | None = None) -> list[str]:
    """Query Google API for active models supporting generateContent for Gemini API key."""
    global AVAILABLE_GEMINI_MODELS
    if AVAILABLE_GEMINI_MODELS:
        return AVAILABLE_GEMINI_MODELS

    candidates = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    keys = [api_key] if api_key else get_gemini_api_keys()
    if not keys:
        return candidates

    for key in keys:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
            req = urllib.request.Request(url, headers={"User-Agent": "DocumentAI/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
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
                        and not any(x in name_lower for x in ["tts", "embed", "imagen", "robotics", "computer-use"])
                    ):
                        discovered.append(name)

                # Prioritize flash models
                flash_models = [m for m in discovered if "flash" in m]
                other_models = [m for m in discovered if "flash" not in m]
                sorted_models = flash_models + other_models
                if sorted_models:
                    AVAILABLE_GEMINI_MODELS = sorted_models
                    print(f"Discovered {len(AVAILABLE_GEMINI_MODELS)} available Gemini models: {AVAILABLE_GEMINI_MODELS[:5]}")
                    return AVAILABLE_GEMINI_MODELS
        except Exception as e:
            print(f"Notice: Model discovery via API key ({e}), trying next candidate/defaults.")
            continue

    AVAILABLE_GEMINI_MODELS = candidates
    return AVAILABLE_GEMINI_MODELS


def get_llm(model_name: str | None = None, api_key: str | None = None) -> ChatGoogleGenerativeAI:
    key = api_key or get_current_api_key()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please configure GEMINI_API_KEY in Render or in your .env file."
        )
    if not model_name:
        models = get_available_models(key)
        model_name = models[0] if models else "gemini-2.5-flash"

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key,
        temperature=0.2,
        request_timeout=20,
    )


def load_docx(path: str) -> list[Document]:
    doc = docx.Document(path)
    text_parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                text_parts.append(row_text)
    return [Document(page_content="\n\n".join(text_parts), metadata={"source": path})]


def extract_shape_images(shape):
    imgs = []
    try:
        if hasattr(shape, "image"):
            imgs.append(shape.image)
        elif getattr(shape, "shape_type", None) == 6 and hasattr(shape, "shapes"):  # Group shape
            for sub_shape in shape.shapes:
                imgs.extend(extract_shape_images(sub_shape))
    except Exception:
        pass
    return imgs


def load_pptx(path: str, session_id: str | None = None) -> tuple[list[Document], list[dict]]:
    # Deduce session_id from path if not explicitly passed
    if not session_id:
        norm_path = os.path.normpath(path)
        parts = norm_path.split(os.sep)
        if "uploaded_docs" in parts:
            idx = parts.index("uploaded_docs")
            if len(parts) > idx + 1:
                session_id = parts[idx + 1]

    prs = Presentation(path)
    docs = []
    slides_data = []
    for slide_idx, slide in enumerate(prs.slides):
        slide_texts = []
        title = ""
        bullets = []
        tables = []
        images = []

        # Check title shape first
        try:
            if slide.shapes.title and slide.shapes.title.text.strip():
                title = slide.shapes.title.text.strip()
                slide_texts.append(title)
        except Exception:
            pass

        for shape in slide.shapes:
            try:
                if shape == getattr(slide.shapes, "title", None):
                    continue

                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            slide_texts.append(text)
                            if not title and len(text) < 80:
                                title = text
                            elif text != title:
                                bullets.append({
                                    "text": text,
                                    "level": getattr(paragraph, "level", 0),
                                })

                elif shape.has_table:
                    table_rows = []
                    for row in shape.table.rows:
                        row_cells = [cell.text.strip() for cell in row.cells]
                        if any(row_cells):
                            table_rows.append(row_cells)
                            row_text = " | ".join(c for c in row_cells if c)
                            slide_texts.append(row_text)
                    if table_rows:
                        tables.append(table_rows)

                # Extract pictures/images
                shape_imgs = extract_shape_images(shape)
                for img in shape_imgs:
                    if len(images) >= 4:  # Cap at 4 images per slide
                        break
                    try:
                        pil_img = Image.open(io.BytesIO(img.blob))
                        if pil_img.width > 900:
                            ratio = 900 / pil_img.width
                            new_size = (900, int(pil_img.height * ratio))
                            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                        pil_format = "PNG" if pil_img.mode in ("RGBA", "P") else "JPEG"

                        if session_id:
                            img_dir = os.path.join(UPLOAD_DIR, session_id, "slide_images")
                            os.makedirs(img_dir, exist_ok=True)
                            ext_name = "png" if pil_format == "PNG" else "jpg"
                            img_filename = f"slide_{slide_idx + 1}_img_{len(images) + 1}.{ext_name}"
                            img_file_path = os.path.join(img_dir, img_filename)
                            pil_img.save(img_file_path, format=pil_format, quality=85)
                            images.append(f"/slide_image/{session_id}/{img_filename}")
                        else:
                            buf = io.BytesIO()
                            pil_img.save(buf, format=pil_format, quality=85)
                            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                            mime = f"image/{pil_format.lower()}"
                            images.append(f"data:{mime};base64,{b64_str}")
                    except Exception as img_err:
                        print(f"Notice: Image extraction ({img_err})")
            except Exception:
                continue

        # Speaker notes
        notes = ""
        try:
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
        except Exception:
            pass

        if not title and bullets:
            title = bullets.pop(0)
        if not title:
            title = f"Slide {slide_idx + 1}"

        slides_data.append({
            "slide_number": slide_idx + 1,
            "title": title,
            "bullets": bullets,
            "tables": tables,
            "images": images,
            "notes": notes,
            "raw_text": "\n".join(slide_texts),
        })

        img_context = f" [Slide contains {len(images)} figure(s)/diagram(s)]" if images else ""
        content_for_doc = ("\n".join(slide_texts) if slide_texts else f"Slide {slide_idx + 1}: {title}") + img_context
        docs.append(
            Document(
                page_content=content_for_doc,
                metadata={"slide": slide_idx + 1, "source": path, "has_images": len(images) > 0},
            )
        )
    return docs, slides_data


def load_file(path: str, ext: str, session_id: str | None = None) -> tuple[list[Document], list[dict] | None]:
    if ext == ".pdf":
        return PyPDFLoader(path).load(), None
    elif ext == ".docx":
        return load_docx(path), None
    elif ext in (".pptx", ".ppt"):
        return load_pptx(path, session_id=session_id)
    elif ext == ".txt":
        return TextLoader(path, encoding="utf-8").load(), None
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def get_embeddings():
    global EMBEDDINGS, EMBEDDINGS_ERROR

    if EMBEDDINGS is not None:
        return EMBEDDINGS

    if EMBEDDINGS_ERROR is not None:
        raise EMBEDDINGS_ERROR

    # 1. Primary: GoogleGenerativeAIEmbeddings (cloud API, 0 MB local RAM used)
    # Saves ~250MB RAM compared to local ONNX models, preventing Render 512MB OOM
    key = get_current_api_key()
    if key:
        try:
            EMBEDDINGS = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=key,
            )
            print("Using GoogleGenerativeAIEmbeddings (cloud API - 0 MB local RAM).")
            return EMBEDDINGS
        except Exception as g_err:
            print(f"Notice: Google embeddings init ({g_err}), falling back to FastEmbed...")

    # 2. Fallback: FastEmbedEmbeddings (local ONNX model) only if no API key is available
    try:
        EMBEDDINGS = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        print("Using FastEmbedEmbeddings (local ONNX - ~150MB RAM).")
        return EMBEDDINGS
    except Exception as exc:
        EMBEDDINGS_ERROR = exc
        raise


# ── Request / Response models ────────────────────────────────────────────────
class AskRequest(BaseModel):
    session_id: str
    question: str


class SummarizeRequest(BaseModel):
    session_id: str
    filename: str
    text: str | None = None


class YouTubeVideo(BaseModel):
    id: str
    title: str
    channel: str
    duration: str
    url: str
    thumbnail: str


class AskResponse(BaseModel):
    answer: str
    session_id: str
    sources_used: int
    youtube_sources: List[YouTubeVideo] = []


class SessionResponse(BaseModel):
    session_id: str
    files: List[str]


def search_youtube(query: str, max_results: int = 3) -> list[dict]:
    """Search YouTube for educational tutorials matching the query without API key."""
    try:
        clean_query = query.strip()
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(clean_query + " tutorial")
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=2.0) as response:
            html = response.read().decode("utf-8")
            match = re.search(r"var ytInitialData = ({.*?});</script>", html)
            if not match:
                return []
            data = json.loads(match.group(1))
            contents = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
            videos = []
            for section in contents:
                items = section.get("itemSectionRenderer", {}).get("contents", [])
                for item in items:
                    v = item.get("videoRenderer")
                    if v and "videoId" in v:
                        vid_id = v.get("videoId")
                        title = v.get("title", {}).get("runs", [{}])[0].get("text", "")
                        channel = v.get("ownerText", {}).get("runs", [{}])[0].get("text", "")
                        duration = v.get("lengthText", {}).get("simpleText", "")
                        if vid_id and title:
                            videos.append({
                                "id": vid_id,
                                "title": title,
                                "channel": channel,
                                "duration": duration,
                                "url": f"https://www.youtube.com/watch?v={vid_id}",
                                "thumbnail": f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg",
                            })
                            if len(videos) >= max_results:
                                break
                if len(videos) >= max_results:
                    break
            return videos
    except Exception as e:
        print(f"YouTube search notice: {e}")
        return []


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/session/new", response_model=SessionResponse)
def new_session():
    """Create a new chat/upload session."""
    sid = str(uuid.uuid4())
    session_histories[sid] = []
    session_files[sid] = []
    session_docs[sid] = {}
    return SessionResponse(session_id=sid, files=[])


@app.get("/session/{session_id}")
def get_session_state(session_id: str):
    """Retrieve full session state (files, slides, docs, chat history) for page reload."""
    if session_id not in session_histories and session_id not in session_files:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    history_list = []
    for msg in session_histories.get(session_id, []):
        role = "user" if isinstance(msg, HumanMessage) else "bot"
        history_list.append({"role": role, "content": msg.content})

    return {
        "session_id": session_id,
        "files": session_files.get(session_id, []),
        "slides": session_slides.get(session_id, {}),
        "docs": session_docs.get(session_id, {}),
        "history": history_list,
        "history_turns": len(history_list) // 2,
    }


import threading


@app.on_event("startup")
def startup_prewarm():
    """Load sessions and conditionally prewarm only if no cloud API key is configured."""
    load_sessions_from_disk()

    # Only prewarm local FastEmbed if no Gemini API key is configured
    if not get_gemini_api_keys():
        def _prewarm():
            try:
                get_embeddings()
                print("FastEmbed model pre-warmed successfully.")
            except Exception as err:
                print(f"Notice: FastEmbed pre-warm ({err})")
        threading.Thread(target=_prewarm, daemon=True).start()
    else:
        print("Gemini API key detected: Skipping local ONNX pre-warm to conserve Render RAM.")


@app.get("/raw/{session_id}/{filename}")
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

    if supabase_client:
        try:
            from fastapi.responses import RedirectResponse
            pub_url = supabase_client.storage.from_("documents").get_public_url(f"{session_id}/{filename}")
            if pub_url:
                return RedirectResponse(pub_url)
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="File not found.")


@app.get("/slide_image/{session_id}/{filename}")
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


@app.post("/upload/{session_id}")
async def upload_files(
    session_id: str,
    files: List[UploadFile] = File(...),
):
    """
    Upload one or more study files into the session's vector store.
    Supports PDF, DOCX, PPTX, TXT.
    """
    if session_id not in session_histories:
        session_histories[session_id] = []
        session_files[session_id] = []
        session_docs[session_id] = {}

    all_docs = []
    uploaded_names = []
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

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
            full_text = "\n\n".join(d.page_content for d in docs)
            session_docs.setdefault(session_id, {})[uf.filename] = full_text

            # If Supabase is connected, store the raw file in Supabase Storage
            if supabase_client:
                try:
                    supabase_client.storage.from_("documents").upload(
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

    chunks = TEXT_SPLITTER.split_documents(all_docs)
    for c in chunks:
        c.metadata["session_id"] = session_id

    try:
        embeddings = get_embeddings()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding model could not be initialized: {exc}",
        ) from exc

    # Persist in Supabase pgvector if configured, otherwise use in-memory Chroma
    stored_in_supabase = False
    if supabase_client:
        try:
            supabase_vectorstore = SupabaseVectorStore(
                client=supabase_client,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents",
            )
            supabase_vectorstore.add_documents(chunks)
            stored_in_supabase = True
        except Exception as err:
            print(f"Warning: Supabase vector store insert failed ({err}), falling back to Chroma.")

    if not stored_in_supabase:
        vector_dir = os.path.join(session_dir, "chroma_db")
        os.makedirs(vector_dir, exist_ok=True)
        if session_id in session_vectorstores:
            session_vectorstores[session_id].add_texts(
                [c.page_content for c in chunks]
            )
        else:
            if len(session_vectorstores) >= MAX_CACHED_VECTORSTORES:
                oldest_sid = next(iter(session_vectorstores))
                del session_vectorstores[oldest_sid]
            session_vectorstores[session_id] = Chroma.from_documents(
                chunks, embeddings, persist_directory=vector_dir
            )

    # Clean up upload buffers and trigger garbage collection immediately
    del all_docs
    del chunks
    gc.collect()

    session_files[session_id].extend(uploaded_names)
    save_sessions_to_disk()

    return {
        "message": f"Uploaded {len(uploaded_names)} file(s), indexed {len(chunks)} chunks.",
        "files": session_files[session_id],
        "chunks": len(chunks),
        "slides": session_slides.get(session_id, {}),
        "docs": session_docs.get(session_id, {}),
        "storage": "supabase" if stored_in_supabase else "in-memory",
    }


@app.get("/slides/{session_id}/{filename}")
def get_slides(session_id: str, filename: str):
    """Retrieve structured slides for PPT viewer."""
    if session_id not in session_slides or filename not in session_slides[session_id]:
        raise HTTPException(status_code=404, detail="Slides not found.")
    return {
        "filename": filename,
        "slides": session_slides[session_id][filename],
    }


@app.get("/document/{session_id}/{filename}")
def get_document_text(session_id: str, filename: str):
    """Retrieve extracted document text for DOCX/TXT viewer."""
    if session_id not in session_docs or filename not in session_docs[session_id]:
        raise HTTPException(status_code=404, detail="Document text not found.")
    return {
        "filename": filename,
        "text": session_docs[session_id][filename],
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Ask a question. Uses RAG + multi-turn conversation history.
    """
    sid = req.session_id

    if sid not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found.")

    retriever = None
    if supabase_client:
        try:
            supabase_vectorstore = SupabaseVectorStore(
                client=supabase_client,
                embedding=get_embeddings(),
                table_name="documents",
                query_name="match_documents",
            )
            retriever = supabase_vectorstore.as_retriever(
                search_kwargs={"k": 5, "filter": {"session_id": sid}}
            )
        except Exception as err:
            print(f"Supabase retriever error ({err}), falling back to in-memory.")

    if retriever is None:
        vs = get_or_load_vectorstore(sid)
        if vs is not None:
            retriever = vs.as_retriever(search_kwargs={"k": 5})
        else:
            if sid not in session_histories and sid not in session_files:
                raise HTTPException(
                    status_code=400,
                    detail="No files uploaded for this session yet. Upload files first via /upload/{session_id}.",
                )

    # Build the chain with history
    history = session_histories[sid]

    retrieved_docs = []
    if retriever:
        try:
            retrieved_docs = retriever.invoke(req.question)
        except Exception as r_err:
            print(f"Primary retriever failed ({r_err}), falling back to disk/in-memory Chroma...")
            vs = get_or_load_vectorstore(sid)
            if vs is not None:
                retriever = vs.as_retriever(search_kwargs={"k": 5})
                try:
                    retrieved_docs = retriever.invoke(req.question)
                except Exception as c_err:
                    print(f"Chroma retriever notice: {c_err}")

    context = format_docs(retrieved_docs)
    if not context and sid in session_docs:
        context = "\n\n".join(list(session_docs[sid].values()))[:4000]

    models_to_try = get_available_models()[:3]
    keys = get_gemini_api_keys()
    if not keys:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in Render or in your .env file.",
        )

    answer = None
    last_error = None

    for m_name in models_to_try:
        for _ in range(len(keys)):
            current_key = get_current_api_key()
            try:
                curr_llm = get_llm(m_name, api_key=current_key)
                chain = RAG_PROMPT | curr_llm | StrOutputParser()
                answer = chain.invoke({
                    "context": context,
                    "chat_history": history,
                    "question": req.question,
                })
                if answer:
                    break
            except Exception as llm_err:
                last_error = llm_err
                if is_rate_limit_error(llm_err):
                    print(f"Rate limit / 429 on model {m_name} with current key: {llm_err}. Rotating key...")
                    rotate_api_key()
                    continue
                else:
                    print(f"Model {m_name} failed ({llm_err}), falling back to next model candidate...")
                    break
        if answer:
            break

    if not answer:
        raise HTTPException(status_code=500, detail=f"AI model error: {str(last_error)}")

    # Persist turn to history
    session_histories[sid].append(HumanMessage(content=req.question))
    session_histories[sid].append(AIMessage(content=answer))

    # Keep last 20 messages (10 turns) to avoid token bloat
    if len(session_histories[sid]) > 20:
        session_histories[sid] = session_histories[sid][-20:]

    save_sessions_to_disk()

    yt_results = []
    try:
        yt_results = search_youtube(req.question, max_results=3)
    except Exception:
        pass

    return AskResponse(
        answer=answer,
        session_id=sid,
        sources_used=len(retrieved_docs),
        youtube_sources=[YouTubeVideo(**v) for v in yt_results],
    )


def normalize_doc_name(name: str) -> str:
    """Normalize filename for fuzzy matching (case, hyphens, underscores, spaces)."""
    return re.sub(r"[\s\-_]+", "", name.lower())


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    """Summarize a specific uploaded file with multi-tier fallback."""
    sid = req.session_id
    fname = req.filename
    text = (req.text or "").strip()

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
                    text = "\n\n".join(d.page_content for d in docs)
                    session_docs.setdefault(sid, {})[fname] = text
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

    prompt = ChatPromptTemplate.from_template(
        "You are an expert summarizer. Please provide a comprehensive and concise summary of the following document:\n\n{text}"
    )

    models_to_try = get_available_models()[:3]
    keys = get_gemini_api_keys()
    if not keys:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in Render or in your .env file.",
        )

    summary = None
    last_error = None

    for m_name in models_to_try:
        for _ in range(len(keys)):
            current_key = get_current_api_key()
            try:
                curr_llm = get_llm(m_name, api_key=current_key)
                chain = prompt | curr_llm | StrOutputParser()
                summary = chain.invoke({"text": text[:20000]})
                if summary:
                    break
            except Exception as e:
                last_error = e
                if is_rate_limit_error(e):
                    print(f"Summarize rate limit / 429 on model {m_name} with current key: {e}. Rotating key...")
                    rotate_api_key()
                    continue
                else:
                    print(f"Summarize model {m_name} failed ({e}), falling back to next model candidate...")
                    break
        if summary:
            break

    if not summary:
        raise HTTPException(status_code=500, detail=f"AI model error: {str(last_error)}")

    return {"summary": summary}



@app.delete("/session/{session_id}/history")
def clear_history(session_id: str):
    """Clear conversation history for a session (keeps files/vectorstore)."""
    if session_id not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found.")
    session_histories[session_id] = []
    save_sessions_to_disk()
    return {"message": "Conversation history cleared."}


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}
