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
from langchain_google_genai import ChatGoogleGenerativeAI
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
# In-memory session fallbacks
session_vectorstores: dict[str, Chroma] = {}
session_histories: dict[str, list] = {}
session_files: dict[str, list[str]] = {}
session_docs: dict[str, dict[str, str]] = {}
session_slides: dict[str, dict[str, list]] = {}
session_api_keys: dict[str, str] = {}

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

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

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


def get_llm(model_name: str = "gemini-1.5-flash") -> ChatGoogleGenerativeAI:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please configure GEMINI_API_KEY in Render or in your .env file."
        )
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key,
        temperature=0.2,
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


def load_pptx(path: str) -> tuple[list[Document], list[dict]]:
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
                        buf = io.BytesIO()
                        pil_format = "PNG" if pil_img.mode in ("RGBA", "P") else "JPEG"
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


def load_file(path: str, ext: str) -> tuple[list[Document], list[dict] | None]:
    if ext == ".pdf":
        return PyPDFLoader(path).load(), None
    elif ext == ".docx":
        return load_docx(path), None
    elif ext in (".pptx", ".ppt"):
        return load_pptx(path)
    elif ext == ".txt":
        return TextLoader(path).load(), None
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def get_embeddings() -> FastEmbedEmbeddings:
    global EMBEDDINGS, EMBEDDINGS_ERROR

    if EMBEDDINGS is not None:
        return EMBEDDINGS

    if EMBEDDINGS_ERROR is not None:
        raise EMBEDDINGS_ERROR

    try:
        # BAAI/bge-small-en-v1.5 runs locally via ONNX Runtime (~120MB RAM, 0 API tokens used)
        EMBEDDINGS = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
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
    if session_id not in session_histories:
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
    }


import threading


@app.on_event("startup")
def startup_prewarm():
    """Pre-warm FastEmbed in background so first upload does not stall."""
    def _prewarm():
        try:
            get_embeddings()
            print("FastEmbed model pre-warmed successfully.")
        except Exception as err:
            print(f"Notice: FastEmbed pre-warm ({err})")
    threading.Thread(target=_prewarm, daemon=True).start()


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
        raise HTTPException(status_code=404, detail="Session not found. Call /session/new first.")

    all_docs = []
    uploaded_names = []

    for uf in files:
        ext = os.path.splitext(uf.filename)[1].lower()
        file_bytes = await uf.read()

        # Write to a temp file so loaders can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            docs, slides_info = load_file(tmp_path, ext)
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
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

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
        if session_id in session_vectorstores:
            session_vectorstores[session_id].add_texts(
                [c.page_content for c in chunks]
            )
        else:
            session_vectorstores[session_id] = Chroma.from_documents(chunks, embeddings)

    session_files[session_id].extend(uploaded_names)

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

    try:
        llm = get_llm()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

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
                search_kwargs={"k": 4, "filter": {"session_id": sid}}
            )
        except Exception as err:
            print(f"Supabase retriever error ({err}), falling back to in-memory.")

    if retriever is None:
        if sid not in session_vectorstores:
            raise HTTPException(
                status_code=400,
                detail="No files uploaded for this session yet. Upload files first via /upload/{session_id}.",
            )
        retriever = session_vectorstores[sid].as_retriever(search_kwargs={"k": 4})

    # Build the chain with history
    history = session_histories[sid]

    retrieved_docs = []
    if retriever:
        try:
            retrieved_docs = retriever.invoke(req.question)
        except Exception as r_err:
            print(f"Primary retriever failed ({r_err}), falling back to in-memory Chroma...")
            if sid in session_vectorstores:
                retriever = session_vectorstores[sid].as_retriever(search_kwargs={"k": 4})
                try:
                    retrieved_docs = retriever.invoke(req.question)
                except Exception as c_err:
                    print(f"Chroma retriever notice: {c_err}")

    context = format_docs(retrieved_docs)
    if not context and sid in session_docs:
        context = "\n\n".join(list(session_docs[sid].values()))[:4000]

    try:
        chain = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "chat_history": history,
            "question": req.question,
        })
    except Exception as llm_err:
        print(f"LLM invoke failed ({llm_err}), trying gemini-2.0-flash...")
        try:
            fallback_llm = get_llm("gemini-2.0-flash")
            chain = RAG_PROMPT | fallback_llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "chat_history": history,
                "question": req.question,
            })
        except Exception as f_err:
            raise HTTPException(status_code=500, detail=f"AI model error: {str(llm_err)}")

    # Persist turn to history
    session_histories[sid].append(HumanMessage(content=req.question))
    session_histories[sid].append(AIMessage(content=answer))

    # Keep last 20 messages (10 turns) to avoid token bloat
    if len(session_histories[sid]) > 20:
        session_histories[sid] = session_histories[sid][-20:]

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


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    """Summarize a specific uploaded file."""
    sid = req.session_id
    fname = req.filename

    if sid not in session_docs or fname not in session_docs[sid]:
        raise HTTPException(status_code=404, detail="File not found in session.")

    try:
        llm = get_llm()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    text = session_docs[sid][fname]
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert summarizer. Please provide a comprehensive and concise summary of the following document:\n\n{text}"
    )
    chain = prompt | llm | StrOutputParser()
    
    try:
        summary = chain.invoke({"text": text})
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}/history")
def clear_history(session_id: str):
    """Clear conversation history for a session (keeps files/vectorstore)."""
    if session_id not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found.")
    session_histories[session_id] = []
    return {"message": "Conversation history cleared."}


@app.get("/session/{session_id}")
def session_info(session_id: str):
    """Get session info — files uploaded and history length."""
    if session_id not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "files": session_files.get(session_id, []),
        "history_turns": len(session_histories[session_id]) // 2,
        "has_vectorstore": session_id in session_vectorstores,
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}
