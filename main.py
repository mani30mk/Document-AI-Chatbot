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
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

import tempfile

# ── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Study Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_index():
    """Serve the frontend single page app."""
    return FileResponse("index.html")


# ── Global state & Cloud storage setup ─────────────────────────────────────
# In-memory session fallbacks
session_vectorstores: dict[str, Chroma] = {}
session_histories: dict[str, list] = {}
session_files: dict[str, list[str]] = {}
session_docs: dict[str, dict[str, str]] = {}
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


def get_llm() -> ChatGoogleGenerativeAI:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please configure GEMINI_API_KEY in Render or in your .env file."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=key,
        temperature=0.2,
    )


def load_file(path: str, ext: str):
    loaders = {
        ".pdf":  PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".txt":  TextLoader,
    }
    loader_cls = loaders.get(ext)
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_cls(path).load()


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


class AskResponse(BaseModel):
    answer: str
    session_id: str
    sources_used: int


class SessionResponse(BaseModel):
    session_id: str
    files: List[str]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/session/new", response_model=SessionResponse)
def new_session():
    """Create a new chat/upload session."""
    sid = str(uuid.uuid4())
    session_histories[sid] = []
    session_files[sid] = []
    session_docs[sid] = {}
    return SessionResponse(session_id=sid, files=[])


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
            docs = load_file(tmp_path, ext)
            all_docs.extend(docs)
            uploaded_names.append(uf.filename)
            
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
        except ValueError as e:
            return {"error": str(e), "skipped": uf.filename}
        finally:
            os.unlink(tmp_path)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No documents could be loaded.")

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
        "storage": "supabase" if stored_in_supabase else "in-memory",
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

    retrieved_docs = retriever.invoke(req.question)
    context = format_docs(retrieved_docs)

    chain = RAG_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "chat_history": history,
        "question": req.question,
    })

    # Persist turn to history
    session_histories[sid].append(HumanMessage(content=req.question))
    session_histories[sid].append(AIMessage(content=answer))

    # Keep last 20 messages (10 turns) to avoid token bloat
    if len(session_histories[sid]) > 20:
        session_histories[sid] = session_histories[sid][-20:]

    return AskResponse(
        answer=answer,
        session_id=sid,
        sources_used=len(retrieved_docs),
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


@app.get("/health")
def health():
    return {"status": "ok"}
