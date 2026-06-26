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

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── LangChain imports ───────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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

# ── Global state (per-session stores + chat histories) ──────────────────────
# In production, swap these dicts for Redis / a proper session store.
session_vectorstores: dict[str, Chroma] = {}
session_histories: dict[str, list] = {}
session_files: dict[str, list[str]] = {}
session_docs: dict[str, dict[str, str]] = {}

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


def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
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


def get_embeddings():
    global EMBEDDINGS, EMBEDDINGS_ERROR

    if EMBEDDINGS is not None:
        return EMBEDDINGS

    if EMBEDDINGS_ERROR is not None:
        raise EMBEDDINGS_ERROR

    try:
        EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return EMBEDDINGS
    except Exception as exc:
        EMBEDDINGS_ERROR = exc
        raise


# ── Request / Response models ────────────────────────────────────────────────
class AskRequest(BaseModel):
    session_id: str
    question: str
    gemini_api_key: str


class SummarizeRequest(BaseModel):
    session_id: str
    filename: str
    gemini_api_key: str


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
async def upload_files(session_id: str, files: List[UploadFile] = File(...)):
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
        # Write to a temp file so loaders can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await uf.read())
            tmp_path = tmp.name

        try:
            docs = load_file(tmp_path, ext)
            all_docs.extend(docs)
            uploaded_names.append(uf.filename)
            
            # Store full text for summarization
            full_text = "\n\n".join(d.page_content for d in docs)
            session_docs.setdefault(session_id, {})[uf.filename] = full_text
        except ValueError as e:
            return {"error": str(e), "skipped": uf.filename}
        finally:
            os.unlink(tmp_path)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No documents could be loaded.")

    chunks = TEXT_SPLITTER.split_documents(all_docs)

    try:
        embeddings = get_embeddings()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding model could not be initialized: {exc}",
        ) from exc

    # Add to existing vectorstore or create new one for this session
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
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Ask a question. Uses RAG + multi-turn conversation history.
    """
    sid = req.session_id

    if sid not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found.")

    if sid not in session_vectorstores:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded for this session yet. Upload files first via /upload/{session_id}.",
        )

    retriever = session_vectorstores[sid].as_retriever(search_kwargs={"k": 4})
    llm = get_llm(req.gemini_api_key)

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

    text = session_docs[sid][fname]
    llm = get_llm(req.gemini_api_key)
    
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
