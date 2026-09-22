# Document AI Chatbot 📚🤖

A modern, full-stack Retrieval-Augmented Generation (RAG) application that allows you to upload study materials, view them natively in your browser side-by-side, generate one-click summaries, search relevant educational lecture videos, and ask detailed questions using Google's Gemini models.

---

## 🏗️ System Workflows

To keep the architecture clean and easy to understand, the system is divided into two separate, dedicated pipelines:

### 1. Document Ingestion & Indexing Pipeline (Write Path)
How files are parsed, chunked, and stored as searchable vector embeddings without incurring API costs:

```mermaid
flowchart LR
    A["👤 User<br/>(Browser)"] -->|"1. Upload Document<br/>(PDF, PPTX, DOCX)"| B["⚙️ Backend Server<br/>(FastAPI)"]
    B -->|"2. Extract & Split<br/>(1000 chars / 200 overlap)"| C["✂️ Text Chunker"]
    C -->|"3. Micro-Batches<br/>(12 chunks / req)"| D["⚡ Embedding Microservice<br/>(FastEmbed ONNX 384-d)"]
    D -->|"4. Dense Vectors<br/>(0 Gemini API Cost)"| B
    B -->|"5. Store Embeddings & Chunks"| E[("💾 Supabase<br/>(pgvector)")]

    style A fill:#1e293b,stroke:#6366f1,stroke-width:2px,color:#fff
    style B fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
    style C fill:#0f172a,stroke:#38bdf8,stroke-width:1.5px,color:#fff
    style D fill:#0f172a,stroke:#34d399,stroke-width:2px,color:#fff
    style E fill:#0f172a,stroke:#f59e0b,stroke-width:2px,color:#fff
```

---

### 2. Question Answering & Retrieval Pipeline (Read / RAG Path)
How questions are vectorized, matched against relevant document context, synthesized by Gemini, and paired with educational video recommendations:

```mermaid
flowchart LR
    U["👤 User<br/>(Chat Query)"] -->|"1. Question"| S["⚙️ RAG Orchestrator<br/>(FastAPI)"]
    S -->|"2. Vectorize Query"| EM["⚡ FastEmbed<br/>Microservice"]
    EM -->|"3. Query Vector"| S
    S -->|"4. Cosine Match (Top-K)"| DB[("💾 Supabase<br/>(pgvector)")]
    DB -->|"5. Retrieved Context"| S
    S -->|"6. Context + Prompt"| G["🤖 Google Gemini<br/>(gemini-2.5-flash)"]
    G -->|"7. Grounded Answer"| S
    S -->|"8. Search Academic Topic"| YT["📺 YouTube Search"]
    YT -->|"9. Tutorial Videos"| S
    S -->|"10. Answer + Citations + Videos"| U

    style U fill:#1e293b,stroke:#6366f1,stroke-width:2px,color:#fff
    style S fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
    style EM fill:#0f172a,stroke:#34d399,stroke-width:2px,color:#fff
    style DB fill:#0f172a,stroke:#f59e0b,stroke-width:2px,color:#fff
    style G fill:#0f172a,stroke:#a855f7,stroke-width:2px,color:#fff
    style YT fill:#0f172a,stroke:#ef4444,stroke-width:2px,color:#fff
```

---

### 3. One-Click Document Summarization Pipeline

Summarize entire textbooks, slide decks, or lecture notes instantly:

```mermaid
flowchart TD
    Start["📑 Click 'Summarize' Button"] --> Fetch["Extract Full Document Text from Session"]
    Fetch --> Truncate{"Text Size Check"}
    Truncate -->|"Within Token Budget"| Gemini["Send Full Document to Gemini Context Window"]
    Truncate -->|"Exceeds Budget"| TopP["Select Representative Sections & Outline"] --> Gemini
    Gemini --> Format["Generate Structured Summary:<br/>• Core Overview<br/>• Key Concepts & Bullet Points<br/>• Important Takeaways"]
    Format --> UI["Render Animated Glassmorphic Summary Card in UI"]

    style Start fill:#6366f1,stroke:#4338ca,color:#fff
    style Gemini fill:#8b5cf6,stroke:#6d28d9,color:#fff
    style UI fill:#10b981,stroke:#059669,color:#fff
```

---

## ✨ Key Features

- **Decoupled Architecture**: High-speed embeddings served via dedicated ONNX microservice — **0 Google API tokens** spent on vectorization and minimal RAM footprint (~140 MB main app / ~110 MB embedding service).
- **RAG Chatbot**: Chat with multi-page PDFs, lecture presentations, and research papers with strict factual grounding.
- **Native Document Viewer**: Side-by-side reading experience with PDF viewer and rendered text preview.
- **Context-Aware YouTube Recommendations**: Identifies key academic concepts from questions and answers to recommend high-quality educational lecture videos.
- **One-Click Instant Summaries**: Synthesize long documents into executive summaries and bulleted study guides.
- **Automatic Fallbacks & Key Rotation**: Multi-tier LLM fallback (`gemini-2.5-flash` → `gemini-1.5-flash`) with automatic API key rotation on rate limits (429).
- **Multi-Format Support**: `.pdf`, `.docx`, `.pptx`, `.txt`.

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | Vanilla HTML5, Modern CSS3 (Glassmorphism, Dark Theme), JavaScript ES6+ |
| **Backend API** | Python 3.11, FastAPI, Uvicorn, Pydantic |
| **Embedding Engine** | FastEmbed (`BAAI/bge-small-en-v1.5`), ONNX Runtime (384 dimensions) |
| **LLM Inference** | Google Gemini (`gemini-2.5-flash`, `gemini-1.5-flash`) via `google-genai` |
| **File Parsing** | `pypdf`, `python-docx`, `python-pptx`, `Pillow` |
| **Database & Vectors** | Supabase PostgreSQL + `pgvector` (with in-memory fallback) |
| **Deployment** | Render Cloud Platform (Free Tier Blueprint) |

---

## 🚀 Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/mani30mk/Document-AI-Chatbot.git
cd Document-AI-Chatbot
```

### 2. Set Up Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your-gemini-api-key"
EMBEDDING_SERVICE_URL="https://document-ai-chatbot-7ch2.onrender.com"

# (Optional) Supabase Persistent Storage
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_KEY="your-supabase-key"
```

### 4. Start the Application
```bash
uvicorn main:app --reload --port 8000
```
Open **`http://localhost:8000`** in your browser. The app connects automatically and is ready to use!

---

## 🌐 Deploy to Render

This repository includes a multi-service configuration in [`render.yaml`](render.yaml) configured for Render's **Free Tier**:

| Service | Role | Memory Footprint |
| :--- | :--- | :--- |
| **`document-ai-chatbot`** | Web Application & Chatbot UI | ~140 MB / 512 MB |
| **`document-ai-embeddings`** | FastEmbed ONNX Microservice | ~110 MB / 512 MB |

### One-Click Blueprint Setup
1. Push your repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com), click **New +** → **Blueprint**.
3. Connect your repository — Render will automatically provision both web services.
4. Add your `GEMINI_API_KEY` in the environment variables tab.

---

## 🗄️ (Optional) Supabase Vector Database Setup

To persist vectors permanently across container restarts:
1. Create a project at [supabase.com](https://supabase.com).
2. Run the SQL schema script in Supabase's SQL Editor:
```sql
create extension if not exists vector;

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text,
  metadata jsonb,
  embedding vector(384)
);

create or replace function match_documents (
  query_embedding vector(384),
  filter jsonb default '{}'::jsonb,
  match_count int default 4
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
```
3. Create a public storage bucket named **`documents`**.
4. Set `SUPABASE_URL` and `SUPABASE_KEY` in your environment.
