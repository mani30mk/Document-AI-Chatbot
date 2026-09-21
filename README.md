# Document AI Chatbot 📚🤖

A full-stack Retrieval-Augmented Generation (RAG) application that allows you to upload study materials, view them natively in your browser, and interact with them using a smart chatbot powered by Google's Gemini.

## Features ✨
- **RAG Chatbot**: Ask questions about your uploaded documents. The application chunks, embeds, and stores your files in a ChromaDB vector store, retrieving the most relevant context to answer your questions accurately.
- **Native File Viewer**: Read PDFs and Text files directly in the beautifully styled browser interface side-by-side with the chatbot.
- **One-Click Summarization**: Instantly generate comprehensive, cohesive summaries of entire documents leveraging Gemini's massive context window.
- **Multi-Format Support**: Upload `.pdf`, `.txt`, `.docx`, and `.pptx` files.

## Tech Stack 🛠️
- **Backend**: Python, FastAPI, LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: FastEmbed (`BAAI/bge-small-en-v1.5` — 100% local ONNX runtime, **0 API tokens used**, ~120 MB RAM)
- **LLM**: Google Gemini (`gemini-2.5-flash`)
- **Frontend**: Vanilla HTML, CSS (Custom Design System), JavaScript

## How to Run Locally 🚀

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mani30mk/Document-AI-Chatbot.git
   cd Document-AI-Chatbot
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the FastAPI backend:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. **Open the App:**
   Visit `http://localhost:8000` in your browser. The app connects automatically and is ready to upload files and answer questions!

---

## Deploy to Render 🌐

This project includes a ready-to-use [`render.yaml`](render.yaml) blueprint configured for Render's **Free Tier**.

### Option 1: One-Click Blueprint
1. Push your repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com), click **New +** -> **Blueprint**.
3. Connect your repository — Render will automatically read `render.yaml` and configure the Web Service.

### Option 2: Manual Web Service
1. In the [Render Dashboard](https://dashboard.render.com), click **New +** -> **Web Service**.
2. Connect your GitHub repository.
3. Configure the settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`
4. Click **Deploy Web Service**.
5. Once deployed, visit your Render URL (e.g. `https://your-app.onrender.com`). Both the frontend and backend are served from this single service!

---

## (Optional) Supabase Cloud Storage & Persistent Vectors 🗄️

By default, the app runs in-memory. If you want documents and vectors to **persist permanently** in the cloud:

1. Create a free project at [supabase.com](https://supabase.com).
2. Go to **SQL Editor** and run:
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
3. Under **Storage**, create a new public bucket named `documents`.
4. In Render (or your `.env` locally), add the environment variables:
   - `SUPABASE_URL`: Your Supabase Project URL
   - `SUPABASE_KEY`: Your Supabase `service_role` (or `anon`) key

