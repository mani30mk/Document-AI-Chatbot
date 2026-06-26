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
- **LLMs & Embeddings**: Google Gemini (`gemini-2.5-flash`), HuggingFace (`all-MiniLM-L6-v2`)
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
   Simply double-click on `index.html` to open it in your browser. Enter your **Gemini API Key** in the top navigation bar, click "Connect", and start uploading your study files!
