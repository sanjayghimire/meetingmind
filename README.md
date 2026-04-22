# 🧠 MeetingMind

> Ask anything about your team's meetings. Get cited, grounded answers powered by RAG + Claude.

## What it does

MeetingMind is an AI-powered meeting intelligence tool. Paste any meeting transcript, then ask questions in natural language. It finds the relevant parts and answers with citations — no hallucination.

**Example:**
> "What was decided about the mobile app?"
> → "The mobile app was pushed to Q4 [Source: meeting-april-22]. Alice explained this was because there is too much on the team's plate [Source: meeting-april-22]."

## Tech stack

| Layer | Tool |
|---|---|
| LLM | Claude Haiku 4.5 (Anthropic) |
| RAG framework | LangChain |
| Embeddings | sentence-transformers (local, free) |
| Vector database | ChromaDB |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Deployment | Render + Streamlit Cloud |

## How it works

Transcript → Chunker → Embedder → Chroma (vector store)
↑
Question → Embedder → Retriever ────────┘
↓
Claude (LLM) → Cited answer


## Local setup

### 1. Clone and install
```bash
git clone https://github.com/YOURUSERNAME/meetingmind.git
cd meetingmind
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API key
```bash
copy .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Run the backend
```bash
python main.py
# FastAPI running at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### 4. Run the UI
```bash
streamlit run streamlit_app/app.py
# Opens at http://localhost:8501
```

### 5. Use it
1. Paste a meeting transcript in the sidebar
2. Click Ingest
3. Ask questions in the chat

## Project structure

meetingmind/
├── main.py                        FastAPI entry point
├── requirements.txt               All dependencies
├── app/
│   ├── core/config.py             Settings from .env
│   ├── ingestion/
│   │   ├── chunker.py             Speaker-aware text chunker
│   │   ├── embedder.py            Local sentence-transformers
│   │   ├── vector_store.py        Chroma wrapper
│   │   └── pipeline.py            Ingestion orchestrator
│   └── rag/
│       └── chain.py               RAG chain + Claude generation
└── streamlit_app/
└── app.py                     Chat UI



## Concepts covered

- **LLM** — Large Language Models and how they work
- **RAG** — Retrieval Augmented Generation
- **Embeddings** — Converting text to vectors
- **Vector databases** — Semantic similarity search
- **Chunking** — Splitting documents for retrieval
- **FastAPI** — Building Python REST APIs
- **Streamlit** — Python-only web UIs

## Built with

Python 3.11 · LangChain · ChromaDB · Anthropic Claude · Streamlit · FastAPI