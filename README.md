# TraceIQ

A local research assistant that reads your documents and answers questions with cited reports. No API keys. No cloud. Runs entirely on your machine.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black)](https://ollama.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

Upload PDFs, Word docs, PowerPoints, or spreadsheets. Ask a research question. TraceIQ runs a multi-stage retrieval pipeline and returns a structured report with inline citations back to your source documents. The whole thing runs locally — Ollama handles text generation, sentence-transformers handles embeddings, ChromaDB stores the vectors.

---

## How the pipeline works

```
Your question
    │
    ▼
1. Planner        — breaks the question into 3 focused sub-questions
    │
    ▼
2. Retrieval      — hybrid search (semantic + BM25, fused via RRF) across your docs
    │
    ▼
3. Reranker       — scores and filters the top evidence chunks
    │
    ▼
4. Context Builder — applies token budget, attaches citation labels to each chunk
    │
    ▼
5. Synthesizer    — Ollama LLM writes the report using only the retrieved context
    │
    ▼
6. Gap Analysis   — checks if sub-questions were answered; loops back if not (max 3×)
    │
    ▼
7. Report         — final structured JSON with sections, claims, and citation map
```

---

## Setup

**Prerequisites:** [Ollama](https://ollama.com/download) installed and a model pulled.

```bash
ollama pull mistral
```

Any model works. `llama3`, `phi3`, and `gemma2` are good alternatives.

**Install:**

```bash
git clone https://github.com/your-username/TraceIQ.git
cd TraceIQ
pip install -r requirements.txt
```

If you want to override defaults, copy `.env.example` to `.env` and edit it. This is optional — the defaults work out of the box.

On Windows, if you see `ssl.SSLError: [ASN1] nested asn1 error` during install, run `pip install certifi`. The app patches this automatically at startup.

---

## Running

**One-click (Windows):**
```
Double-click start.bat
```

**One-click (Mac/Linux):**
```bash
./start.sh
```

**Manual (two terminals):**

Terminal 1 — backend:
```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Terminal 2 — open the frontend:
```
Open frontend/index.html in your browser
```

The first time sentence-transformers runs, it downloads the embedding model (~90 MB). After that it uses the local cache.

---

## Configuration

All settings have working defaults. Copy `.env.example` to `.env` to override any of them.

| Variable | Default | What it controls |
|---|---|---|
| `OLLAMA_MODEL` | `mistral` | Which Ollama model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_TEMPERATURE` | `0.25` | Generation temperature |
| `OLLAMA_NUM_CTX` | `8192` | Context window (tokens) |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHROMA_COLLECTION_NAME` | `research_docs` | ChromaDB collection name |
| `RETRIEVAL_TOP_K` | `10` | Chunks fetched per sub-question |
| `RERANK_TOP_K` | `5` | Chunks kept after reranking |
| `MAX_RESEARCH_ITERATIONS` | `3` | Max retrieval loop passes |
| `CONTEXT_TOKEN_BUDGET` | `6000` | Max tokens fed to the synthesizer |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Project structure

```
TraceIQ/
├── app/
│   ├── api.py                   # FastAPI backend — all HTTP endpoints
│   ├── main.py                  # CLI entrypoint (alternative to the web UI)
│   ├── config/
│   │   ├── settings.py          # Env vars and path constants
│   │   └── prompts.py           # LLM prompt templates
│   ├── ingestion/
│   │   ├── embedder.py          # Local sentence-transformers embeddings
│   │   ├── parser.py            # PDF / DOCX / PPTX / CSV parsers
│   │   ├── chunker.py           # Document chunking strategies
│   │   └── indexer.py           # Orchestrates parse → chunk → embed → store
│   ├── vectorstore/
│   │   ├── chroma_store.py      # ChromaDB vector store
│   │   └── bm25_store.py        # BM25 keyword index
│   ├── retrieval/
│   │   ├── hybrid_search.py     # RRF fusion of semantic + keyword results
│   │   ├── reranker.py          # Chunk reranking
│   │   └── context_builder.py   # Token budget + citation label injection
│   ├── agents/
│   │   ├── planner_agent.py     # Breaks topic into sub-questions
│   │   ├── synthesizer_agent.py # Writes the report from retrieved context
│   │   └── gap_analysis_agent.py# Evaluates coverage, decides whether to loop
│   └── orchestrator/
│       ├── state.py             # LangGraph state schema
│       └── graph.py             # Pipeline graph definition
├── frontend/
│   └── index.html               # Single-file web UI (no framework, no build step)
├── config.py                    # Pydantic config models
├── requirements.txt
├── .env.example                 # Copy to .env to override defaults
├── start.bat                    # Windows one-click launcher
├── start.sh                     # Mac/Linux one-click launcher
├── data/
│   ├── uploads/                 # Uploaded documents go here
│   └── chroma_db/               # Vector store (auto-created)
├── outputs/                     # Saved report JSON files (auto-created)
└── logs/                        # Log files (auto-created)
```

---

## API reference

The backend exposes a REST API on port 8000. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/status` | Returns current config and list of uploaded files |
| `POST` | `/upload` | Uploads one or more documents to `data/uploads/` |
| `POST` | `/index` | Parses, chunks, embeds, and indexes uploaded documents |
| `POST` | `/research` | Runs the full pipeline, returns the final report (blocking) |
| `POST` | `/research/stream` | Same as above but streams progress via Server-Sent Events |
| `GET` | `/reports` | Lists saved report JSON files |
| `GET` | `/reports/{filename}` | Returns the contents of a saved report |
| `DELETE` | `/uploads/{filename}` | Deletes an uploaded file |

---

## Tech stack

| Component | Technology |
|---|---|
| Text generation | Ollama (Mistral, LLaMA 3, Phi-3, etc.) |
| Embeddings | sentence-transformers (local, no API) |
| Vector store | ChromaDB |
| Keyword search | rank-bm25 |
| Retrieval fusion | Reciprocal Rank Fusion (RRF) |
| Pipeline orchestration | LangGraph |
| Backend API | FastAPI + uvicorn |
| Frontend | Vanilla HTML/CSS/JS (single file) |
| Document parsing | pdfplumber, python-docx, python-pptx, openpyxl |

---

## Troubleshooting

**Backend starts but frontend says "Failed to fetch"**
The backend is not running or crashed on startup. Check the terminal window that opened with `start.bat` for error output.

**`ModuleNotFoundError: No module named 'app'`**
Run uvicorn from the project root, not from inside the `app/` directory:
```bash
# from TraceIQ/
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

**Ollama connection error**
Make sure Ollama is running. On Windows it usually starts automatically; on Mac/Linux run `ollama serve` in a separate terminal. Check `ollama list` to confirm your model is downloaded.

**Embedding model download hangs**
If your network is restricted, download the model on another machine first:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```
Then set `HF_LOCAL_MODEL_PATH` in `.env` to point at the cached folder.

**Context is always empty (0 chunks retrieved)**
Documents need to be indexed before you can query them. Upload files via the sidebar, then click "Index Documents" and wait for the confirmation.

**Port 8000 already in use**
Another process is using the port. Find and stop it:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Mac/Linux
lsof -i :8000
kill <pid>
```

---

## License

MIT — see [LICENSE](LICENSE)
