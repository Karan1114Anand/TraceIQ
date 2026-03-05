# Autonomous Research Analyst (ARA)

> **A fully local, agentic RAG pipeline** — upload documents, ask research questions, get structured, cited answers. No API keys. No cloud. Everything runs on your machine.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What it does

1. **Upload** PDFs, Word docs, PowerPoints, or spreadsheets via the web UI
2. **Index** — documents are parsed, chunked, embedded (HuggingFace, local), and stored (ChromaDB + BM25)
3. **Ask** a research question in the chat box
4. **Pipeline runs** — an agentic loop (Planner → Hybrid Retrieval → Reranker → Context Builder → Synthesizer → Gap Analysis) produces a structured, cited report
5. **Download** the full JSON report

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | 20 GB |
| Python | 3.10+ | 3.11 or 3.12 |
| GPU | Optional | Speeds up Ollama 3–5× |

---

## Quick-start (5 steps)

### Step 1 — Install Ollama

**Windows:** Download and run the installer from [ollama.com/download](https://ollama.com/download)

Verify it works:
```bash
ollama --version
```

### Step 2 — Pull a language model

```bash
ollama pull mistral
```

> This downloads ~4 GB once. Other good options: `llama3`, `phi3`, `gemma2`.  
> Check what you have: `ollama list`

### Step 3 — Clone the repo

```bash
git clone https://github.com/Karan1114Anand/TraceIQ.git
cd TraceIQ
```

### Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Windows SSL note:** If you see `ssl.SSLError: [ASN1] nested asn1 error` during install, run:
> ```bash
> pip install certifi
> ```
> The app patches this automatically at startup — no extra action needed.

> 💡 The first time `sentence-transformers` runs, it downloads the embedding model (~90 MB). Subsequent starts use the local cache.

### Step 5 — Start the app

```bash
streamlit run app/ui.py
```

Open **http://localhost:8501** in your browser.

---

## Using the Web UI

```
┌─────────────────────────────────────────────────────┐
│  Sidebar                │  Main Area                 │
│─────────────────────────│───────────────────────────│
│  📂 Upload Documents    │  💬 Chat                   │
│  ┌───────────────────┐  │                            │
│  │  Drop files here  │  │  You: "What are the main  │
│  │  PDF/DOCX/PPTX/   │  │   risks of AI in ICUs?"   │
│  │  CSV/XLSX         │  │                            │
│  └───────────────────┘  │  AI: [Structured Report]  │
│  [📥 Index Documents]   │      ├─ Section 1          │
│                         │      ├─ Section 2          │
│  📑 Indexed files:      │      ├─ Citations          │
│  ✅ paper.pdf (42 ch.)  │      └─ ⬇ Download JSON   │
└─────────────────────────────────────────────────────┘
```

1. **Upload** your files using the sidebar uploader
2. Click **"Index Documents"** — wait for the ✅ confirmation
3. Type your question in the chat box at the bottom
4. The answer appears as a structured report with:
   - Section headings
   - Confidence score
   - Source citations (`[CIT:source:chunk_id:page]`)
   - Downloadable JSON

---

---

## Configuration

Copy `.env.example` to `.env` and edit as needed. All settings have sensible defaults — **no changes required** to run.

```bash
cp .env.example .env
```

### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `mistral` | LLM model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_TEMPERATURE` | `0.25` | Generation temperature |
| `OLLAMA_NUM_CTX` | `4096` | Context window (tokens) |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `HF_LOCAL_MODEL_PATH` | _(none)_ | Path to pre-downloaded model folder |
| `CHROMA_COLLECTION_NAME` | `research_docs` | ChromaDB collection |
| `RETRIEVAL_TOP_K` | `10` | Chunks fetched per query |
| `RERANK_TOP_K` | `5` | Chunks kept after reranking |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `MAX_RESEARCH_ITERATIONS` | `3` | Max retrieval loop iterations |
| `CONTEXT_TOKEN_BUDGET` | `3000` | Max tokens passed to synthesizer |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Offline / no-internet embedding

If HuggingFace model download fails (firewall, no internet):

```bash
# 1. Download on a machine with internet access
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# 2. Copy the model folder to the target machine, then set:
HF_LOCAL_MODEL_PATH=C:\path\to\all-MiniLM-L6-v2
```

---

## Project Structure

```
TraceIQ/
├── app/
│   ├── __init__.py              # Windows SSL patch (auto-applied)
│   ├── main.py                  # CLI entrypoint
│   ├── ui.py                    # Streamlit web UI
│   │
│   ├── config/
│   │   ├── settings.py          # Env vars + path constants
│   │   └── prompts.py           # All LLM prompt templates
│   │
│   ├── ingestion/
│   │   ├── embedder.py          # HuggingFace sentence-transformers
│   │   ├── parser.py            # PDF / DOCX / PPTX / CSV parsers
│   │   ├── chunker.py           # Agentic / section / fixed chunking
│   │   └── indexer.py           # parse → chunk → embed → store
│   │
│   ├── vectorstore/
│   │   ├── chroma_store.py      # ChromaDB vector store
│   │   └── bm25_store.py        # BM25 keyword store
│   │
│   ├── retrieval/
│   │   ├── retriever.py         # Semantic + keyword retrieval
│   │   ├── hybrid_search.py     # Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py          # Heuristic / LLM reranker
│   │   └── context_builder.py   # Token budget + [CIT:] labels
│   │
│   ├── agents/
│   │   ├── planner_agent.py     # Decomposes topic → sub-questions
│   │   ├── synthesizer_agent.py # Builds cited research report
│   │   └── gap_analysis_agent.py# Checks coverage, triggers loops
│   │
│   └── orchestrator/
│       ├── state.py             # LangGraph ResearchState schema
│       └── graph.py             # 7-node pipeline with loop
│
├── config.py                    # Pydantic config models (root)
├── requirements.txt
├── .env.example
├── data/
│   ├── uploads/                 # Put your documents here
│   └── chroma_db/               # Vector store (auto-created)
├── outputs/                     # JSON reports (auto-created)
└── logs/                        # Log files (auto-created)
```

---

## How the pipeline works

```
User question
     │
     ▼
┌─────────┐    7 sub-questions
│ Planner │ ──────────────────────────────────────────────┐
└─────────┘                                               │
                                                          ▼
                                              ┌────────────────────┐
                                              │  Hybrid Retrieval  │
                                              │  (Semantic + BM25  │
                                              │   fused via RRF)   │
                                              └────────┬───────────┘
                                                       │
                                                       ▼
                                              ┌────────────────────┐
                                              │    Reranker        │
                                              │  (heuristic / LLM) │
                                              └────────┬───────────┘
                                                       │
                                                       ▼
                                              ┌────────────────────┐
                                              │  Context Builder   │
                                              │  [CIT:src:id:page] │
                                              │  + token budget    │
                                              └────────┬───────────┘
                                                       │
                                                       ▼
                                              ┌────────────────────┐
                                              │   Synthesizer      │
                                              │   (Ollama LLM)     │
                                              └────────┬───────────┘
                                                       │
                                                       ▼
                                              ┌────────────────────┐
                                              │   Gap Analysis     │
                                              │  per sub-question  │
                                              └────────┬───────────┘
                                                       │
                              has gaps? ───────────────┘
                              (loop back to Retrieval, max 3×)
                                       │
                              no gaps ─┴──▶  Final Report (JSON)
```

---

## Troubleshooting

### `ollama serve` says port already in use
Ollama is already running in the background — this is fine. Skip `ollama serve`.

### `ModuleNotFoundError: No module named 'app'`
Run from the project root:
```bash
# ✅ Correct
streamlit run app/ui.py

# ❌ Wrong
cd app && python main.py
```

### `ssl.SSLError: [ASN1] nested asn1 error`
A Windows certificate store issue. Fixed automatically by `app/__init__.py`. If it persists:
```bash
pip install certifi
```

### Embedding model download hangs
Set a local path instead:
```env
HF_LOCAL_MODEL_PATH=C:\Users\YourName\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\<hash>
```

### Context is always empty (0 chunks)
You haven't indexed documents yet. Upload your files via the sidebar and click **"Index Documents"**.

### Changing the LLM model
```bash
ollama pull llama3        # Download first
# Then in .env:
OLLAMA_MODEL=llama3
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (text generation) | [Ollama](https://ollama.com) — Mistral, LLaMA 3, Phi-3, etc. |
| Embeddings | [sentence-transformers](https://sbert.net) — local, no API |
| Vector store | [ChromaDB](https://www.trychroma.com) |
| Keyword search | [rank-bm25](https://github.com/dorianbrown/rank_bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Web UI | [Streamlit](https://streamlit.io) |
| Document parsers | pdfplumber, python-docx, python-pptx, openpyxl |

---

## License

MIT — see [LICENSE](LICENSE)
