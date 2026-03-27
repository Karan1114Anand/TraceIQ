"""
app/config/settings.py

Thin re-export layer — single source of truth is the root config.py.
Adds path constants and direct env-var shortcuts used across the app.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Re-export Pydantic config classes from root config.py (no duplication)
from config import (  # noqa: F401
    OllamaConfig,
    EmbeddingConfig,
    HybridConfig,
    load_ollama_config,
    load_embedding_config,
    load_hybrid_config,
    get_config,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent  # ARA_2/
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for _d in [DATA_DIR, UPLOADS_DIR, CHROMA_DIR, OUTPUT_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.25"))
OLLAMA_TOP_P: float = float(os.getenv("OLLAMA_TOP_P", "0.9"))
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# HuggingFace embeddings (local sentence-transformers — no API calls)
# ---------------------------------------------------------------------------
HF_EMBEDDING_MODEL: str = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# Set this env var to a local folder path to skip downloading
HF_LOCAL_MODEL_PATH: str | None = os.getenv("HF_LOCAL_MODEL_PATH", None)

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = str(CHROMA_DIR)
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "research_docs")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))
RRF_K: int = int(os.getenv("RRF_K", "60"))          # plain RRF constant

# ---------------------------------------------------------------------------
# Agent / pipeline
# ---------------------------------------------------------------------------
MAX_RESEARCH_ITERATIONS: int = int(os.getenv("MAX_RESEARCH_ITERATIONS", "3"))
MIN_CONFIDENCE_SCORE: float = float(os.getenv("MIN_CONFIDENCE_SCORE", "0.7"))
CONTEXT_TOKEN_BUDGET: int = int(os.getenv("CONTEXT_TOKEN_BUDGET", "3000"))

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "research_analyst.log"
