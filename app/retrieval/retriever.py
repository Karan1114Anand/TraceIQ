"""
app/retrieval/retriever.py

Thin wrappers around ChromaDB (semantic) and BM25 (keyword) stores
providing a uniform retrieve(query, top_k) → List[Dict] interface.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger

from app.config.settings import RETRIEVAL_TOP_K
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.bm25_store import BM25Store


class SemanticRetriever:
    """Dense semantic retriever backed by ChromaDB."""

    def __init__(self, store: Optional[ChromaVectorStore] = None) -> None:
        self.store = store or ChromaVectorStore()

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
        """Return top-k semantically similar chunks."""
        results = self.store.search(query, top_k=top_k)
        logger.debug(f"SemanticRetriever: {len(results)} results for '{query[:60]}'")
        return results


class KeywordRetriever:
    """Sparse keyword retriever backed by BM25."""

    def __init__(self, store: Optional[BM25Store] = None) -> None:
        self.store = store or BM25Store()

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
        """Return top-k keyword-matched chunks."""
        results = self.store.search(query, top_k=top_k)
        logger.debug(f"KeywordRetriever: {len(results)} results for '{query[:60]}'")
        return results
