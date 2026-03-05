"""
app/vectorstore/bm25_store.py

BM25 sparse index for keyword-based retrieval.
Stores chunk_id + full text + metadata so lookups by chunk_id are always complete.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from app.config.settings import DATA_DIR


class BM25Store:
    """BM25Okapi-based keyword retrieval store."""

    def __init__(self, index_name: str = "bm25_index") -> None:
        self.index_name = index_name
        self.index_path = DATA_DIR / f"{index_name}.pkl"
        self.bm25 = None
        self.corpus: List[str] = []          # parallel to tokenized_corpus
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_metadata: List[Dict] = []  # stores {chunk_id, metadata}
        self._load()
        logger.info(f"BM25Store '{index_name}' ready ({len(self.corpus)} docs).")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Add chunks to the BM25 index.

        Each chunk dict must have 'chunk_id', 'text', and 'metadata'.
        """
        if not chunks:
            return 0
        added = 0
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            self.tokenized_corpus.append(self._tokenize(text))
            self.corpus.append(text)
            self.chunk_metadata.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "metadata": chunk.get("metadata", {}),
                }
            )
            added += 1

        if self.tokenized_corpus:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        self._save()
        logger.success(f"BM25: added {added} chunks. Total: {len(self.corpus)}.")
        return added

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Keyword search using BM25.

        Returns list of {chunk_id, text, metadata, score, rank}.
        """
        if not query.strip() or not self.bm25:
            return []
        try:
            tokens = self._tokenize(query)
            scores = self.bm25.get_scores(tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = []
            for rank, idx in enumerate(top_indices, 1):
                if scores[idx] > 0:
                    results.append(
                        {
                            "chunk_id": self.chunk_metadata[idx]["chunk_id"],
                            "text": self.corpus[idx],
                            "metadata": self.chunk_metadata[idx]["metadata"],
                            "score": float(scores[idx]),
                            "rank": rank,
                        }
                    )
            return results
        except Exception as exc:
            logger.error(f"BM25 search error: {exc}")
            return []

    def get_by_chunk_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve a chunk by its chunk_id."""
        for i, meta in enumerate(self.chunk_metadata):
            if meta["chunk_id"] == chunk_id:
                return {
                    "chunk_id": chunk_id,
                    "text": self.corpus[i],
                    "metadata": meta["metadata"],
                }
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            with open(self.index_path, "wb") as fh:
                pickle.dump(
                    {
                        "bm25": self.bm25,
                        "corpus": self.corpus,
                        "tokenized_corpus": self.tokenized_corpus,
                        "chunk_metadata": self.chunk_metadata,
                    },
                    fh,
                )
        except Exception as exc:
            logger.error(f"BM25 save failed: {exc}")

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            with open(self.index_path, "rb") as fh:
                data = pickle.load(fh)
            self.bm25 = data["bm25"]
            self.corpus = data["corpus"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.chunk_metadata = data.get("chunk_metadata", [])
            logger.info(f"Loaded BM25 index: {len(self.corpus)} docs.")
        except Exception as exc:
            logger.error(f"BM25 load failed: {exc}")

    def reset(self) -> None:
        self.bm25 = None
        self.corpus = []
        self.tokenized_corpus = []
        self.chunk_metadata = []
        if self.index_path.exists():
            self.index_path.unlink()
        logger.info("BM25 index reset.")

    def stats(self) -> Dict:
        return {"index_name": self.index_name, "document_count": len(self.corpus)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()
