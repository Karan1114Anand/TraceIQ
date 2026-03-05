"""
app/ingestion/indexer.py

Orchestrates the full ingestion pipeline:
  parse → chunk → assign stable chunk_ids → embed → store in Chroma + BM25

Chunk IDs are set ONCE here and passed identically to both stores,
ensuring consistent lookup across semantic and keyword retrieval.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from app.config.settings import UPLOADS_DIR
from app.ingestion.parser import ParserDispatcher
from app.ingestion.chunker import DocumentChunker
from app.ingestion.embedder import HuggingFaceEmbedder
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.bm25_store import BM25Store


def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Generate a stable, reproducible chunk ID.
    Format: <doc_hash_8chars>__<chunk_index>
    """
    return f"{doc_id}__{chunk_index:04d}"


def _make_doc_id(file_path: Path) -> str:
    """Short deterministic ID for a document based on its path."""
    return hashlib.md5(str(file_path).encode()).hexdigest()[:8]


class DocumentIndexer:
    """
    Ingestion pipeline: file → parse → chunk → embed → store.
    Guarantees consistent chunk_id and metadata across Chroma and BM25.
    """

    def __init__(
        self,
        chroma_store: Optional[ChromaVectorStore] = None,
        bm25_store: Optional[BM25Store] = None,
        embedder: Optional[HuggingFaceEmbedder] = None,
        use_agentic_chunking: bool = False,   # disabled by default for speed
    ) -> None:
        self.parser = ParserDispatcher()
        self.chunker = DocumentChunker(use_agentic=use_agentic_chunking)
        self.embedder = embedder or HuggingFaceEmbedder()
        self.chroma = chroma_store or ChromaVectorStore(embedder=self.embedder)
        self.bm25 = bm25_store or BM25Store()
        logger.info("DocumentIndexer ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_file(self, file_path: Path) -> int:
        """
        Index a single document file.

        Returns:
            Number of chunks successfully indexed.
        """
        file_path = Path(file_path)
        if not self.parser.is_supported(file_path):
            logger.warning(f"Unsupported file type: {file_path.suffix} — skipping {file_path.name}")
            return 0

        # 1. Parse
        logger.info(f"Indexing: {file_path.name}")
        try:
            parsed = self.parser.parse(file_path)
        except Exception as exc:
            logger.error(f"Parse failed for {file_path.name}: {exc}")
            return 0

        if not parsed or not parsed.get("text", "").strip():
            logger.warning(f"No text extracted from {file_path.name}")
            return 0

        # 2. Chunk
        doc_metadata = {
            "source": parsed["file_name"],
            "file_path": parsed["file_path"],
            "file_type": parsed["file_type"],
            "page_count": parsed.get("page_count", 0),
            **{k: str(v) for k, v in parsed.get("metadata", {}).items()},
        }
        raw_chunks = self.chunker.chunk_document(parsed["text"], metadata=doc_metadata)
        if not raw_chunks:
            logger.warning(f"No chunks produced for {file_path.name}")
            return 0

        # 3. Assign stable chunk IDs (single source of truth)
        doc_id = _make_doc_id(file_path)
        prepared: List[Dict] = []
        for i, chunk in enumerate(raw_chunks):
            stable_id = _make_chunk_id(doc_id, i)
            # Merge chunk-level metadata with doc metadata
            chunk_meta = {
                **doc_metadata,
                "chunk_id": stable_id,
                "chunk_index": i,
                "page": chunk.get("metadata", {}).get("page", ""),
                "chunking_strategy": chunk.get("metadata", {}).get("chunking_strategy", "fixed"),
            }
            prepared.append(
                {
                    "chunk_id": stable_id,
                    "text": chunk["text"],
                    "metadata": chunk_meta,
                }
            )

        # 4. Store — identical chunk_id + metadata to both stores
        chroma_added = self.chroma.add_chunks(prepared)
        bm25_added = self.bm25.add_chunks(prepared)

        logger.success(
            f"Indexed {file_path.name}: {chroma_added} chroma chunks, {bm25_added} BM25 chunks."
        )
        return chroma_added

    def index_directory(
        self,
        directory: Path = UPLOADS_DIR,
        recursive: bool = True,
    ) -> Dict[str, int]:
        """
        Index all supported files in a directory.

        Returns:
            Dict mapping filename → chunk count.
        """
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return {}

        pattern = "**/*" if recursive else "*"
        results: Dict[str, int] = {}
        for file_path in directory.glob(pattern):
            if file_path.is_file() and self.parser.is_supported(file_path):
                results[file_path.name] = self.index_file(file_path)

        total = sum(results.values())
        logger.success(f"Directory indexing complete. {total} chunks across {len(results)} files.")
        return results
