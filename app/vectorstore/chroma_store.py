"""
app/vectorstore/chroma_store.py

ChromaDB vector store using local HuggingFace sentence-transformer embeddings.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from loguru import logger

from app.config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from app.ingestion.embedder import HuggingFaceEmbedder


class ChromaVectorStore:
    """Semantic vector store backed by ChromaDB + local HF embeddings."""

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_directory: str = CHROMA_PERSIST_DIR,
        embedder: Optional[HuggingFaceEmbedder] = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedder = embedder or HuggingFaceEmbedder()

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{collection_name}' ready. "
            f"Docs: {self.collection.count()}"
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Dict], batch_size: int = 100) -> int:
        """
        Add chunks to ChromaDB. Expects each chunk to have:
            chunk_id, text, metadata (with source, page, file_name).
        """
        if not chunks:
            return 0

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            ids = [c["chunk_id"] for c in batch]
            # Chroma metadatas must be flat str/int/float dicts
            metadatas = [
                {k: str(v) for k, v in c.get("metadata", {}).items()} for c in batch
            ]

            embeddings = self.embedder.embed_batch(texts)
            valid = [
                (id_, txt, emb, meta)
                for id_, txt, emb, meta in zip(ids, texts, embeddings, metadatas)
                if emb is not None
            ]
            if not valid:
                continue

            v_ids, v_texts, v_embs, v_metas = zip(*valid)
            try:
                self.collection.add(
                    ids=list(v_ids),
                    documents=list(v_texts),
                    embeddings=list(v_embs),
                    metadatas=list(v_metas),
                )
                added += len(valid)
            except Exception as exc:
                logger.error(f"Chroma add batch error: {exc}")

        logger.success(f"ChromaDB: added {added}/{len(chunks)} chunks.")
        return added

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Semantic search. Returns list of {chunk_id, text, metadata, score, rank}.
        """
        if not query.strip():
            return []
        try:
            q_emb = self.embedder.embed_text(query)
            if not q_emb:
                return []
            results = self.collection.query(
                query_embeddings=[q_emb],
                n_results=min(top_k, max(self.collection.count(), 1)),
                where=filter_metadata,
            )
            formatted = []
            if results["ids"] and results["ids"][0]:
                for rank, (cid, doc, meta, dist) in enumerate(
                    zip(
                        results["ids"][0],
                        results["documents"][0],
                        results["metadatas"][0] if results["metadatas"] else [{}] * top_k,
                        results["distances"][0] if results["distances"] else [1.0] * top_k,
                    ),
                    1,
                ):
                    formatted.append(
                        {
                            "chunk_id": cid,
                            "text": doc,
                            "metadata": meta,
                            "score": round(1.0 - dist, 4),  # cosine similarity
                            "rank": rank,
                        }
                    )
            return formatted
        except Exception as exc:
            logger.error(f"Chroma search error: {exc}")
            return []

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def reset_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{self.collection_name}' reset.")

    def stats(self) -> Dict:
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self.persist_directory,
        }
