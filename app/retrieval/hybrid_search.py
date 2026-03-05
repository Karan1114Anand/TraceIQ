"""
app/retrieval/hybrid_search.py

Hybrid retriever using plain Reciprocal Rank Fusion (RRF).

RRF score for chunk i: sum_over_retrievers( 1 / (rrf_k + rank_i) )
No alpha parameter — use rrf_k to tune rank decay.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from loguru import logger

from app.config.settings import RETRIEVAL_TOP_K, RRF_K
from app.retrieval.retriever import SemanticRetriever, KeywordRetriever
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.bm25_store import BM25Store


class HybridRetriever:
    """
    Fuses semantic and keyword retrieval via Reciprocal Rank Fusion.

    Args:
        rrf_k:  RRF constant controlling rank-decay steepness (default 60).
                Higher → gentler decay, lower → gives top ranks more weight.
    """

    def __init__(
        self,
        semantic: Optional[SemanticRetriever] = None,
        keyword: Optional[KeywordRetriever] = None,
        chroma_store: Optional[ChromaVectorStore] = None,
        bm25_store: Optional[BM25Store] = None,
    ) -> None:
        self.semantic = semantic or SemanticRetriever(store=chroma_store)
        self.keyword = keyword or KeywordRetriever(store=bm25_store)

    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        rrf_k: int = RRF_K,
    ) -> List[Dict]:
        """
        Run both retrievers, fuse results with RRF, return top-k chunks.

        Returns:
            List of chunk dicts with extra key 'rrf_score'.
        """
        sem_results = self.semantic.retrieve(query, top_k=top_k)
        kw_results = self.keyword.retrieve(query, top_k=top_k)

        return self._rrf_merge(sem_results, kw_results, top_k=top_k, rrf_k=rrf_k)

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_merge(
        sem: List[Dict],
        kw: List[Dict],
        top_k: int,
        rrf_k: int,
    ) -> List[Dict]:
        """Merge two ranked lists using plain RRF."""
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_data: Dict[str, Dict] = {}

        for rank, item in enumerate(sem, 1):
            cid = item["chunk_id"]
            rrf_scores[cid] += 1.0 / (rrf_k + rank)
            chunk_data[cid] = item  # prefer semantic metadata if conflict

        for rank, item in enumerate(kw, 1):
            cid = item["chunk_id"]
            rrf_scores[cid] += 1.0 / (rrf_k + rank)
            if cid not in chunk_data:
                chunk_data[cid] = item

        sorted_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)[:top_k]

        results = []
        for final_rank, cid in enumerate(sorted_ids, 1):
            entry = dict(chunk_data[cid])
            entry["rrf_score"] = round(rrf_scores[cid], 6)
            entry["rank"] = final_rank
            results.append(entry)

        logger.info(
            f"HybridRetriever: {len(sem)} semantic + {len(kw)} keyword → {len(results)} fused results."
        )
        return results
