"""
app/retrieval/reranker.py

Two-mode reranker:
 1. Heuristic (default, use_llm=False): dedup by chunk_id + sort by rrf_score.
 2. LLM (use_llm=True): score each chunk with Ollama RERANKING_PROMPT.
    Results are cached by (query_hash, chunk_id) via functools.lru_cache.
"""

from __future__ import annotations

import functools
import hashlib
from typing import Dict, List, Optional

import ollama
from loguru import logger

from app.config.settings import RERANK_TOP_K, OLLAMA_MODEL, OLLAMA_BASE_URL
from app.config.prompts import RERANKING_PROMPT


def _ollama_score(query: str, text: str, model: str, base_url: str) -> float:
    """Call Ollama for a relevance score. Returns 0.0 on failure."""
    try:
        client = ollama.Client(host=base_url)
        prompt = RERANKING_PROMPT.format(query=query, chunk=text[:600])
        resp = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return min(1.0, max(0.0, float(resp["message"]["content"].strip())))
    except Exception as exc:
        logger.warning(f"LLM scoring failed: {exc}")
        return 0.0



class Reranker:
    """
    Reranks retrieved chunks before synthesis.

    Args:
        use_llm: Use Ollama LLM for scoring (slow, accurate).
                 Defaults to False (heuristic dedup + score sort).
        top_k:   Number of chunks to keep after reranking.
    """

    def __init__(
        self,
        use_llm: bool = False,
        top_k: int = RERANK_TOP_K,
        model_name: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.use_llm = use_llm
        self.top_k = top_k
        self.model_name = model_name
        self.base_url = base_url
        if use_llm:
            logger.info("LLM reranker enabled (ollama SDK).")


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Rerank chunks for the given query.

        Returns:
            Top-k chunks sorted by relevance (best first), with 'rerank_score' added.
        """
        top_k = top_k or self.top_k
        if not chunks:
            return []

        # Dedup by chunk_id (always — even in LLM mode)
        seen = set()
        deduped = []
        for c in chunks:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                deduped.append(c)

        if self.use_llm:
            scored = self._llm_score(query, deduped)
        else:
            scored = self._heuristic_score(deduped)


        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        result = scored[:top_k]

        logger.info(f"Reranker: {len(chunks)} → {len(deduped)} deduped → {len(result)} kept.")
        return result

    # ------------------------------------------------------------------
    # Scoring strategies
    # ------------------------------------------------------------------

    def _heuristic_score(self, chunks: List[Dict]) -> List[Dict]:
        """
        Deterministic heuristic: prefer rrf_score if present, fall back to retrieval score.
        """
        result = []
        for c in chunks:
            score = c.get("rrf_score") or c.get("score") or 0.0
            entry = dict(c)
            entry["rerank_score"] = round(float(score), 6)
            result.append(entry)
        return result

    def _llm_score(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Score chunks using Ollama. Results are LRU-cached."""
        result = []
        for c in chunks:
            score = self._cached_score(query, c["chunk_id"], c["text"])
            entry = dict(c)
            entry["rerank_score"] = score
            result.append(entry)
        return result

    @functools.lru_cache(maxsize=512)
    def _cached_score(self, query: str, chunk_id: str, text: str) -> float:
        """LRU-cached LLM relevance score for (query, chunk_id)."""
        return _ollama_score(query, text, self.model_name, self.base_url)

