"""
app/retrieval/context_builder.py

Formats reranked chunks into an LLM-ready context string with explicit
citation labels: [CIT:source_name:chunk_id:page]

Input:  reranked_chunks  List[Dict] — each chunk has text + metadata
Output: {
    "context_str":   str   — labeled context within token budget
    "citation_map":  Dict  — chunk_id → {source, page, file_name}
}
"""

from __future__ import annotations

from typing import Dict, List

from loguru import logger

from app.config.settings import CONTEXT_TOKEN_BUDGET


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


class ContextBuilder:
    """
    Builds a single context string from reranked chunks, respecting a
    token budget and inserting [CIT:source_name:chunk_id:page] labels.
    """

    def __init__(self, token_budget: int = CONTEXT_TOKEN_BUDGET) -> None:
        self.token_budget = token_budget

    def build(self, reranked_chunks: List[Dict]) -> Dict:
        """
        Build context package from reranked chunks.

        Args:
            reranked_chunks: List of chunk dicts (must have chunk_id + metadata).

        Returns:
            {
                "context_str":  Labeled context string within budget,
                "citation_map": {chunk_id: {source, page, file_name, ...}}
            }
        """
        context_parts: List[str] = []
        citation_map: Dict[str, Dict] = {}
        tokens_used = 0

        for chunk in reranked_chunks:
            chunk_id = chunk["chunk_id"]
            text = chunk.get("text", "").strip()
            meta = chunk.get("metadata", {})

            source = meta.get("source", meta.get("file_name", "unknown"))
            page = meta.get("page", meta.get("page_number", "?"))
            file_name = meta.get("file_name", source)

            # Build the citation label
            label = f"[CIT:{source}:{chunk_id}:{page}]"
            labeled = f"{label}\n{text}"

            chunk_tokens = _estimate_tokens(labeled)
            if tokens_used + chunk_tokens > self.token_budget:
                logger.info(
                    f"Token budget ({self.token_budget}) reached at chunk {chunk_id}. "
                    f"Stopping context build ({tokens_used} tokens used)."
                )
                break

            context_parts.append(labeled)
            citation_map[chunk_id] = {
                "source": source,
                "page": page,
                "file_name": file_name,
                "chunk_id": chunk_id,
            }
            tokens_used += chunk_tokens

        context_str = "\n\n".join(context_parts)
        logger.info(
            f"ContextBuilder: {len(citation_map)} chunks, ~{tokens_used} tokens."
        )
        return {
            "context_str": context_str,
            "citation_map": citation_map,
        }


def build_context(
    reranked_chunks: List[Dict],
    token_budget: int = CONTEXT_TOKEN_BUDGET,
) -> Dict:
    """Convenience function."""
    return ContextBuilder(token_budget=token_budget).build(reranked_chunks)
