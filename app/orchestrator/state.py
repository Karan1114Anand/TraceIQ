"""
app/orchestrator/state.py

LangGraph state schema for the Autonomous Research Analyst pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    """
    Shared mutable state passed between LangGraph nodes.

    Lifecycle:
        topic           → set by user / main entrypoint
        sub_questions   → set by PlannerNode
        retrieved_chunks→ set by RetrievalNode (raw hybrid results)
        reranked_chunks → set by RerankerNode
        context_package → set by ContextBuilderNode {context_str, citation_map}
        draft_report    → set by SynthesizerNode
        gap_result      → set by GapAnalysisNode
        iteration       → incremented each loop
        final_report    → set at finalization
        status          → "running" | "complete" | "error"
        error           → error message if status == "error"
    """

    topic: str
    sub_questions: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    context_package: Dict[str, Any]   # {context_str, citation_map}
    draft_report: Dict[str, Any]
    gap_result: Dict[str, Any]
    iteration: int
    final_report: Dict[str, Any]
    status: str
    error: Optional[str]
