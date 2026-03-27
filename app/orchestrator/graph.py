"""
app/orchestrator/graph.py

LangGraph pipeline for the Autonomous Research Analyst.

Flow:
    plan → retrieve → rerank → build_context → synthesize → analyze_gaps
                                                                  |
              (has_gaps AND iteration < MAX_ITERATIONS) ──────→ retrieve [loop]
              else                                        ──────→ finalize
"""

from __future__ import annotations

from typing import Dict, Any

from loguru import logger

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise ImportError(
        "langgraph is not installed. Fix: pip install langgraph"
    )

from app.orchestrator.state import ResearchState
from app.config.settings import (
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    RRF_K,
    MAX_RESEARCH_ITERATIONS,
    MIN_CONFIDENCE_SCORE,
)
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.context_builder import ContextBuilder
from app.agents.planner_agent import PlannerAgent
from app.agents.synthesizer_agent import SynthesizerAgent
from app.agents.gap_analysis_agent import GapAnalysisAgent


# ---------------------------------------------------------------------------
# Instantiate shared components (singletons within a pipeline run)
# ---------------------------------------------------------------------------

_hybrid_retriever = HybridRetriever()
_reranker = Reranker(use_llm=False, top_k=RERANK_TOP_K)
_context_builder = ContextBuilder()
_planner = PlannerAgent()
_synthesizer = SynthesizerAgent()
_gap_agent = GapAnalysisAgent()


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def plan_node(state: ResearchState) -> ResearchState:
    """Decompose topic into sub-questions."""
    logger.info(f"[plan] topic='{state['topic']}'")
    try:
        qs = _planner.plan(state["topic"])
        return {**state, "sub_questions": qs, "iteration": state.get("iteration", 0)}
    except Exception as exc:
        logger.error(f"plan_node error: {exc}")
        return {**state, "status": "error", "error": str(exc)}


def retrieve_node(state: ResearchState) -> ResearchState:
    """Run hybrid retrieval for each sub-question, merge results."""
    logger.info(f"[retrieve] iteration={state.get('iteration', 0)}")
    sub_questions = state.get("sub_questions", [])
    gap_result = state.get("gap_result", {})

    # On re-loop: only retrieve for sub-questions with gaps
    if gap_result.get("has_gaps") and gap_result.get("gap_details"):
        queries = [
            g["follow_up_query"]
            for g in gap_result["gap_details"]
            if g.get("follow_up_query")
        ]
    else:
        queries = [q["question"] for q in sub_questions]

    all_chunks: Dict[str, Any] = {}
    for query in queries:
        hits = _hybrid_retriever.retrieve(query, top_k=RETRIEVAL_TOP_K, rrf_k=RRF_K)
        for h in hits:
            # Deduplicate by chunk_id, keep highest rrf_score
            cid = h["chunk_id"]
            if cid not in all_chunks or h.get("rrf_score", 0) > all_chunks[cid].get("rrf_score", 0):
                all_chunks[cid] = h

    return {**state, "retrieved_chunks": list(all_chunks.values())}


def rerank_node(state: ResearchState) -> ResearchState:
    """Rerank retrieved chunks."""
    logger.info("[rerank]")
    topic = state.get("topic", "")
    chunks = state.get("retrieved_chunks", [])
    reranked = _reranker.rerank(query=topic, chunks=chunks, top_k=RERANK_TOP_K)
    return {**state, "reranked_chunks": reranked}


def build_context_node(state: ResearchState) -> ResearchState:
    """Build labeled context string + citation map from reranked chunks."""
    logger.info("[build_context]")
    reranked = state.get("reranked_chunks", [])
    context_package = _context_builder.build(reranked)
    return {**state, "context_package": context_package}


def synthesize_node(state: ResearchState) -> ResearchState:
    """Generate draft research report."""
    logger.info("[synthesize]")
    report = _synthesizer.synthesize(
        sub_questions=state.get("sub_questions", []),
        context_package=state.get("context_package", {}),
    )
    return {**state, "draft_report": report}


def analyze_gaps_node(state: ResearchState) -> ResearchState:
    """Evaluate per-sub-question coverage and decide to loop or finalize."""
    iteration = state.get("iteration", 0) + 1
    logger.info(f"[analyze_gaps] iteration={iteration}")
    gap = _gap_agent.analyze(
        report=state.get("draft_report", {}),
        sub_questions=state.get("sub_questions", []),
        iteration=iteration,
    )
    return {**state, "gap_result": gap, "iteration": iteration}


def finalize_node(state: ResearchState) -> ResearchState:
    """Mark pipeline as complete, copy draft to final_report."""
    logger.info("[finalize]")
    return {
        **state,
        "final_report": state.get("draft_report", {}),
        "status": "complete",
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def _should_loop(state: ResearchState) -> str:
    gap = state.get("gap_result", {})
    iteration = state.get("iteration", 0)

    # Hard cap on iterations
    if iteration >= MAX_RESEARCH_ITERATIONS:
        return "finalize"

    # Skip loop if the synthesizer already reported high confidence
    draft = state.get("draft_report", {})
    confidence = draft.get("confidence_score", 0.0)
    if confidence >= MIN_CONFIDENCE_SCORE:
        logger.info(
            f"[loop gate] confidence={confidence:.2f} >= threshold={MIN_CONFIDENCE_SCORE} "
            f"— skipping further retrieval."
        )
        return "finalize"

    if gap.get("has_gaps"):
        return "retrieve"
    return "finalize"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph research pipeline.

    Returns a compiled graph that accepts ResearchState and returns
    the final ResearchState.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("analyze_gaps", analyze_gaps_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "build_context")
    graph.add_edge("build_context", "synthesize")
    graph.add_edge("synthesize", "analyze_gaps")
    graph.add_conditional_edges("analyze_gaps", _should_loop, {"retrieve": "retrieve", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile()
