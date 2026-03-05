"""
app/main.py

CLI entrypoint for the Autonomous Research Analyst.

Usage:
    python app/main.py --topic "Impact of AI on healthcare"
    python app/main.py --topic "..." --docs-dir data/uploads
    python app/main.py --topic "..." --dry-run          # skip LLM calls
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Windows SSL cert store patch — MUST run before any langchain/aiohttp import.
# Fixes: ssl.SSLError: [ASN1] nested asn1 error
# Also requires: pip install "aiohttp==3.8.6"
# ---------------------------------------------------------------------------
_orig_create_default_context = ssl.create_default_context


def _safe_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, **kwargs):
    try:
        return _orig_create_default_context(purpose=purpose, **kwargs)
    except ssl.SSLError:
        try:
            import certifi
            return _orig_create_default_context(purpose=purpose, cafile=certifi.where())
        except Exception:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx


ssl.create_default_context = _safe_create_default_context


# ---------------------------------------------------------------------------
# sys.path guard — ensures `from app.*` works whether you run as:
#   python app/main.py  OR  python -m app.main
# ---------------------------------------------------------------------------
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


from loguru import logger  # noqa: E402

# Setup logging
from app.config.settings import LOG_FILE, LOG_LEVEL, OUTPUT_DIR  # noqa: E402

logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL, colorize=True)
logger.add(LOG_FILE, rotation="10 MB", retention="7 days", level=LOG_LEVEL)


def _run_indexer(docs_dir: Path) -> None:
    """Parse and index all documents in docs_dir."""
    from app.ingestion.indexer import DocumentIndexer
    indexer = DocumentIndexer()
    results = indexer.index_directory(docs_dir)
    total = sum(results.values())
    logger.success(f"Indexed {len(results)} files → {total} chunks total.")


def _run_pipeline(topic: str) -> dict:
    """Execute the full LangGraph research pipeline."""
    from app.orchestrator.graph import build_graph
    graph = build_graph()
    initial_state = {
        "topic": topic,
        "sub_questions": [],
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "context_package": {},
        "draft_report": {},
        "gap_result": {},
        "iteration": 0,
        "final_report": {},
        "status": "running",
        "error": None,
    }
    logger.info(f"Starting pipeline for topic: '{topic}'")
    final_state = graph.invoke(initial_state)
    return final_state


def _save_report(topic: str, final_state: dict) -> Path:
    """Save the final report to the outputs directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:40]
    out_path = OUTPUT_DIR / f"report_{safe_topic}_{timestamp}.json"

    report = final_state.get("final_report", {})
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "topic": topic,
                "iterations": final_state.get("iteration", 0),
                "status": final_state.get("status"),
                "report": report,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    logger.success(f"Report saved → {out_path}")
    return out_path


def _dry_run(topic: str) -> None:
    """Verify all imports and module wiring without calling Ollama."""
    logger.info("Dry-run mode: verifying module imports ...")
    from app.config.settings import OLLAMA_MODEL, CHROMA_COLLECTION_NAME
    from app.config.prompts import PLANNER_PROMPT, SYNTHESIZER_PROMPT
    from app.ingestion.embedder import HuggingFaceEmbedder        # noqa
    from app.ingestion.parser import ParserDispatcher              # noqa
    from app.ingestion.chunker import DocumentChunker              # noqa
    from app.vectorstore.bm25_store import BM25Store               # noqa
    from app.vectorstore.chroma_store import ChromaVectorStore     # noqa
    from app.retrieval.hybrid_search import HybridRetriever        # noqa
    from app.retrieval.reranker import Reranker                    # noqa
    from app.retrieval.context_builder import ContextBuilder       # noqa
    from app.agents.planner_agent import PlannerAgent              # noqa
    from app.agents.synthesizer_agent import SynthesizerAgent      # noqa
    from app.agents.gap_analysis_agent import GapAnalysisAgent     # noqa
    from app.orchestrator.state import ResearchState               # noqa
    from app.orchestrator.graph import build_graph                 # noqa

    logger.success(
        f"All imports OK. Configured for Ollama model='{OLLAMA_MODEL}', "
        f"collection='{CHROMA_COLLECTION_NAME}'."
    )
    logger.info(f"Planner prompt (first 120 chars): {PLANNER_PROMPT[:120]!r}")
    logger.info("Dry-run complete — no LLM calls made.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Research Analyst — local-first agentic RAG"
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Research topic or question to analyse.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Directory of documents to ingest before running the pipeline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify module wiring without calling Ollama.",
    )
    args = parser.parse_args()

    if args.dry_run:
        _dry_run(args.topic)
        return

    if args.docs_dir:
        if not args.docs_dir.exists():
            logger.error(f"--docs-dir not found: {args.docs_dir}")
            sys.exit(1)
        _run_indexer(args.docs_dir)

    final_state = _run_pipeline(args.topic)

    if final_state.get("status") == "error":
        logger.error(f"Pipeline failed: {final_state.get('error')}")
        sys.exit(1)

    out_path = _save_report(args.topic, final_state)
    print(f"\n✅ Report saved: {out_path}")


if __name__ == "__main__":
    main()
