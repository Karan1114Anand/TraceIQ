"""
app/ui.py

Streamlit chat interface for the Autonomous Research Analyst.

Usage:
    streamlit run app/ui.py
"""

from __future__ import annotations

import ssl
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Sys-path + SSL patch (same as main.py)
# ---------------------------------------------------------------------------
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
# Imports
# ---------------------------------------------------------------------------
import json
import shutil
import tempfile
import time

import streamlit as st

from app.config.settings import OUTPUT_DIR, UPLOADS_DIR

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Analyst AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []      # list of filenames indexed
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # list of {role, content, report}
if "indexer" not in st.session_state:
    st.session_state.indexer = None
if "ready" not in st.session_state:
    st.session_state.ready = False           # True once at least 1 doc indexed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading AI pipeline…")
def _load_indexer():
    from app.ingestion.indexer import DocumentIndexer
    return DocumentIndexer(use_agentic_chunking=False)


@st.cache_resource(show_spinner="Building pipeline graph…")
def _load_graph():
    from app.orchestrator.graph import build_graph
    return build_graph()


def _run_research(question: str) -> dict:
    graph = _load_graph()
    state = {
        "topic": question,
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
    return graph.invoke(state)


def _save_report(question: str, final_state: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    safe = "".join(c if c.isalnum() else "_" for c in question)[:40]
    out = OUTPUT_DIR / f"report_{safe}_{ts}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "topic": question,
                "iterations": final_state.get("iteration", 0),
                "status": final_state.get("status"),
                "report": final_state.get("final_report", {}),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    return out


def _render_report(report: dict):
    """Render a structured report dict as readable Streamlit components."""
    if not report:
        st.warning("No report content returned.")
        return

    title = report.get("title", "Research Report")
    st.markdown(f"## {title}")

    conf = report.get("confidence_score", 0)
    st.progress(float(conf), text=f"Confidence: {conf:.0%}")

    sections = report.get("sections", [])
    for sec in sections:
        st.markdown(f"### {sec.get('heading', 'Section')}")
        st.markdown(sec.get("content", ""))

        claims = sec.get("claims", [])
        if claims:
            with st.expander(f"📌 {len(claims)} cited claim(s)"):
                for c in claims:
                    chunks = ", ".join(f"`{cid}`" for cid in c.get("source_chunks", []))
                    st.markdown(f"- **{c.get('claim','')}** → {chunks}")

    unanswered = report.get("unanswered_questions", [])
    if unanswered:
        with st.expander("❓ Unanswered questions"):
            for q in unanswered:
                st.markdown(f"- {q}")

    citation_map = report.get("citation_map", {})
    if citation_map:
        with st.expander("📚 Citation map"):
            for cid, meta in citation_map.items():
                st.markdown(
                    f"- `{cid}` → **{meta.get('source','?')}** "
                    f"p.{meta.get('page','?')}"
                )


# ---------------------------------------------------------------------------
# Sidebar — Document upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 Research Analyst AI")
    st.caption("Upload documents, then ask research questions.")
    st.divider()

    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, PPTX, CSV, XLSX",
        type=["pdf", "docx", "pptx", "csv", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        if st.button("📥 Index Documents", use_container_width=True, type="primary"):
            indexer = _load_indexer()
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

            progress = st.progress(0, text="Indexing…")
            new_indexed = []
            for i, f in enumerate(uploaded_files):
                dest = UPLOADS_DIR / f.name
                dest.write_bytes(f.read())
                n = indexer.index_file(dest)
                new_indexed.append(f"{f.name} ({n} chunks)")
                progress.progress((i + 1) / len(uploaded_files), text=f"Indexed: {f.name}")

            st.session_state.indexed_files.extend(new_indexed)
            st.session_state.ready = True
            st.success(f"✅ Indexed {len(uploaded_files)} document(s).")

    if st.session_state.indexed_files:
        st.divider()
        st.subheader("📑 Indexed documents")
        for f in st.session_state.indexed_files:
            st.markdown(f"✅ {f}")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main area — Chat
# ---------------------------------------------------------------------------
st.header("💬 Ask a Research Question")

if not st.session_state.ready:
    st.info("👈 Upload and index at least one document in the sidebar to get started.")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("report"):
                _render_report(msg["report"])
                if msg.get("report_path"):
                    with open(msg["report_path"], "rb") as fh:
                        st.download_button(
                            "⬇️ Download JSON report",
                            data=fh.read(),
                            file_name=Path(msg["report_path"]).name,
                            mime="application/json",
                        )

    # Chat input
    question = st.chat_input("Ask your research question…")
    if question:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Run pipeline
        with st.chat_message("assistant"):
            with st.spinner("Researching… this may take a minute."):
                try:
                    final_state = _run_research(question)
                    report = final_state.get("final_report", {})
                    report_path = _save_report(question, final_state)
                    iters = final_state.get("iteration", 0)

                    summary = (
                        f"Research complete after **{iters}** retrieval loop(s). "
                        f"Confidence: **{report.get('confidence_score', 0):.0%}**"
                    )
                    st.markdown(summary)
                    _render_report(report)

                    with open(report_path, "rb") as fh:
                        st.download_button(
                            "⬇️ Download JSON report",
                            data=fh.read(),
                            file_name=report_path.name,
                            mime="application/json",
                        )

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": summary,
                            "report": report,
                            "report_path": str(report_path),
                        }
                    )

                except Exception as exc:
                    err = f"❌ Pipeline error: {exc}"
                    st.error(err)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err}
                    )
