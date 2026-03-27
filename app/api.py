"""
app/api.py

FastAPI backend for TraceIQ.

Run with:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

# ---------------------------------------------------------------------------
# Windows SSL cert store patch — MUST run before any langchain/aiohttp import.
# ---------------------------------------------------------------------------
import ssl as _ssl

_orig_create_default_context = _ssl.create_default_context


def _safe_create_default_context(purpose=_ssl.Purpose.SERVER_AUTH, **kwargs):
    try:
        return _orig_create_default_context(purpose=purpose, **kwargs)
    except _ssl.SSLError:
        try:
            import certifi
            return _orig_create_default_context(purpose=purpose, cafile=certifi.where())
        except Exception:
            ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = False
            ctx.verify_mode = _ssl.CERT_NONE
            return ctx


_ssl.create_default_context = _safe_create_default_context

# ---------------------------------------------------------------------------
# sys.path guard — ensures `from app.*` works when run via uvicorn
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ---------------------------------------------------------------------------
# Standard library & third-party imports
# ---------------------------------------------------------------------------
import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Project imports (after sys.path is set)
# ---------------------------------------------------------------------------
from app.config.settings import (
    CHROMA_COLLECTION_NAME,
    HF_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OUTPUT_DIR,
    UPLOADS_DIR,
)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[TraceIQ] API server started -- http://localhost:8000")
    yield
    print("[TraceIQ] API server shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TraceIQ API",
    description="Local agentic research analyst — FastAPI backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    filenames: Optional[List[str]] = []


class ResearchRequest(BaseModel):
    topic: str


# ---------------------------------------------------------------------------
# Helper — save report (mirrors _save_report in main.py)
# ---------------------------------------------------------------------------


def _save_report_sync(topic: str, final_state: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:40]
    out_path = OUTPUT_DIR / f"report_{safe_topic}_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "topic": topic,
                "iterations": final_state.get("iteration", 0),
                "status": final_state.get("status"),
                "report": final_state.get("final_report", {}),
                "sub_questions": final_state.get("sub_questions", []),
                "gap_result": final_state.get("gap_result", {}),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    return out_path


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload one or more documents to UPLOADS_DIR."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    uploaded: List[str] = []

    for file in files:
        dest = UPLOADS_DIR / file.filename
        content = await file.read()
        await asyncio.to_thread(dest.write_bytes, content)
        uploaded.append(file.filename)

    return {"uploaded": uploaded}


@app.post("/index")
async def index_files(req: IndexRequest):
    """
    Index uploaded documents into Chroma + BM25.
    If filenames list is empty, indexes everything in UPLOADS_DIR.
    """

    def _do_index() -> Dict[str, int]:
        from app.ingestion.indexer import DocumentIndexer

        indexer = DocumentIndexer()
        results: Dict[str, int] = {}

        filenames = req.filenames if req.filenames else None

        if not filenames:
            # Index all files in UPLOADS_DIR
            raw = indexer.index_directory(UPLOADS_DIR)
            results.update(raw)
        else:
            for fname in filenames:
                file_path = UPLOADS_DIR / fname
                try:
                    count = indexer.index_file(file_path)
                    results[fname] = count
                except Exception:
                    results[fname] = 0

        return results

    indexed = await asyncio.to_thread(_do_index)
    return {"indexed": indexed}


@app.post("/research")
async def run_research(req: ResearchRequest):
    """
    Execute the full LangGraph research pipeline for the given topic.
    This is a slow operation (30–120s); runs in a thread pool.
    """
    topic = (req.topic or "").strip()
    if not topic:
        raise HTTPException(status_code=422, detail="topic must be a non-empty string")

    def _do_research() -> dict:
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
        return graph.invoke(initial_state)

    final_state = await asyncio.to_thread(_do_research)

    if final_state.get("status") == "error":
        error_msg = final_state.get("error") or "Pipeline encountered an unknown error"
        raise HTTPException(status_code=500, detail=str(error_msg))

    # Save report to disk (best-effort)
    try:
        await asyncio.to_thread(_save_report_sync, topic, final_state)
    except Exception as exc:
        print(f"[TraceIQ] Warning: could not save report — {exc}")

    # Return the complete final_state as JSON
    # Convert non-JSON-serialisable types gracefully
    try:
        return json.loads(json.dumps(final_state, default=str))
    except Exception:
        return {"status": final_state.get("status"), "error": "Result serialisation failed"}


@app.post("/research/stream")
async def stream_research(req: ResearchRequest):
    """
    Execute the research pipeline and stream real-time progress via SSE.
    Each pipeline node emits an event when it starts and when it completes.
    """
    from fastapi.responses import StreamingResponse as _SR

    topic = (req.topic or "").strip()
    if not topic:
        raise HTTPException(status_code=422, detail="topic must be a non-empty string")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    NODE_TO_STAGE = {
        "plan": "planner",
        "retrieve": "retrieval",
        "rerank": "reranker",
        "build_context": "context",
        "synthesize": "synth",
        "analyze_gaps": "gap",
        "finalize": "report",
    }
    NEXT_NODE = {
        "plan": "retrieve",
        "retrieve": "rerank",
        "rerank": "build_context",
        "build_context": "synthesize",
        "synthesize": "analyze_gaps",
    }

    def _emit(data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, data)

    def _run() -> None:
        try:
            from app.orchestrator.graph import build_graph
            from app.config.settings import MAX_RESEARCH_ITERATIONS, MIN_CONFIDENCE_SCORE

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

            _emit({"type": "stage", "stage": "planner", "status": "active"})

            final_state: dict = {}
            for chunk in graph.stream(initial_state):
                for node_name, state_update in chunk.items():
                    stage_id = NODE_TO_STAGE.get(node_name)
                    if stage_id:
                        _emit({"type": "stage", "stage": stage_id, "status": "done"})

                    if node_name == "analyze_gaps":
                        gap = state_update.get("gap_result", {})
                        iteration = state_update.get("iteration", 0)
                        confidence = state_update.get("draft_report", {}).get("confidence_score", 0.0)
                        if (
                            gap.get("has_gaps")
                            and iteration < MAX_RESEARCH_ITERATIONS
                            and confidence < MIN_CONFIDENCE_SCORE
                        ):
                            _emit({"type": "stage", "stage": "retrieval", "status": "active"})
                        else:
                            _emit({"type": "stage", "stage": "report", "status": "active"})
                    elif node_name in NEXT_NODE:
                        next_stage = NODE_TO_STAGE.get(NEXT_NODE[node_name])
                        if next_stage:
                            _emit({"type": "stage", "stage": next_stage, "status": "active"})

                    final_state = state_update

            if final_state.get("status") == "error":
                _emit({"type": "error", "message": final_state.get("error", "Pipeline error")})
            else:
                try:
                    _save_report_sync(topic, final_state)
                except Exception:
                    pass
                try:
                    result = json.loads(json.dumps(final_state, default=str))
                except Exception:
                    result = {"status": "complete"}
                _emit({"type": "result", "data": result})

        except Exception as exc:
            _emit({"type": "error", "message": str(exc)})
        finally:
            _emit({"type": "done"})

    loop.run_in_executor(None, _run)

    async def event_stream():
        while True:
            item = await queue.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item.get("type") in ("done", "error"):
                break

    return _SR(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/status")
async def get_status():
    """Return system configuration and list of uploaded files."""

    def _list_uploads() -> List[str]:
        if not UPLOADS_DIR.exists():
            return []
        return [f.name for f in UPLOADS_DIR.iterdir() if f.is_file()]

    indexed_files = await asyncio.to_thread(_list_uploads)

    return {
        "ollama_model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_BASE_URL,
        "embedding_model": HF_EMBEDDING_MODEL,
        "chroma_collection": CHROMA_COLLECTION_NAME,
        "uploads_dir": str(UPLOADS_DIR.relative_to(_project_root)),
        "indexed_files": indexed_files,
    }


@app.get("/reports")
async def list_reports():
    """List all saved report JSON files."""

    def _list() -> List[str]:
        if not OUTPUT_DIR.exists():
            return []
        return [f.name for f in OUTPUT_DIR.glob("*.json")]

    reports = await asyncio.to_thread(_list)
    return {"reports": reports}


@app.get("/reports/{filename}")
async def get_report(filename: str):
    """Return the contents of a saved report JSON file."""

    def _read() -> dict:
        path = OUTPUT_DIR / filename
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(filename)
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    try:
        data = await asyncio.to_thread(_read)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Report '{filename}' not found")


@app.delete("/uploads/{filename}")
async def delete_upload(filename: str):
    """Delete an uploaded file."""

    def _delete() -> None:
        path = UPLOADS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(filename)
        path.unlink()

    try:
        await asyncio.to_thread(_delete)
        return {"deleted": filename}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
