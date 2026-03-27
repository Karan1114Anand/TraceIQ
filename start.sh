#!/bin/bash
echo ""
echo "  TraceIQ — Autonomous Research Analyst"
echo "  ────────────────────────────────────"
echo ""
echo "  Starting backend on http://localhost:8000 ..."
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 2
echo "  Backend running (PID: $BACKEND_PID)"
echo ""
echo "  Open frontend/index.html in your browser to start."
echo ""
echo "  Press Ctrl+C to stop."
wait $BACKEND_PID
