#!/bin/bash
echo "Starting TraceIQ backend..."
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload &
sleep 2
echo "Open frontend/index.html in your browser"
