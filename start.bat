@echo off
title TraceIQ
echo.
echo  Starting TraceIQ backend on http://localhost:8000 ...
echo.
start "TraceIQ Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn app.api:app --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul
echo  Opening frontend...
start "" "%~dp0frontend\index.html"
echo.
echo  Done. Watch the backend window for logs.
echo  Close this window when finished.
pause
