import sys, os

results = []
results.append(f"Python: {sys.version}")
results.append(f"Executable: {sys.executable}")

# Check fastapi
try:
    import fastapi
    results.append(f"fastapi: {fastapi.__version__}")
except ImportError as e:
    results.append(f"fastapi MISSING: {e}")

# Check uvicorn
try:
    import uvicorn
    results.append(f"uvicorn: OK")
except ImportError as e:
    results.append(f"uvicorn MISSING: {e}")

# Check python_multipart (needed for file uploads)
try:
    import multipart
    results.append("python-multipart: OK")
except ImportError as e:
    results.append(f"python-multipart MISSING: {e}")

# Check app.config.settings
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from app.config.settings import UPLOADS_DIR, OUTPUT_DIR, OLLAMA_MODEL
    results.append(f"app.config.settings: OK (model={OLLAMA_MODEL})")
except Exception as e:
    results.append(f"app.config.settings FAILED: {e}")

# Write results
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_out.txt")
with open(out_path, "w") as f:
    f.write("\n".join(results))

print("\n".join(results))
