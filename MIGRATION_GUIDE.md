# Migration Guide: Cleaned Architecture

## What Changed

The codebase has been cleaned up to remove all unnecessary HuggingFace Inference API code and focus purely on the hybrid local architecture.

### Before (Bloated)
- `config.py`: 400+ lines with HuggingFace API code, retry logic, validation
- `embedder.py`: Imported bloated config with unused fields
- `.env`: Contained API tokens and unused variables
- Dead code: API validation, retry decorators, HTTP requests

### After (Clean)
- `config.py`: 150 lines, focused on local-only configuration
- `embedder.py`: Minimal imports, clean interface
- `.env`: Only relevant local configuration
- Zero dead code, zero API dependencies

## Breaking Changes

### Config Module

**Old:**
```python
from config import HuggingFaceConfig, load_config_from_env

config = load_config_from_env()
embedder = HuggingFaceEmbedder(config=config)
```

**New:**
```python
from config import EmbedderConfig, load_embedder_config

config = load_embedder_config()
embedder = HuggingFaceEmbedder(config=config)
```

### Removed Functions
- `validate_hf_services()` - Made HTTP requests to HuggingFace API (not needed)
- `HuggingFaceConfig` - Replaced with `EmbedderConfig` and `OllamaConfig`

### Removed Environment Variables
- `HF_API_TOKEN` - Not needed for local embeddings
- `HF_TEXT_MODEL` - Text generation uses Ollama
- `HF_TEXT_FALLBACK` - Not applicable
- `HF_TEXT_TIMEOUT` - Not applicable
- `HF_TEXT_MAX_TOKENS` - Not applicable
- `HF_MAX_RETRIES` - Not needed for local models
- `HF_INITIAL_RETRY_DELAY` - Not needed
- `HF_RETRY_BACKOFF` - Not needed
- `HF_MAX_DELAY` - Not needed

### New Environment Variables
- `HF_CACHE_FOLDER` - Optional cache directory for models
- `OLLAMA_MODEL` - Ollama model name (default: mistral)
- `OLLAMA_BASE_URL` - Ollama server URL
- `OLLAMA_TEMPERATURE` - Sampling temperature
- `OLLAMA_TOP_P` - Nucleus sampling
- `OLLAMA_NUM_CTX` - Context window size
- `OLLAMA_NUM_PREDICT` - Max tokens to generate

## Migration Steps

### Step 1: Update Imports

**If you imported `HuggingFaceConfig`:**
```python
# Old
from config import HuggingFaceConfig, load_config_from_env

# New
from config import EmbedderConfig, load_embedder_config
```

**If you imported `validate_hf_services`:**
```python
# Old
from config import validate_hf_services
status = validate_hf_services()

# New - Remove this code (not needed for local models)
```

### Step 2: Update Configuration

**If you created config manually:**
```python
# Old
config = HuggingFaceConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_timeout=60,
    api_token="hf_xxx"  # Not needed!
)

# New
config = EmbedderConfig(
    model="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=None  # Optional
)
```

**If you loaded from environment:**
```python
# Old
config = load_config_from_env()

# New
config = load_embedder_config()
```

### Step 3: Update .env File

**Old `.env`:**
```bash
HF_API_TOKEN=hf_xxx
HF_TEXT_MODEL=mistralai/Mistral-7B-Instruct-v0.2
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**New `.env`:**
```bash
# Embeddings (optional)
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_CACHE_FOLDER=/path/to/cache

# Ollama (optional)
OLLAMA_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEMPERATURE=0.25
```

### Step 4: Update Ollama Usage (Optional)

**Before (hardcoded):**
```python
from langchain_community.llms import Ollama

llm = Ollama(
    model="mistral",
    base_url="http://localhost:11434",
    temperature=0.25
)
```

**After (with config):**
```python
from langchain_community.llms import Ollama
from config import load_ollama_config

config = load_ollama_config()
llm = Ollama(
    model=config.model,
    base_url=config.base_url,
    temperature=config.temperature,
    top_p=config.top_p,
    num_ctx=config.num_ctx,
    num_predict=config.num_predict
)
```

## What Stays the Same

### Embedder Interface
The `HuggingFaceEmbedder` interface is **unchanged**:
```python
embedder = HuggingFaceEmbedder()

# All methods work the same
embedding = embedder.embed_text("text")
embeddings = embedder.embed_batch(["text1", "text2"])
dim = embedder.get_embedding_dimension()
```

### Ollama Usage
Ollama usage is **unchanged** (unless you want to use the new config system):
```python
from langchain_community.llms import Ollama

llm = Ollama(model="mistral")
response = llm.invoke("prompt")
```

### Pipeline Code
Your `pipeline.ipynb` should work **without changes** (unless it imported removed functions).

## Benefits of Migration

1. **Cleaner Code**: 60% reduction in config.py size
2. **No Confusion**: No API-related code that isn't used
3. **Better Docs**: Clear separation between Ollama and embeddings config
4. **Easier Maintenance**: Less code to maintain and understand
5. **Faster Onboarding**: New developers see only relevant code
6. **Type Safety**: Separate configs for separate concerns

## Testing Your Migration

Run the example script to verify everything works:
```bash
python example_usage.py
```

This will test:
- Embedder initialization
- Text embedding (single and batch)
- Configuration loading
- Custom configuration

## Rollback (If Needed)

If you need to rollback, the old code is in git history:
```bash
git log --oneline  # Find commit before migration
git checkout <commit-hash> config.py embedder.py .env
```

## Questions?

- Check `example_usage.py` for working examples
- Review `README.md` for full documentation
- Check `config.py` for all available options

## Summary

This migration removes ~250 lines of dead code and focuses the codebase on what actually runs: local Ollama + local HuggingFace embeddings. No API keys, no cloud services, no confusion.
