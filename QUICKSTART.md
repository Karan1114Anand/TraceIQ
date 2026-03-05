# Quick Start Guide - Hybrid Local Architecture

## 5-Minute Setup

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [ollama.com](https://ollama.com/download)

### 2. Pull Mistral Model

```bash
ollama pull mistral
```

### 3. Install Python Dependencies

```bash
pip install langchain-community langchain-huggingface sentence-transformers pydantic tqdm chromadb
```

### 4. Run Example

```bash
python example_usage.py
```

That's it! Everything runs locally with zero configuration.

## Architecture

```
Text Generation → Ollama (local mistral model)
Embeddings → HuggingFace sentence-transformers (local, ~90MB)
Vector Store → ChromaDB (local)
Cost → $0.00
```

## Basic Usage

### Embeddings

```python
from embedder import HuggingFaceEmbedder

embedder = HuggingFaceEmbedder()
embedding = embedder.embed_text("Your text here")
print(f"Dimension: {len(embedding)}")  # 384
```

### Text Generation

```python
from langchain_community.llms import Ollama

llm = Ollama(model="mistral")
response = llm.invoke("Explain AI in simple terms")
print(response)
```

### With Configuration

```python
from config import load_hybrid_config
from embedder import HuggingFaceEmbedder
from langchain_community.llms import Ollama

# Load config from environment
config = load_hybrid_config()

# Initialize with config
embedder = HuggingFaceEmbedder(config=config.embedding)
llm = Ollama(
    model=config.ollama.model,
    temperature=config.ollama.temperature
)
```

## Configuration (Optional)

Create `.env` file:

```bash
# Ollama (optional - defaults work)
OLLAMA_MODEL=mistral
OLLAMA_TEMPERATURE=0.25

# Embeddings (optional - defaults work)
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Troubleshooting

### "Connection refused" to Ollama

```bash
# Start Ollama
ollama serve
```

### "Model not found"

```bash
# Pull the model
ollama pull mistral
```

### Slow first run

First-time model download (~90MB for embeddings). This only happens once.

## Next Steps

- Read [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for detailed configuration
- Check [example_usage.py](example_usage.py) for more examples
- Review [README.md](README.md) for complete documentation

## Key Points

✅ No API keys required
✅ Everything runs locally
✅ Works offline (after initial setup)
✅ $0.00 cost
✅ Privacy-preserving (data never leaves your machine)

## Support

- Ollama docs: https://ollama.com/docs
- sentence-transformers: https://www.sbert.net/
- LangChain: https://python.langchain.com/
