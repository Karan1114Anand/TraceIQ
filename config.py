"""
Configuration for Autonomous Research Analyst - Hybrid Local Architecture

ACTIVE ARCHITECTURE:
  - Text Generation: Ollama (local) - mistral model
  - Embeddings: HuggingFace sentence-transformers (local) - no API calls
  - Vector Store: ChromaDB (local)
  - Cost: $0.00 - everything runs locally

ACTIVE SETTINGS:
  ✓ OLLAMA_MODEL - Ollama model name (default: mistral)
  ✓ OLLAMA_BASE_URL - Ollama server URL (default: http://localhost:11434)
  ✓ OLLAMA_TEMPERATURE - Sampling temperature (default: 0.25)
  ✓ OLLAMA_TOP_P - Nucleus sampling (default: 0.9)
  ✓ OLLAMA_NUM_CTX - Context window size (default: 4096)
  ✓ OLLAMA_NUM_PREDICT - Max tokens to generate (default: 512)
  ✓ OLLAMA_TIMEOUT - Request timeout in seconds (default: 120)
  ✓ HF_EMBEDDING_MODEL - Local sentence-transformers model (default: sentence-transformers/all-MiniLM-L6-v2)
  ✓ HF_CACHE_FOLDER - Model cache directory (default: ~/.cache/huggingface)
  ✓ HF_EMBEDDING_TIMEOUT - Embedding timeout in seconds (default: 60)

DEPRECATED/UNUSED SETTINGS:
  ✗ HF_API_TOKEN - Not used (embeddings run locally, no API calls)
  ✗ HF_TEXT_MODEL - Not used (text generation uses Ollama)
  ✗ Any HuggingFace Inference API settings - Not used

No API keys required. Everything runs locally.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import os


class OllamaConfig(BaseModel):
    """
    Configuration for local Ollama text generation.
    
    Ollama handles ALL text generation tasks:
    - AgenticChunker (document chunking)
    - PropositionExtractor (proposition extraction)
    - Research agents (PLANNER, SYNTHESIZER, GAP_ANALYSIS, HYDE, RERANKING)
    
    Attributes:
        model: Ollama model name (e.g., mistral, llama3, codellama)
        base_url: Ollama server URL
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        top_p: Nucleus sampling parameter (0.0-1.0)
        num_ctx: Context window size in tokens
        num_predict: Maximum tokens to generate per request
        timeout: Request timeout in seconds
    """
    
    model: str = Field(
        default="mistral",
        description="Ollama model name",
        min_length=1
    )
    
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    
    temperature: float = Field(
        default=0.25,
        description="Sampling temperature (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    top_p: float = Field(
        default=0.9,
        description="Nucleus sampling parameter (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    num_ctx: int = Field(
        default=4096,
        description="Context window size in tokens",
        gt=0,
        le=32768
    )
    
    num_predict: int = Field(
        default=512,
        description="Maximum tokens to generate",
        gt=0,
        le=4096
    )
    
    timeout: int = Field(
        default=120,
        description="Request timeout in seconds",
        gt=0,
        le=600
    )
    
    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate Ollama base URL format."""
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            raise ValueError(f"base_url must start with http:// or https://, got: {v}")
        return v.rstrip('/')
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "mistral",
                "base_url": "http://localhost:11434",
                "temperature": 0.25,
                "top_p": 0.9,
                "num_ctx": 4096,
                "num_predict": 512,
                "timeout": 120
            }
        }


class EmbeddingConfig(BaseModel):
    """
    Configuration for local HuggingFace sentence-transformers embeddings.
    
    Embeddings run entirely locally using sentence-transformers library.
    No API calls, no API keys, no internet required (after initial download).
    
    Common models:
    - sentence-transformers/all-MiniLM-L6-v2: 384-dim, ~90MB, fast
    - sentence-transformers/all-mpnet-base-v2: 768-dim, ~420MB, better quality
    - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: 384-dim, multilingual
    
    Attributes:
        model: HuggingFace sentence-transformers model name
        cache_folder: Optional cache directory for downloaded models
        timeout: Timeout for embedding operations in seconds
    """
    
    model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local sentence-transformers model (runs locally, no API)",
        min_length=1
    )
    
    cache_folder: Optional[str] = Field(
        default=None,
        description="Cache directory for model files (default: ~/.cache/huggingface)"
    )
    
    timeout: int = Field(
        default=60,
        description="Embedding operation timeout in seconds",
        gt=0,
        le=300
    )
    
    @field_validator('model')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name format."""
        v = v.strip()
        if not v:
            raise ValueError("model name cannot be empty")
        # Most sentence-transformers models follow org/model format
        if '/' not in v:
            print(f"Warning: Model '{v}' doesn't follow 'org/model' format. This may not be a valid sentence-transformers model.")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "cache_folder": None,
                "timeout": 60
            }
        }


class HybridConfig(BaseModel):
    """
    Complete configuration for hybrid local architecture.
    
    Architecture:
    - Text Generation: Ollama (local)
    - Embeddings: HuggingFace sentence-transformers (local)
    - Vector Store: ChromaDB (local)
    
    No cloud services, no API keys, $0.00 cost.
    
    Attributes:
        ollama: Ollama configuration for text generation
        embedding: Embedding configuration for sentence-transformers
    """
    
    ollama: OllamaConfig = Field(
        default_factory=OllamaConfig,
        description="Ollama text generation configuration"
    )
    
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Local embedding configuration"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "ollama": {
                    "model": "mistral",
                    "base_url": "http://localhost:11434",
                    "temperature": 0.25,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                    "num_predict": 512,
                    "timeout": 120
                },
                "embedding": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "cache_folder": None,
                    "timeout": 60
                }
            }
        }


def load_ollama_config() -> OllamaConfig:
    """
    Load Ollama configuration from environment variables.
    
    Environment Variables (all optional):
        OLLAMA_MODEL: Model name (default: mistral)
        OLLAMA_BASE_URL: Server URL (default: http://localhost:11434)
        OLLAMA_TEMPERATURE: Sampling temperature (default: 0.25)
        OLLAMA_TOP_P: Nucleus sampling (default: 0.9)
        OLLAMA_NUM_CTX: Context window (default: 4096)
        OLLAMA_NUM_PREDICT: Max tokens (default: 512)
        OLLAMA_TIMEOUT: Request timeout in seconds (default: 120)
    
    Returns:
        OllamaConfig with values from environment or defaults
    
    Example:
        >>> import os
        >>> os.environ['OLLAMA_MODEL'] = 'llama3'
        >>> config = load_ollama_config()
        >>> print(config.model)
        llama3
    """
    config_dict = {}
    
    if model := os.environ.get('OLLAMA_MODEL'):
        config_dict['model'] = model
    
    if base_url := os.environ.get('OLLAMA_BASE_URL'):
        config_dict['base_url'] = base_url
    
    if temperature := os.environ.get('OLLAMA_TEMPERATURE'):
        try:
            config_dict['temperature'] = float(temperature)
        except ValueError:
            print(f"Warning: Invalid OLLAMA_TEMPERATURE '{temperature}', using default 0.25")
    
    if top_p := os.environ.get('OLLAMA_TOP_P'):
        try:
            config_dict['top_p'] = float(top_p)
        except ValueError:
            print(f"Warning: Invalid OLLAMA_TOP_P '{top_p}', using default 0.9")
    
    if num_ctx := os.environ.get('OLLAMA_NUM_CTX'):
        try:
            config_dict['num_ctx'] = int(num_ctx)
        except ValueError:
            print(f"Warning: Invalid OLLAMA_NUM_CTX '{num_ctx}', using default 4096")
    
    if num_predict := os.environ.get('OLLAMA_NUM_PREDICT'):
        try:
            config_dict['num_predict'] = int(num_predict)
        except ValueError:
            print(f"Warning: Invalid OLLAMA_NUM_PREDICT '{num_predict}', using default 512")
    
    if timeout := os.environ.get('OLLAMA_TIMEOUT'):
        try:
            config_dict['timeout'] = int(timeout)
        except ValueError:
            print(f"Warning: Invalid OLLAMA_TIMEOUT '{timeout}', using default 120")
    
    return OllamaConfig(**config_dict)


def load_embedding_config() -> EmbeddingConfig:
    """
    Load embedding configuration from environment variables.
    
    Environment Variables (all optional):
        HF_EMBEDDING_MODEL: sentence-transformers model name
            (default: sentence-transformers/all-MiniLM-L6-v2)
        HF_CACHE_FOLDER: Cache directory for model files
            (default: ~/.cache/huggingface)
        HF_EMBEDDING_TIMEOUT: Embedding timeout in seconds
            (default: 60)
    
    Deprecated Variables (ignored with warning):
        HF_API_TOKEN: Not used (embeddings run locally)
        HF_TEXT_MODEL: Not used (text generation uses Ollama)
    
    Returns:
        EmbeddingConfig with values from environment or defaults
    
    Example:
        >>> import os
        >>> os.environ['HF_EMBEDDING_MODEL'] = 'sentence-transformers/all-mpnet-base-v2'
        >>> config = load_embedding_config()
        >>> print(config.model)
        sentence-transformers/all-mpnet-base-v2
    """
    config_dict = {}
    
    if model := os.environ.get('HF_EMBEDDING_MODEL'):
        config_dict['model'] = model
    
    if cache_folder := os.environ.get('HF_CACHE_FOLDER'):
        config_dict['cache_folder'] = cache_folder
    
    if timeout := os.environ.get('HF_EMBEDDING_TIMEOUT'):
        try:
            config_dict['timeout'] = int(timeout)
        except ValueError:
            print(f"Warning: Invalid HF_EMBEDDING_TIMEOUT '{timeout}', using default 60")
    
    # Warn about deprecated variables
    if os.environ.get('HF_API_TOKEN'):
        print("Warning: HF_API_TOKEN is set but not used. Embeddings run locally without API calls.")
    
    if os.environ.get('HF_TEXT_MODEL'):
        print("Warning: HF_TEXT_MODEL is set but not used. Text generation uses Ollama, not HuggingFace.")
    
    return EmbeddingConfig(**config_dict)


def load_hybrid_config() -> HybridConfig:
    """
    Load complete hybrid configuration from environment variables.
    
    Loads both Ollama and embedding configurations.
    
    Returns:
        HybridConfig with all settings from environment or defaults
    
    Example:
        >>> config = load_hybrid_config()
        >>> print(config.ollama.model)
        mistral
        >>> print(config.embedding.model)
        sentence-transformers/all-MiniLM-L6-v2
    """
    return HybridConfig(
        ollama=load_ollama_config(),
        embedding=load_embedding_config()
    )


# Convenience aliases for backward compatibility
load_embedder_config = load_embedding_config  # Alias for old name


# Module-level convenience function
def get_config() -> HybridConfig:
    """
    Get complete hybrid configuration (convenience function).
    
    Returns:
        HybridConfig with all settings
    """
    return load_hybrid_config()
