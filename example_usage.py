"""
Example usage of the Autonomous Research Analyst hybrid local architecture.

Demonstrates:
- Local Ollama text generation
- Local HuggingFace embeddings
- Unified configuration management
"""

# Fix SSL certificate issues on Windows
import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from embedder import HuggingFaceEmbedder
from config import (
    load_hybrid_config,
    load_ollama_config,
    load_embedding_config,
    OllamaConfig,
    EmbeddingConfig
)
from langchain_community.llms import Ollama


def example_embeddings():
    """Example: Generate embeddings locally"""
    print("=" * 60)
    print("Example 1: Local Embeddings")
    print("=" * 60)
    
    # Initialize embedder (uses defaults or environment variables)
    embedder = HuggingFaceEmbedder()
    
    # Single text embedding
    text = "Machine learning is transforming business intelligence"
    embedding = embedder.embed_text(text)
    print(f"\nText: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Batch embedding
    texts = [
        "Artificial intelligence in healthcare",
        "Cloud computing trends 2024",
        "Sustainable energy solutions"
    ]
    embeddings = embedder.embed_batch(texts, show_progress=True)
    print(f"\nBatch embedded {len(embeddings)} texts")
    
    # Get dimension
    dim = embedder.get_embedding_dimension()
    print(f"Model produces {dim}-dimensional embeddings")


def example_text_generation():
    """Example: Generate text with Ollama"""
    print("\n" + "=" * 60)
    print("Example 2: Local Text Generation")
    print("=" * 60)
    
    # Load config from environment
    config = load_ollama_config()
    
    # Initialize Ollama
    llm = Ollama(
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        top_p=config.top_p,
        num_ctx=config.num_ctx,
        num_predict=config.num_predict
    )
    
    # Generate text
    prompt = "Explain the benefits of hybrid local AI architectures in 2-3 sentences."
    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")
    
    response = llm.invoke(prompt)
    print(f"\nResponse: {response}")


def example_custom_config():
    """Example: Custom configuration"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)
    
    # Custom embedder config
    embedder_config = EmbeddingConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=None,
        timeout=60
    )
    
    embedder = HuggingFaceEmbedder(config=embedder_config)
    print(f"\nEmbedder model: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    
    # Custom Ollama config
    ollama_config = OllamaConfig(
        model="mistral",
        temperature=0.1,
        num_predict=256,
        timeout=120
    )
    
    print(f"\nOllama model: {ollama_config.model}")
    print(f"Temperature: {ollama_config.temperature}")
    print(f"Max tokens: {ollama_config.num_predict}")
    print(f"Timeout: {ollama_config.timeout}s")


def example_hybrid_config():
    """Example: Load complete hybrid configuration"""
    print("\n" + "=" * 60)
    print("Example 4: Hybrid Configuration")
    print("=" * 60)
    
    # Load complete config
    config = load_hybrid_config()
    
    print("\nOllama Configuration:")
    print(f"  Model: {config.ollama.model}")
    print(f"  Base URL: {config.ollama.base_url}")
    print(f"  Temperature: {config.ollama.temperature}")
    print(f"  Context window: {config.ollama.num_ctx}")
    print(f"  Max tokens: {config.ollama.num_predict}")
    print(f"  Timeout: {config.ollama.timeout}s")
    
    print("\nEmbedding Configuration:")
    print(f"  Model: {config.embedding.model}")
    print(f"  Cache folder: {config.embedding.cache_folder or 'default'}")
    print(f"  Timeout: {config.embedding.timeout}s")
    
    # Use the config
    embedder = HuggingFaceEmbedder(config=config.embedding)
    print(f"\nEmbedding dimension: {embedder.get_embedding_dimension()}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Autonomous Research Analyst - Example Usage")
    print("Hybrid Local Architecture (Ollama + HuggingFace)")
    print("=" * 60)
    
    try:
        # Example 1: Embeddings
        example_embeddings()
        
        # Example 2: Text generation (requires Ollama running)
        print("\n\nNote: Example 2 requires Ollama to be running.")
        print("Start Ollama with: ollama serve")
        print("Pull model with: ollama pull mistral")
        
        response = input("\nRun text generation example? (y/n): ")
        if response.lower() == 'y':
            example_text_generation()
        
        # Example 3: Custom config
        example_custom_config()
        
        # Example 4: Hybrid config
        example_hybrid_config()
        
        print("\n" + "=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Ensure mistral model is pulled: ollama pull mistral")
        print("3. Check internet connection for first-time model download")


if __name__ == "__main__":
    main()
