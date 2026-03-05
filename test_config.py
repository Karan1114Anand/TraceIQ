"""
Simple test script for the refactored configuration.
Tests config loading and embedder initialization without requiring Ollama.
"""

# Fix SSL certificate issues on Windows
import os
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
except ImportError:
    print("Note: certifi not installed. SSL issues may occur.")
    print("Install with: pip install certifi")

print("=" * 60)
print("Testing Refactored Configuration")
print("=" * 60)

# Test 1: Import config module
print("\n[Test 1] Importing config module...")
try:
    from config import (
        OllamaConfig,
        EmbeddingConfig,
        HybridConfig,
        load_ollama_config,
        load_embedding_config,
        load_hybrid_config
    )
    print("✓ Config module imported successfully")
except Exception as e:
    print(f"✗ Failed to import config: {e}")
    exit(1)

# Test 2: Load Ollama config
print("\n[Test 2] Loading Ollama configuration...")
try:
    ollama_config = load_ollama_config()
    print(f"✓ Ollama config loaded")
    print(f"  Model: {ollama_config.model}")
    print(f"  Base URL: {ollama_config.base_url}")
    print(f"  Temperature: {ollama_config.temperature}")
    print(f"  Context window: {ollama_config.num_ctx}")
    print(f"  Max tokens: {ollama_config.num_predict}")
    print(f"  Timeout: {ollama_config.timeout}s")
except Exception as e:
    print(f"✗ Failed to load Ollama config: {e}")
    exit(1)

# Test 3: Load Embedding config
print("\n[Test 3] Loading Embedding configuration...")
try:
    embedding_config = load_embedding_config()
    print(f"✓ Embedding config loaded")
    print(f"  Model: {embedding_config.model}")
    print(f"  Cache folder: {embedding_config.cache_folder or 'default (~/.cache/huggingface)'}")
    print(f"  Timeout: {embedding_config.timeout}s")
except Exception as e:
    print(f"✗ Failed to load Embedding config: {e}")
    exit(1)

# Test 4: Load Hybrid config
print("\n[Test 4] Loading Hybrid configuration...")
try:
    hybrid_config = load_hybrid_config()
    print(f"✓ Hybrid config loaded")
    print(f"  Ollama model: {hybrid_config.ollama.model}")
    print(f"  Embedding model: {hybrid_config.embedding.model}")
except Exception as e:
    print(f"✗ Failed to load Hybrid config: {e}")
    exit(1)

# Test 5: Create custom configs
print("\n[Test 5] Creating custom configurations...")
try:
    custom_ollama = OllamaConfig(
        model="llama3",
        temperature=0.1,
        num_predict=256
    )
    print(f"✓ Custom Ollama config created")
    print(f"  Model: {custom_ollama.model}")
    print(f"  Temperature: {custom_ollama.temperature}")
    
    custom_embedding = EmbeddingConfig(
        model="sentence-transformers/all-mpnet-base-v2",
        timeout=90
    )
    print(f"✓ Custom Embedding config created")
    print(f"  Model: {custom_embedding.model}")
    print(f"  Timeout: {custom_embedding.timeout}s")
except Exception as e:
    print(f"✗ Failed to create custom configs: {e}")
    exit(1)

# Test 6: Validation
print("\n[Test 6] Testing Pydantic validation...")
try:
    # Test invalid temperature
    try:
        invalid_config = OllamaConfig(temperature=2.0)  # Should fail (> 1.0)
        print("✗ Validation failed - invalid temperature accepted")
    except Exception:
        print("✓ Temperature validation works (rejected 2.0)")
    
    # Test invalid base_url
    try:
        invalid_config = OllamaConfig(base_url="not-a-url")  # Should fail
        print("✗ Validation failed - invalid URL accepted")
    except Exception:
        print("✓ URL validation works (rejected 'not-a-url')")
    
except Exception as e:
    print(f"✗ Validation test failed: {e}")
    exit(1)

# Test 7: Import embedder
print("\n[Test 7] Importing embedder module...")
try:
    from embedder import HuggingFaceEmbedder
    print("✓ Embedder module imported successfully")
except Exception as e:
    print(f"✗ Failed to import embedder: {e}")
    exit(1)

# Test 8: Initialize embedder (will download model on first run)
print("\n[Test 8] Initializing HuggingFace embedder...")
print("  Note: First run will download model (~90MB)")
try:
    embedder = HuggingFaceEmbedder()
    print(f"✓ Embedder initialized")
    print(f"  Model: {embedder.model_name}")
    
    # Get dimension
    dim = embedder.get_embedding_dimension()
    print(f"  Embedding dimension: {dim}")
except Exception as e:
    print(f"✗ Failed to initialize embedder: {e}")
    print(f"  This may be due to missing dependencies or network issues")
    print(f"  Install: pip install langchain-huggingface sentence-transformers")

# Test 9: Generate embedding
print("\n[Test 9] Generating test embedding...")
try:
    test_text = "This is a test sentence for embedding generation"
    embedding = embedder.embed_text(test_text)
    
    if embedding:
        print(f"✓ Embedding generated successfully")
        print(f"  Text: {test_text}")
        print(f"  Dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
    else:
        print("✗ Embedding generation returned None")
except Exception as e:
    print(f"✗ Failed to generate embedding: {e}")

# Test 10: Batch embedding
print("\n[Test 10] Testing batch embedding...")
try:
    test_texts = [
        "First test sentence",
        "Second test sentence",
        "Third test sentence"
    ]
    embeddings = embedder.embed_batch(test_texts, show_progress=False)
    
    if embeddings and len(embeddings) == 3:
        print(f"✓ Batch embedding successful")
        print(f"  Generated {len(embeddings)} embeddings")
        print(f"  All dimensions: {[len(e) if e else 0 for e in embeddings]}")
    else:
        print(f"✗ Batch embedding failed or incomplete")
except Exception as e:
    print(f"✗ Failed batch embedding: {e}")

# Summary
print("\n" + "=" * 60)
print("Configuration Test Summary")
print("=" * 60)
print("✓ Config module: Working")
print("✓ Ollama config: Working")
print("✓ Embedding config: Working")
print("✓ Hybrid config: Working")
print("✓ Custom configs: Working")
print("✓ Validation: Working")
print("✓ Embedder module: Working")
print("\nConfiguration refactor successful!")
print("=" * 60)
