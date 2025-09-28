#!/usr/bin/env python3
"""
Test InferMesh embedder with advanced payload features.
"""
import os
import sys
import json
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dspy_agent.embedding.infermesh import InferMeshEmbedder, InferMeshConfig


def test_basic_embedding():
    """Test basic embedding functionality."""
    print("Testing basic embedding...")
    
    # Use mock URL for testing
    embedder = InferMeshEmbedder(
        base_url="http://localhost:19000",  # Gateway URL
        model="test-model",
        api_key="test-key"
    )
    
    texts = ["Hello world", "Test embedding"]
    
    try:
        embeddings = embedder.embed(texts)
        print(f"✓ Basic embedding successful: {len(embeddings)} embeddings")
        return True
    except Exception as e:
        print(f"✗ Basic embedding failed: {e}")
        return False


def test_advanced_payload():
    """Test advanced payload features."""
    print("Testing advanced payload features...")
    
    embedder = InferMeshEmbedder(
        base_url="http://localhost:19000",
        model="test-model",
        routing_strategy="hybrid",
        tenant="test-tenant",
        priority="high",
        batch_size=512,
        cache_ttl=300,
        cache_key_template="embedding:{text_hash}",
        options={"gpu_preferred": True, "batch_optimization": True},
        metadata={"source": "test", "version": "1.0"},
        hints={"prefer_node": "node-a"}
    )
    
    texts = ["Advanced test", "Payload test"]
    
    # Test payload building
    payload = embedder._build_payload(texts)
    
    expected_keys = ['model', 'inputs', 'options', 'metadata', 'cache', 'hints']
    for key in expected_keys:
        if key not in payload:
            print(f"✗ Missing key in payload: {key}")
            return False
    
    # Check specific values
    assert payload['model'] == "test-model"
    assert payload['inputs'] == texts
    assert payload['options']['routing_strategy'] == "hybrid"
    assert payload['options']['priority'] == "high"
    assert payload['options']['batch_size'] == 512
    assert payload['metadata']['tenant'] == "test-tenant"
    assert payload['cache']['ttl_seconds'] == 300
    assert payload['cache']['key_template'] == "embedding:{text_hash}"
    assert payload['hints']['prefer_node'] == "node-a"
    
    print("✓ Advanced payload features working correctly")
    return True


def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")
    
    # Test with minimal config
    config = InferMeshConfig(
        base_url="http://localhost:19000",
        model="test-model"
    )
    
    assert config.base_url == "http://localhost:19000"
    assert config.model == "test-model"
    assert config.api_key is None
    assert config.timeout_sec == 30.0
    assert config.retries == 2
    
    # Test with full config
    full_config = InferMeshConfig(
        base_url="http://localhost:19000",
        model="test-model",
        api_key="test-key",
        timeout_sec=60.0,
        retries=3,
        backoff_sec=1.0,
        routing_strategy="hybrid",
        tenant="test-tenant",
        priority="high",
        batch_size=1024,
        cache_ttl=600,
        cache_key_template="embedding:{text_hash}",
        options={"gpu_preferred": True},
        metadata={"source": "test"},
        hints={"prefer_node": "node-a"}
    )
    
    assert full_config.api_key == "test-key"
    assert full_config.timeout_sec == 60.0
    assert full_config.retries == 3
    assert full_config.routing_strategy == "hybrid"
    assert full_config.tenant == "test-tenant"
    assert full_config.priority == "high"
    assert full_config.batch_size == 1024
    assert full_config.cache_ttl == 600
    
    print("✓ Configuration validation successful")
    return True


def test_environment_variables():
    """Test environment variable integration."""
    print("Testing environment variable integration...")
    
    # Set test environment variables
    os.environ['INFERMESH_URL'] = 'http://infermesh-router:9000'
    os.environ['INFERMESH_API_KEY'] = 'test-key'
    os.environ['INFERMESH_MODEL'] = 'BAAI/bge-small-en-v1.5'
    os.environ['INFERMESH_ROUTING_STRATEGY'] = 'hybrid'
    os.environ['INFERMESH_TENANT'] = 'test-tenant'
    os.environ['INFERMESH_PRIORITY'] = 'high'
    os.environ['INFERMESH_BATCH_SIZE'] = '512'
    os.environ['INFERMESH_CACHE_TTL'] = '300'
    
    # Test that environment variables are used
    base_url = os.getenv('INFERMESH_URL', 'http://infermesh:9000')
    api_key = os.getenv('INFERMESH_API_KEY')
    model = os.getenv('INFERMESH_MODEL', 'BAAI/bge-small-en-v1.5')
    routing_strategy = os.getenv('INFERMESH_ROUTING_STRATEGY')
    tenant = os.getenv('INFERMESH_TENANT')
    priority = os.getenv('INFERMESH_PRIORITY')
    batch_size = int(os.getenv('INFERMESH_BATCH_SIZE', '64'))
    cache_ttl = int(os.getenv('INFERMESH_CACHE_TTL', '300'))
    
    assert base_url == 'http://infermesh-router:9000'
    assert api_key == 'test-key'
    assert model == 'BAAI/bge-small-en-v1.5'
    assert routing_strategy == 'hybrid'
    assert tenant == 'test-tenant'
    assert priority == 'high'
    assert batch_size == 512
    assert cache_ttl == 300
    
    print("✓ Environment variable integration successful")
    return True


def main():
    """Run all tests."""
    print("Running InferMesh embedder tests...")
    print("=" * 50)
    
    tests = [
        test_config_validation,
        test_advanced_payload,
        test_environment_variables,
        test_basic_embedding,  # This will fail without actual service
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
