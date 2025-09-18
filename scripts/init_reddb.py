#!/usr/bin/env python3
"""
Initialize and test the RedDB data model for DSPy Agent.

This script initializes the database schema, runs migrations, and performs
basic tests to ensure the data model is working correctly.
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dspy_agent.db import (
    initialize_database, get_enhanced_data_manager, get_migration_manager,
    SignatureMetrics, VerifierMetrics, TrainingMetrics, ActionRecord,
    LogEntry, ContextState, PatchRecord, EmbeddingVector,
    Environment, ActionType, AgentState,
    create_log_entry, create_action_record, create_embedding_vector
)


def test_basic_operations():
    """Test basic CRUD operations"""
    print("Testing basic operations...")
    
    dm = get_enhanced_data_manager()
    
    # Test signature metrics
    sig_metrics = SignatureMetrics(
        signature_name="TestSignature",
        performance_score=85.5,
        success_rate=92.3,
        avg_response_time=2.1,
        memory_usage="256MB",
        iterations=100,
        last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
        signature_type="test"
    )
    
    dm.store_signature_metrics(sig_metrics)
    retrieved = dm.get_signature_metrics("TestSignature")
    
    assert retrieved is not None, "Failed to retrieve signature metrics"
    assert retrieved.performance_score == 85.5, "Performance score mismatch"
    print("✓ Signature metrics test passed")
    
    # Test verifier metrics
    ver_metrics = VerifierMetrics(
        verifier_name="TestVerifier",
        accuracy=94.2,
        status="active",
        checks_performed=500,
        issues_found=12,
        last_run=time.strftime("%Y-%m-%d %H:%M:%S"),
        avg_execution_time=1.5
    )
    
    dm.store_verifier_metrics(ver_metrics)
    retrieved_ver = dm.get_verifier_metrics("TestVerifier")
    
    assert retrieved_ver is not None, "Failed to retrieve verifier metrics"
    assert retrieved_ver.accuracy == 94.2, "Accuracy mismatch"
    print("✓ Verifier metrics test passed")
    
    # Test action recording
    action = create_action_record(
        action_type=ActionType.CODE_ANALYSIS,
        state_before={"files": ["test.py"], "status": "idle"},
        state_after={"files": ["test.py"], "status": "analyzed"},
        parameters={"target_file": "test.py"},
        result={"analysis": "completed", "issues": 2},
        reward=0.8,
        confidence=0.9,
        execution_time=3.2,
        environment=Environment.DEVELOPMENT
    )
    
    dm.record_action(action)
    recent_actions = dm.get_recent_actions(limit=10)
    
    assert len(recent_actions) > 0, "No actions recorded"
    print("✓ Action recording test passed")
    
    # Test logging
    log_entry = create_log_entry(
        level="INFO",
        source="test_script",
        message="Test log message",
        context={"test": True, "timestamp": time.time()},
        environment=Environment.DEVELOPMENT
    )
    
    dm.log(log_entry)
    recent_logs = dm.get_recent_logs(limit=10)
    
    assert len(recent_logs) > 0, "No logs recorded"
    print("✓ Logging test passed")
    
    print("All basic operations tests passed!")


def test_enhanced_features():
    """Test enhanced features like caching and queries"""
    print("Testing enhanced features...")
    
    dm = get_enhanced_data_manager()
    
    # Test caching
    sig_name = "CachedTestSignature"
    sig_metrics = SignatureMetrics(
        signature_name=sig_name,
        performance_score=88.7,
        success_rate=95.1,
        avg_response_time=1.8,
        memory_usage="128MB",
        iterations=200,
        last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
        signature_type="cached_test"
    )
    
    dm.store_signature_metrics(sig_metrics)
    
    # First retrieval (from storage)
    start_time = time.time()
    retrieved1 = dm.get_signature_metrics(sig_name)
    time1 = time.time() - start_time
    
    # Second retrieval (from cache)
    start_time = time.time()
    retrieved2 = dm.get_signature_metrics(sig_name)
    time2 = time.time() - start_time
    
    assert retrieved1 is not None and retrieved2 is not None
    assert retrieved1.signature_name == retrieved2.signature_name
    print(f"✓ Caching test passed (cache speedup: {time1/time2:.2f}x)")
    
    # Test query building
    actions = []
    for i in range(10):
        action = create_action_record(
            action_type=ActionType.CODE_EDIT,
            state_before={"file": f"test{i}.py"},
            state_after={"file": f"test{i}.py", "modified": True},
            parameters={"edit_type": "refactor"},
            result={"success": i % 2 == 0},
            reward=0.5 + (i * 0.05),
            confidence=0.7 + (i * 0.02),
            execution_time=1.0 + (i * 0.1),
            environment=Environment.DEVELOPMENT
        )
        actions.append(action)
        dm.record_action(action)
    
    # Query high reward actions
    high_reward_actions = dm.get_actions_by_reward_range(0.7, 1.0, limit=5)
    assert len(high_reward_actions) > 0, "No high reward actions found"
    print("✓ Query building test passed")
    
    # Test performance summary
    summary = dm.get_performance_summary(hours=1)
    assert "signature_performance" in summary, "Missing signature performance in summary"
    assert "action_performance" in summary, "Missing action performance in summary"
    print("✓ Performance summary test passed")
    
    # Test cache stats
    cache_stats = dm.get_cache_stats()
    assert "main_cache" in cache_stats, "Missing main cache stats"
    assert "query_cache" in cache_stats, "Missing query cache stats"
    print("✓ Cache statistics test passed")
    
    print("All enhanced features tests passed!")


def test_migrations():
    """Test migration system"""
    print("Testing migration system...")
    
    manager = get_migration_manager()
    
    # Check current version
    current_version = manager.get_current_version()
    print(f"Current database version: {current_version}")
    
    # Check applied migrations
    applied = manager.get_applied_migrations()
    print(f"Applied migrations: {applied}")
    
    # Check pending migrations
    pending = manager.get_pending_migrations()
    print(f"Pending migrations: {len(pending)}")
    
    print("✓ Migration system test passed")


def main():
    """Main test function"""
    print("🚀 Initializing DSPy Agent RedDB Data Model")
    print("=" * 50)
    
    # Set environment variables for testing
    os.environ.setdefault("REDDB_NAMESPACE", "dspy_agent_test")
    
    try:
        # Initialize database
        print("Initializing database...")
        success = initialize_database()
        if not success:
            print("❌ Database initialization failed")
            return 1
        print("✅ Database initialized successfully")
        
        # Run tests
        test_migrations()
        test_basic_operations()
        test_enhanced_features()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! RedDB data model is ready.")
        
        # Display summary
        dm = get_enhanced_data_manager()
        cache_stats = dm.get_cache_stats()
        print(f"\nCache Statistics:")
        print(f"  Main Cache: {cache_stats['main_cache']['size']} items")
        print(f"  Query Cache: {cache_stats['query_cache']['size']} items")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
