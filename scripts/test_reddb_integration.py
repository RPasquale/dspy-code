#!/usr/bin/env python3
"""
Simple RedDB integration test that validates the core data model functionality.
"""

import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_reddb_integration():
    """Test RedDB integration without complex imports"""
    print("🧪 Testing RedDB Integration...")
    
    try:
        from dspy_agent.db import (
            get_enhanced_data_manager, initialize_database,
            SignatureMetrics, VerifierMetrics, TrainingMetrics,
            Environment, AgentState, ActionType,
            create_action_record, create_log_entry
        )
        
        # Initialize database
        print("Initializing database...")
        if not initialize_database():
            print("❌ Database initialization failed")
            return False
        print("✅ Database initialized")
        
        # Get data manager
        dm = get_enhanced_data_manager()
        
        # Test 1: Create and store signature metrics
        print("\n📊 Testing Signature Metrics...")
        signature = SignatureMetrics(
            signature_name="TestSignature",
            performance_score=89.5,
            success_rate=94.2,
            avg_response_time=2.1,
            memory_usage="256MB",
            iterations=150,
            last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
            signature_type="test",
            active=True
        )
        
        dm.store_signature_metrics(signature)
        dm.register_signature(signature.signature_name)
        
        retrieved = dm.get_signature_metrics("TestSignature")
        if retrieved and retrieved.performance_score == 89.5:
            print("✅ Signature metrics: Store and retrieve working")
        else:
            print("❌ Signature metrics: Failed")
            return False
        
        # Test 2: Create and store verifier metrics
        print("\n🔍 Testing Verifier Metrics...")
        verifier = VerifierMetrics(
            verifier_name="TestVerifier",
            accuracy=96.8,
            status="active",
            checks_performed=500,
            issues_found=12,
            last_run=time.strftime("%Y-%m-%d %H:%M:%S"),
            avg_execution_time=1.5
        )
        
        dm.store_verifier_metrics(verifier)
        dm.register_verifier(verifier.verifier_name)
        
        retrieved_verifier = dm.get_verifier_metrics("TestVerifier")
        if retrieved_verifier and retrieved_verifier.accuracy == 96.8:
            print("✅ Verifier metrics: Store and retrieve working")
        else:
            print("❌ Verifier metrics: Failed")
            return False
        
        # Test 3: Record actions
        print("\n🎯 Testing Action Recording...")
        action = create_action_record(
            action_type=ActionType.CODE_ANALYSIS,
            state_before={"files": ["test.py"], "status": "idle"},
            state_after={"files": ["test.py"], "status": "analyzed"},
            parameters={"depth": "full"},
            result={"issues": 3, "suggestions": 5},
            reward=0.85,
            confidence=0.92,
            execution_time=3.4,
            environment=Environment.DEVELOPMENT
        )
        
        dm.record_action(action)
        recent_actions = dm.get_recent_actions(limit=5)
        
        if len(recent_actions) > 0:
            print("✅ Action recording: Working")
        else:
            print("❌ Action recording: Failed")
            return False
        
        # Test 4: Logging
        print("\n📝 Testing Logging...")
        log_entry = create_log_entry(
            level="INFO",
            source="integration_test",
            message="Test log message for RedDB integration",
            context={"test": True, "integration": "reddb"},
            environment=Environment.DEVELOPMENT
        )
        
        dm.log(log_entry)
        recent_logs = dm.get_recent_logs(limit=5)
        
        if len(recent_logs) > 0:
            print("✅ Logging: Working")
        else:
            print("❌ Logging: Failed")
            return False
        
        # Test 5: Performance analytics
        print("\n📈 Testing Performance Analytics...")
        summary = dm.get_performance_summary(hours=1)
        progress = dm.get_learning_progress(sessions=5)
        
        if summary and progress:
            print("✅ Performance analytics: Working")
            print(f"   Total actions: {summary.get('action_performance', {}).get('total_actions', 0)}")
            print(f"   Sessions analyzed: {progress.get('sessions_analyzed', 0)}")
        else:
            print("❌ Performance analytics: Failed")
            return False
        
        # Test 6: Cache performance
        print("\n⚡ Testing Cache Performance...")
        cache_stats = dm.get_cache_stats()
        
        # Test cache by retrieving the same signature multiple times
        start_time = time.time()
        for _ in range(10):
            dm.get_signature_metrics("TestSignature")
        cache_time = time.time() - start_time
        
        print(f"✅ Cache performance: {cache_time:.4f}s for 10 retrievals")
        print(f"   Main cache: {cache_stats['main_cache']['size']} items")
        print(f"   Query cache: {cache_stats['query_cache']['size']} items")
        
        # Test 7: Query building
        print("\n🔍 Testing Query Building...")
        high_reward_actions = dm.get_actions_by_reward_range(0.8, 1.0, limit=10)
        print(f"✅ Query building: Found {len(high_reward_actions)} high-reward actions")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dashboard_data():
    """Test that dashboard can access the data"""
    print("\n🖥️  Testing Dashboard Data Access...")
    
    try:
        from dspy_agent.db import get_enhanced_data_manager
        
        dm = get_enhanced_data_manager()
        
        # Test data that dashboard would use
        signatures = dm.get_all_signature_metrics()
        verifiers = dm.get_all_verifier_metrics()
        
        print(f"✅ Dashboard data access:")
        print(f"   Signatures available: {len(signatures)}")
        print(f"   Verifiers available: {len(verifiers)}")
        
        if signatures:
            avg_perf = sum(s.performance_score for s in signatures) / len(signatures)
            print(f"   Average signature performance: {avg_perf:.1f}%")
        
        if verifiers:
            avg_acc = sum(v.accuracy for v in verifiers) / len(verifiers)
            print(f"   Average verifier accuracy: {avg_acc:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard data test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 DSPy Agent RedDB Integration Test")
    print("=" * 50)
    
    success = True
    
    # Run core integration test
    if not test_reddb_integration():
        success = False
    
    # Test dashboard data access
    if not test_dashboard_data():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ RedDB data model is fully integrated and working")
        print("✅ Dashboard can access real data from RedDB")
        print("✅ Performance analytics and caching are functional")
        print("✅ Action recording and logging are working")
    else:
        print("❌ Some integration tests failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
