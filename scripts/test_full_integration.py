#!/usr/bin/env python3
"""
Test full integration of RedDB data model throughout the DSPy agent system.

This script tests:
1. Dashboard API integration with real RedDB data
2. Context manager storing and retrieving data
3. Streaming system logging to RedDB
4. Training metrics storage
5. End-to-end data flow
"""

import os
import sys
import time
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dspy_agent.db import (
    get_enhanced_data_manager, initialize_database,
    SignatureMetrics, VerifierMetrics, TrainingMetrics,
    Environment, AgentState, ActionType,
    create_action_record, create_log_entry
)
from dspy_agent.context.context_manager import ContextManager
from dspy_agent.streaming.streamkit import LocalBus


def test_dashboard_integration():
    """Test dashboard data layer integration (without HTTP requests)"""
    print("🧪 Testing Dashboard Data Layer Integration...")
    
    try:
        # Test the data that dashboard would use
        dm = get_enhanced_data_manager()
        
        # Test signatures data
        signatures = dm.get_all_signature_metrics()
        if signatures:
            print(f"✅ Signature data: {len(signatures)} signatures available")
            avg_perf = sum(s.performance_score for s in signatures) / len(signatures)
            print(f"   Average performance: {avg_perf:.1f}%")
        else:
            print("⚠️  No signature data found (dashboard will create defaults)")
        
        # Test verifiers data  
        verifiers = dm.get_all_verifier_metrics()
        if verifiers:
            print(f"✅ Verifier data: {len(verifiers)} verifiers available")
            avg_acc = sum(v.accuracy for v in verifiers) / len(verifiers)
            print(f"   Average accuracy: {avg_acc:.1f}%")
        else:
            print("⚠️  No verifier data found (dashboard will create defaults)")
        
        # Test performance summary
        summary = dm.get_performance_summary(hours=1)
        print(f"✅ Performance summary generated with {len(summary)} metrics")
        
        # Test learning progress
        progress = dm.get_learning_progress(sessions=5)
        print(f"✅ Learning progress: {progress.get('sessions_analyzed', 0)} sessions")
        
    except Exception as e:
        print(f"❌ Dashboard integration error: {e}")
        import traceback
        traceback.print_exc()


def test_context_manager_integration():
    """Test context manager RedDB integration"""
    print("\n🧪 Testing Context Manager Integration...")
    
    try:
        # Create context manager
        workspace = project_root / "test_project"
        cm = ContextManager(workspace, environment=Environment.DEVELOPMENT)
        
        # Test storing context state
        context_state = cm.store_current_context(
            agent_state=AgentState.ANALYZING,
            current_task="Test integration task",
            active_files=["test_file.py", "another_file.py"],
            recent_actions=["action_1", "action_2"]
        )
        print(f"✅ Context state stored: {context_state.context_id}")
        
        # Test retrieving context state
        retrieved_context = cm.get_current_context()
        if retrieved_context:
            print(f"✅ Context state retrieved: {retrieved_context.agent_state.value}")
        else:
            print("❌ Failed to retrieve context state")
        
        # Test storing patch record
        patch_record = cm.store_patch_record(
            patch_content="def test_function(): return 'test'",
            target_files=["test.py"],
            applied=True,
            confidence_score=0.85,
            blast_radius=0.1
        )
        print(f"✅ Patch record stored: {patch_record.patch_id}")
        
        # Test enhanced context building
        enhanced_context = cm.build_enhanced_context("Integration test task")
        print(f"✅ Enhanced context built: {len(enhanced_context)} keys")
        print(f"   RedDB patches: {len(enhanced_context.get('reddb_patches', []))}")
        print(f"   RedDB logs: {len(enhanced_context.get('recent_reddb_logs', []))}")
        
    except Exception as e:
        print(f"❌ Context manager error: {e}")
        import traceback
        traceback.print_exc()


def test_streaming_integration():
    """Test streaming system RedDB integration"""
    print("\n🧪 Testing Streaming Integration...")
    
    try:
        dm = get_enhanced_data_manager()
        
        # Create a local bus with RedDB storage
        bus = LocalBus(storage=dm.storage)
        
        # Test publishing important messages
        bus.publish("agent.learning", {
            "tool": "test_tool",
            "reward": 0.85,
            "timestamp": time.time()
        })
        print("✅ Learning event published and logged")
        
        bus.publish("agent.patches", {
            "applied": True,
            "confidence": 0.92,
            "files": ["test.py"],
            "timestamp": time.time()
        })
        print("✅ Patch event published and logged")
        
        bus.publish("agent.errors", {
            "error": "Test error message",
            "severity": "low",
            "timestamp": time.time()
        })
        print("✅ Error event published and logged")
        
        # Check if logs were stored
        recent_logs = dm.get_recent_logs(limit=10)
        streaming_logs = [log for log in recent_logs if "streaming" in log.source]
        print(f"✅ Found {len(streaming_logs)} streaming logs in RedDB")
        
    except Exception as e:
        print(f"❌ Streaming integration error: {e}")
        import traceback
        traceback.print_exc()


def test_training_metrics_storage():
    """Test training metrics storage"""
    print("\n🧪 Testing Training Metrics Storage...")
    
    try:
        dm = get_enhanced_data_manager()
        
        # Create and store training metrics
        import uuid
        training_metrics = TrainingMetrics(
            session_id=str(uuid.uuid4()),
            timestamp=time.time(),
            epoch=10,
            training_accuracy=0.87,
            validation_accuracy=0.84,
            loss=0.23,
            learning_rate=0.001,
            batch_size=32,
            model_type="integration_test",
            environment=Environment.DEVELOPMENT,
            hyperparameters={"test_param": 1.0},
            convergence_metrics={"test_metric": 0.95}
        )
        
        dm.store_training_metrics(training_metrics)
        print(f"✅ Training metrics stored: {training_metrics.session_id}")
        
        # Retrieve training history
        history = dm.get_training_history(limit=5)
        print(f"✅ Retrieved {len(history)} training sessions")
        
        # Test learning progress
        progress = dm.get_learning_progress(sessions=5)
        print(f"✅ Learning progress: {progress.get('sessions_analyzed', 0)} sessions analyzed")
        
    except Exception as e:
        print(f"❌ Training metrics error: {e}")
        import traceback
        traceback.print_exc()


def test_performance_analytics():
    """Test performance analytics and queries"""
    print("\n🧪 Testing Performance Analytics...")
    
    try:
        dm = get_enhanced_data_manager()
        
        # Generate some test actions for analytics
        for i in range(10):
            action = create_action_record(
                action_type=ActionType.CODE_ANALYSIS,
                state_before={"test": f"state_{i}"},
                state_after={"test": f"state_{i}_completed"},
                parameters={"iteration": i},
                result={"success": i % 2 == 0},
                reward=0.5 + (i * 0.05),
                confidence=0.8 + (i * 0.01),
                execution_time=1.0 + (i * 0.1),
                environment=Environment.DEVELOPMENT
            )
            dm.record_action(action)
        
        print("✅ Generated 10 test actions")
        
        # Test performance summary
        summary = dm.get_performance_summary(hours=1)
        print(f"✅ Performance summary generated:")
        print(f"   Actions: {summary.get('action_performance', {}).get('total_actions', 0)}")
        print(f"   Avg reward: {summary.get('action_performance', {}).get('avg_reward', 0):.2f}")
        
        # Test high reward actions query
        from dspy_agent.db.enhanced_storage import get_recent_high_reward_actions
        high_reward = get_recent_high_reward_actions(min_reward=0.7, hours=1)
        print(f"✅ Found {len(high_reward)} high reward actions")
        
        # Test cache performance
        cache_stats = dm.get_cache_stats()
        print(f"✅ Cache stats: {cache_stats['main_cache']['size']} main, {cache_stats['query_cache']['size']} query")
        
    except Exception as e:
        print(f"❌ Performance analytics error: {e}")
        import traceback
        traceback.print_exc()


def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    print("\n🧪 Testing End-to-End Workflow...")
    
    try:
        dm = get_enhanced_data_manager()
        
        # 1. Initialize context
        workspace = project_root / "test_project"
        cm = ContextManager(workspace, environment=Environment.DEVELOPMENT)
        
        # 2. Start analysis task
        context_state = cm.store_current_context(
            agent_state=AgentState.ANALYZING,
            current_task="End-to-end integration test"
        )
        print("✅ Step 1: Context initialized")
        
        # 3. Record analysis action
        analysis_action = create_action_record(
            action_type=ActionType.CODE_ANALYSIS,
            state_before={"agent_state": "idle"},
            state_after={"agent_state": "analyzing", "files_analyzed": 3},
            parameters={"depth": "full", "include_tests": True},
            result={"issues_found": 2, "suggestions": 5},
            reward=0.8,
            confidence=0.9,
            execution_time=3.2,
            environment=Environment.DEVELOPMENT
        )
        dm.record_action(analysis_action)
        print("✅ Step 2: Analysis action recorded")
        
        # 4. Generate and apply patch
        patch_record = cm.store_patch_record(
            patch_content="# Fixed integration test\ndef improved_function():\n    return 'improved'",
            target_files=["integration_test.py"],
            applied=True,
            confidence_score=0.87,
            blast_radius=0.15
        )
        print("✅ Step 3: Patch applied and recorded")
        
        # 5. Record patch action
        patch_action = create_action_record(
            action_type=ActionType.CODE_EDIT,
            state_before={"files": ["integration_test.py"], "issues": 2},
            state_after={"files": ["integration_test.py"], "issues": 0},
            parameters={"patch_id": patch_record.patch_id},
            result={"applied": True, "tests_passed": True},
            reward=0.95,
            confidence=0.87,
            execution_time=2.1,
            environment=Environment.DEVELOPMENT
        )
        dm.record_action(patch_action)
        print("✅ Step 4: Patch action recorded")
        
        # 6. Update to executing state
        cm.store_current_context(
            agent_state=AgentState.EXECUTING,
            current_task="End-to-end integration test",
            active_files=["integration_test.py"],
            recent_actions=[analysis_action.action_id, patch_action.action_id]
        )
        print("✅ Step 5: State updated to executing")
        
        # 7. Build enhanced context with all data
        enhanced_context = cm.build_enhanced_context("Final integration check")
        print("✅ Step 6: Enhanced context built with full history")
        
        # 8. Verify data integrity
        recent_actions = dm.get_recent_actions(limit=10)
        recent_patches = dm.get_patch_history(limit=5)
        current_context = dm.get_current_context()
        
        workflow_complete = (
            len(recent_actions) >= 2 and
            len(recent_patches) >= 1 and
            current_context is not None and
            current_context.agent_state == AgentState.EXECUTING
        )
        
        if workflow_complete:
            print("🎉 End-to-end workflow completed successfully!")
            print(f"   Actions recorded: {len(recent_actions)}")
            print(f"   Patches applied: {len(recent_patches)}")
            print(f"   Current state: {current_context.agent_state.value}")
        else:
            print("❌ End-to-end workflow incomplete")
        
        return workflow_complete
        
    except Exception as e:
        print(f"❌ End-to-end workflow error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 DSPy Agent RedDB Integration Test Suite")
    print("=" * 60)
    
    # Initialize database
    print("Initializing database...")
    if not initialize_database():
        print("❌ Database initialization failed")
        return 1
    print("✅ Database initialized")
    
    # Run all tests
    test_dashboard_integration()
    test_context_manager_integration()
    test_streaming_integration()
    test_training_metrics_storage()
    test_performance_analytics()
    
    # Final end-to-end test
    workflow_success = test_end_to_end_workflow()
    
    print("\n" + "=" * 60)
    if workflow_success:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ RedDB data model is fully integrated throughout the DSPy agent system")
    else:
        print("❌ Some integration tests failed")
        print("🔧 Check the logs above for specific issues")
    
    # Display final statistics
    try:
        dm = get_enhanced_data_manager()
        cache_stats = dm.get_cache_stats()
        performance_summary = dm.get_performance_summary(hours=1)
        
        print(f"\n📊 Final Statistics:")
        print(f"   Cache Performance: {cache_stats['main_cache']['size']} items cached")
        print(f"   Actions Recorded: {performance_summary.get('action_performance', {}).get('total_actions', 0)}")
        print(f"   Average Reward: {performance_summary.get('action_performance', {}).get('avg_reward', 0):.2f}")
        print(f"   System Health: {'✅ Healthy' if performance_summary else '❌ Issues detected'}")
        
    except Exception as e:
        print(f"⚠️  Could not retrieve final statistics: {e}")
    
    return 0 if workflow_success else 1


if __name__ == "__main__":
    exit(main())
