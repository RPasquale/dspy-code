#!/usr/bin/env python3
"""
Comprehensive RedDB Performance Test Suite

This script tests all aspects of the RedDB data model including:
- Data storage and retrieval performance
- Query optimization and caching
- Concurrent access patterns
- Memory usage and scalability
- Stream processing capabilities
- Analytics and aggregation performance
"""

import sys
import time
import threading
import random
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def generate_test_data():
    """Generate realistic test data for performance testing"""
    from dspy_agent.db import (
        SignatureMetrics, VerifierMetrics, TrainingMetrics,
        Environment, AgentState, ActionType,
        create_action_record, create_log_entry, create_embedding_vector
    )
    
    # Generate signature metrics
    signatures = []
    signature_names = [
        "CodeAnalyzerSig", "TestGeneratorSig", "BugFixerSig", "RefactorSig",
        "DocumenterSig", "OptimizerSig", "ValidatorSig", "SearcherSig",
        "PatcherSig", "ReviewerSig", "ContextBuilderSig", "PlannerSig"
    ]
    
    for i, name in enumerate(signature_names):
        for version in range(1, 4):  # 3 versions each
            signatures.append(SignatureMetrics(
                signature_name=f"{name}_v{version}",
                performance_score=random.uniform(70.0, 98.0),
                success_rate=random.uniform(85.0, 99.0),
                avg_response_time=random.uniform(0.5, 5.0),
                memory_usage=f"{random.randint(64, 512)}MB",
                iterations=random.randint(50, 500),
                last_updated=datetime.now().isoformat(),
                signature_type=random.choice(["analysis", "generation", "validation", "search"]),
                active=random.choice([True, True, True, False])  # 75% active
            ))
    
    # Generate verifier metrics
    verifiers = []
    verifier_names = [
        "SyntaxVerifier", "LogicVerifier", "SecurityVerifier", "PerformanceVerifier",
        "TestVerifier", "StyleVerifier", "CompatibilityVerifier", "IntegrationVerifier"
    ]
    
    for name in verifier_names:
        for env in ["dev", "test", "prod"]:
            verifiers.append(VerifierMetrics(
                verifier_name=f"{name}_{env}",
                accuracy=random.uniform(90.0, 99.9),
                status=random.choice(["active", "active", "inactive"]),  # 67% active
                checks_performed=random.randint(100, 2000),
                issues_found=random.randint(0, 50),
                last_run=datetime.now().isoformat(),
                avg_execution_time=random.uniform(0.1, 3.0)
            ))
    
    # Generate training metrics
    training_sessions = []
    for i in range(20):
        training_sessions.append(TrainingMetrics(
            session_id=f"train_session_{i:03d}",
            timestamp=time.time(),
            epoch=random.randint(1, 50),
            training_accuracy=random.uniform(0.7, 0.99),
            validation_accuracy=random.uniform(0.6, 0.95),
            loss=random.uniform(0.01, 2.0),
            learning_rate=random.uniform(0.0001, 0.01),
            batch_size=random.choice([8, 16, 32, 64]),
            model_type=random.choice(["gepa", "rl", "signature_optimizer"]),
            environment=random.choice(list(Environment)),
            hyperparameters={
                "optimizer": random.choice(["adam", "sgd", "rmsprop"]),
                "dropout": random.uniform(0.1, 0.5),
                "weight_decay": random.uniform(0.0001, 0.01)
            },
            convergence_metrics={
                "training_time": random.uniform(300, 7200),  # 5 minutes to 2 hours
                "memory_usage_gb": random.randint(1, 16),
                "gpu_usage": random.uniform(0.3, 0.95),
                "convergence_epoch": random.randint(10, 40)
            }
        ))
    
    # Generate action records
    actions = []
    for i in range(100):
        action_type = random.choice(list(ActionType))
        actions.append(create_action_record(
            action_type=action_type,
            state_before={"status": "idle", "context_size": random.randint(10, 100)},
            state_after={"status": "completed", "context_size": random.randint(10, 150)},
            parameters={"depth": random.choice(["shallow", "medium", "deep"])},
            result={"success": random.choice([True, False]), "items": random.randint(1, 20)},
            reward=random.uniform(0.0, 1.0),
            confidence=random.uniform(0.5, 1.0),
            execution_time=random.uniform(0.1, 10.0),
            environment=random.choice(list(Environment))
        ))
    
    # Generate log entries
    logs = []
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    sources = ["agent", "context", "training", "verifier", "signature", "db"]
    
    for i in range(200):
        level = random.choice(log_levels)
        source = random.choice(sources)
        logs.append(create_log_entry(
            level=level,
            source=source,
            message=f"Test log message {i} from {source}",
            context={"test_id": i, "batch": i // 10, "random_data": random.random()},
            environment=random.choice(list(Environment))
        ))
    
    # Generate embedding vectors
    embeddings = []
    for i in range(50):
        embeddings.append(create_embedding_vector(
            vector_id=f"content_{i:03d}",
            vector=[random.random() for _ in range(384)],  # 384-dim embedding
            source_type=random.choice(["code", "documentation", "log", "context"]),
            metadata={"source": f"file_{i}.py", "lines": random.randint(10, 500)},
            source_path=f"src/file_{i}.py",
            chunk_start=random.randint(1, 100),
            chunk_end=random.randint(101, 500)
        ))
    
    return {
        "signatures": signatures,
        "verifiers": verifiers,
        "training": training_sessions,
        "actions": actions,
        "logs": logs,
        "embeddings": embeddings
    }


def test_basic_operations(dm, test_data):
    """Test basic CRUD operations performance"""
    print("\n🔧 Testing Basic Operations Performance")
    print("-" * 50)
    
    results = {}
    
    # Test signature storage
    start_time = time.time()
    for signature in test_data["signatures"]:
        dm.store_signature_metrics(signature)
        dm.register_signature(signature.signature_name)
    signature_store_time = time.time() - start_time
    
    print(f"✓ Stored {len(test_data['signatures'])} signatures in {signature_store_time:.3f}s")
    print(f"  Average: {signature_store_time/len(test_data['signatures'])*1000:.2f}ms per signature")
    results["signature_store_time"] = signature_store_time
    
    # Test signature retrieval
    start_time = time.time()
    all_signatures = dm.get_all_signature_metrics()
    signature_retrieve_time = time.time() - start_time
    
    print(f"✓ Retrieved {len(all_signatures)} signatures in {signature_retrieve_time:.3f}s")
    results["signature_retrieve_time"] = signature_retrieve_time
    
    # Test verifier operations
    start_time = time.time()
    for verifier in test_data["verifiers"]:
        dm.store_verifier_metrics(verifier)
        dm.register_verifier(verifier.verifier_name)
    verifier_store_time = time.time() - start_time
    
    print(f"✓ Stored {len(test_data['verifiers'])} verifiers in {verifier_store_time:.3f}s")
    results["verifier_store_time"] = verifier_store_time
    
    # Test action recording
    start_time = time.time()
    for action in test_data["actions"]:
        dm.record_action(action)
    action_store_time = time.time() - start_time
    
    print(f"✓ Recorded {len(test_data['actions'])} actions in {action_store_time:.3f}s")
    results["action_store_time"] = action_store_time
    
    # Test logging
    start_time = time.time()
    for log in test_data["logs"]:
        dm.log(log)
    log_store_time = time.time() - start_time
    
    print(f"✓ Logged {len(test_data['logs'])} entries in {log_store_time:.3f}s")
    results["log_store_time"] = log_store_time
    
    # Test training metrics
    start_time = time.time()
    for training in test_data["training"]:
        dm.store_training_metrics(training)
    training_store_time = time.time() - start_time
    
    print(f"✓ Stored {len(test_data['training'])} training sessions in {training_store_time:.3f}s")
    results["training_store_time"] = training_store_time
    
    return results


def test_query_performance(dm):
    """Test query performance and optimization"""
    print("\n🔍 Testing Query Performance")
    print("-" * 50)
    
    results = {}
    
    # Test basic queries
    start_time = time.time()
    all_signatures = dm.get_all_signature_metrics()
    query_time1 = time.time() - start_time
    print(f"✓ All signatures query: {query_time1:.3f}s ({len(all_signatures)} results)")
    results["all_signatures_query"] = query_time1
    
    start_time = time.time()
    all_verifiers = dm.get_all_verifier_metrics()
    query_time2 = time.time() - start_time
    print(f"✓ All verifiers query: {query_time2:.3f}s ({len(all_verifiers)} results)")
    results["all_verifiers_query"] = query_time2
    
    start_time = time.time()
    recent_actions = dm.get_recent_actions(limit=50)
    query_time3 = time.time() - start_time
    print(f"✓ Recent actions query: {query_time3:.3f}s ({len(recent_actions)} results)")
    results["recent_actions_query"] = query_time3
    
    start_time = time.time()
    recent_logs = dm.get_recent_logs(limit=100)
    query_time4 = time.time() - start_time
    print(f"✓ Recent logs query: {query_time4:.3f}s ({len(recent_logs)} results)")
    results["recent_logs_query"] = query_time4
    
    start_time = time.time()
    training_history = dm.get_training_history(limit=20)
    query_time5 = time.time() - start_time
    print(f"✓ Training history query: {query_time5:.3f}s ({len(training_history)} results)")
    results["training_history_query"] = query_time5
    
    # Test enhanced queries if available
    if hasattr(dm, 'get_actions_by_reward_range'):
        start_time = time.time()
        range_actions = dm.get_actions_by_reward_range(0.7, 1.0, limit=50)
        query_time6 = time.time() - start_time
        print(f"✓ Reward range query: {query_time6:.3f}s ({len(range_actions)} results)")
        results["range_query"] = query_time6
    
    if hasattr(dm, 'get_performance_summary'):
        start_time = time.time()
        performance_summary = dm.get_performance_summary(hours=24)
        query_time7 = time.time() - start_time
        print(f"✓ Performance summary query: {query_time7:.3f}s")
        results["performance_summary_query"] = query_time7
    
    return results


def test_cache_performance(dm):
    """Test caching effectiveness"""
    print("\n⚡ Testing Cache Performance")
    print("-" * 50)
    
    results = {}
    
    # Get a signature name to test with
    all_sigs = dm.get_all_signature_metrics()
    if not all_sigs:
        print("❌ No signatures available for cache testing")
        return results
    
    test_sig_name = all_sigs[0].signature_name
    
    # Test cold cache (first access)
    start_time = time.time()
    dm.get_signature_metrics(test_sig_name)
    cold_time = time.time() - start_time
    
    # Test warm cache (repeated access)
    warm_times = []
    for _ in range(10):
        start_time = time.time()
        dm.get_signature_metrics(test_sig_name)
        warm_times.append(time.time() - start_time)
    
    avg_warm_time = sum(warm_times) / len(warm_times)
    speedup = cold_time / avg_warm_time if avg_warm_time > 0 else 0
    
    print(f"✓ Cold cache access: {cold_time:.4f}s")
    print(f"✓ Warm cache access: {avg_warm_time:.4f}s (avg of 10)")
    print(f"✓ Cache speedup: {speedup:.1f}x")
    
    results["cold_cache_time"] = cold_time
    results["warm_cache_time"] = avg_warm_time
    results["cache_speedup"] = speedup
    
    # Test cache statistics
    cache_stats = dm.get_cache_stats()
    print(f"✓ Main cache: {cache_stats['main_cache']['size']} items, {cache_stats['main_cache']['total_accesses']} total accesses")
    print(f"✓ Query cache: {cache_stats['query_cache']['size']} items, {cache_stats['query_cache']['total_accesses']} total accesses")
    
    results["cache_stats"] = cache_stats
    
    return results


def test_concurrent_access(dm):
    """Test concurrent access patterns"""
    print("\n🔄 Testing Concurrent Access")
    print("-" * 50)
    
    results = {}
    
    def worker_read_signatures(worker_id, iterations=10):
        """Worker function for concurrent reads"""
        times = []
        for i in range(iterations):
            start_time = time.time()
            dm.get_all_signature_metrics()
            times.append(time.time() - start_time)
        return worker_id, times
    
    def worker_write_actions(worker_id, iterations=5):
        """Worker function for concurrent writes"""
        from dspy_agent.db import create_action_record, ActionType, Environment
        times = []
        for i in range(iterations):
            action = create_action_record(
                action_type=ActionType.CODE_ANALYSIS,
                state_before={"worker": worker_id, "iteration": i},
                state_after={"worker": worker_id, "iteration": i, "completed": True},
                parameters={"concurrent_test": True},
                result={"success": True},
                reward=random.uniform(0.5, 1.0),
                confidence=random.uniform(0.7, 1.0),
                execution_time=random.uniform(0.1, 2.0),
                environment=Environment.DEVELOPMENT
            )
            
            start_time = time.time()
            dm.record_action(action)
            times.append(time.time() - start_time)
        return worker_id, times
    
    # Test concurrent reads
    num_readers = 5
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_readers) as executor:
        read_futures = [executor.submit(worker_read_signatures, i) for i in range(num_readers)]
        read_results = [future.result() for future in as_completed(read_futures)]
    
    total_read_time = time.time() - start_time
    
    all_read_times = []
    for worker_id, times in read_results:
        all_read_times.extend(times)
    
    avg_read_time = sum(all_read_times) / len(all_read_times)
    print(f"✓ Concurrent reads: {num_readers} workers, {len(all_read_times)} total operations")
    print(f"  Total time: {total_read_time:.3f}s, Average per operation: {avg_read_time:.4f}s")
    
    results["concurrent_reads"] = {
        "workers": num_readers,
        "total_time": total_read_time,
        "avg_operation_time": avg_read_time
    }
    
    # Test concurrent writes
    num_writers = 3
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_writers) as executor:
        write_futures = [executor.submit(worker_write_actions, i) for i in range(num_writers)]
        write_results = [future.result() for future in as_completed(write_futures)]
    
    total_write_time = time.time() - start_time
    
    all_write_times = []
    for worker_id, times in write_results:
        all_write_times.extend(times)
    
    avg_write_time = sum(all_write_times) / len(all_write_times)
    print(f"✓ Concurrent writes: {num_writers} workers, {len(all_write_times)} total operations")
    print(f"  Total time: {total_write_time:.3f}s, Average per operation: {avg_write_time:.4f}s")
    
    results["concurrent_writes"] = {
        "workers": num_writers,
        "total_time": total_write_time,
        "avg_operation_time": avg_write_time
    }
    
    return results


def test_analytics_performance(dm):
    """Test analytics and aggregation performance"""
    print("\n📊 Testing Analytics Performance")
    print("-" * 50)
    
    results = {}
    
    # Test learning progress analytics
    start_time = time.time()
    progress = dm.get_learning_progress(sessions=10)
    progress_time = time.time() - start_time
    print(f"✓ Learning progress analysis: {progress_time:.3f}s")
    print(f"  Sessions analyzed: {progress.get('sessions_analyzed', 0)}")
    results["learning_progress_time"] = progress_time
    
    # Test performance summary
    start_time = time.time()
    summary = dm.get_performance_summary(hours=24)
    summary_time = time.time() - start_time
    print(f"✓ Performance summary: {summary_time:.3f}s")
    
    if summary:
        print(f"  Metrics computed: {len(summary)} categories")
        if 'signature_performance' in summary:
            sig_perf = summary['signature_performance']
            print(f"  Signatures analyzed: {sig_perf.get('total_signatures', 0)}")
        if 'action_performance' in summary:
            act_perf = summary['action_performance']
            print(f"  Actions analyzed: {act_perf.get('total_actions', 0)}")
    
    results["performance_summary_time"] = summary_time
    
    # Test signature trend analysis
    all_sigs = dm.get_all_signature_metrics()
    if all_sigs:
        test_sig = all_sigs[0].signature_name
        start_time = time.time()
        trend = dm.get_signature_performance_trend(test_sig, hours=24*7)
        trend_time = time.time() - start_time
        print(f"✓ Signature trend analysis: {trend_time:.3f}s")
        print(f"  Data points: {len(trend) if isinstance(trend, list) else 'N/A'}")
        results["trend_analysis_time"] = trend_time
    
    return results


def test_memory_usage():
    """Test memory usage patterns"""
    print("\n💾 Testing Memory Usage")
    print("-" * 50)
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"✓ Initial memory usage: {initial_memory:.1f} MB")
        
        # Initialize fresh data manager
        from dspy_agent.db import get_enhanced_data_manager, initialize_database
        initialize_database()
        dm = get_enhanced_data_manager()
        
        # Generate and store test data
        test_data = generate_test_data()
        
        # Store all data
        for signature in test_data["signatures"]:
            dm.store_signature_metrics(signature)
        for verifier in test_data["verifiers"]:
            dm.store_verifier_metrics(verifier)
        for action in test_data["actions"]:
            dm.record_action(action)
        for log in test_data["logs"]:
            dm.log(log)
        
        # Get memory after data load
        loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"✓ Memory after loading test data: {loaded_memory:.1f} MB")
        print(f"✓ Memory increase: {loaded_memory - initial_memory:.1f} MB")
        
        # Test memory with heavy cache usage
        for _ in range(100):
            dm.get_all_signature_metrics()
            dm.get_performance_summary(hours=1)
        
        cached_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"✓ Memory after cache usage: {cached_memory:.1f} MB")
        print(f"✓ Cache overhead: {cached_memory - loaded_memory:.1f} MB")
        
        return {
            "initial_memory": initial_memory,
            "loaded_memory": loaded_memory,
            "cached_memory": cached_memory,
            "data_overhead": loaded_memory - initial_memory,
            "cache_overhead": cached_memory - loaded_memory
        }
        
    except ImportError:
        print("⚠️  psutil not available, skipping memory tests")
        return {}


def generate_performance_report(all_results):
    """Generate a comprehensive performance report"""
    print("\n📋 Performance Report Summary")
    print("=" * 60)
    
    # Basic operations summary
    if "basic_ops" in all_results:
        basic = all_results["basic_ops"]
        print(f"\n🔧 Basic Operations:")
        print(f"  Signature storage: {basic.get('signature_store_time', 0):.3f}s")
        print(f"  Signature retrieval: {basic.get('signature_retrieve_time', 0):.3f}s")
        print(f"  Action recording: {basic.get('action_store_time', 0):.3f}s")
        print(f"  Log storage: {basic.get('log_store_time', 0):.3f}s")
    
    # Query performance
    if "queries" in all_results:
        queries = all_results["queries"]
        print(f"\n🔍 Query Performance:")
        for query_name, query_time in queries.items():
            print(f"  {query_name}: {query_time:.3f}s")
    
    # Cache performance
    if "cache" in all_results:
        cache = all_results["cache"]
        print(f"\n⚡ Cache Performance:")
        print(f"  Speedup: {cache.get('cache_speedup', 0):.1f}x")
        print(f"  Cold access: {cache.get('cold_cache_time', 0):.4f}s")
        print(f"  Warm access: {cache.get('warm_cache_time', 0):.4f}s")
    
    # Concurrent access
    if "concurrent" in all_results:
        concurrent = all_results["concurrent"]
        if "concurrent_reads" in concurrent:
            reads = concurrent["concurrent_reads"]
            print(f"\n🔄 Concurrent Access:")
            print(f"  Read workers: {reads['workers']}, avg time: {reads['avg_operation_time']:.4f}s")
        if "concurrent_writes" in concurrent:
            writes = concurrent["concurrent_writes"]
            print(f"  Write workers: {writes['workers']}, avg time: {writes['avg_operation_time']:.4f}s")
    
    # Memory usage
    if "memory" in all_results:
        memory = all_results["memory"]
        print(f"\n💾 Memory Usage:")
        print(f"  Data overhead: {memory.get('data_overhead', 0):.1f} MB")
        print(f"  Cache overhead: {memory.get('cache_overhead', 0):.1f} MB")
        print(f"  Total usage: {memory.get('cached_memory', 0):.1f} MB")
    
    # Performance grades
    print(f"\n🎯 Performance Grades:")
    
    # Grade basic operations
    if "basic_ops" in all_results:
        basic = all_results["basic_ops"]
        avg_store_time = (basic.get('signature_store_time', 0) + 
                         basic.get('action_store_time', 0) + 
                         basic.get('log_store_time', 0)) / 3
        if avg_store_time < 0.1:
            print("  Storage operations: A+ (Excellent)")
        elif avg_store_time < 0.5:
            print("  Storage operations: A (Very Good)")
        elif avg_store_time < 1.0:
            print("  Storage operations: B (Good)")
        else:
            print("  Storage operations: C (Needs Improvement)")
    
    # Grade cache performance
    if "cache" in all_results:
        speedup = all_results["cache"].get("cache_speedup", 1)
        if speedup > 10:
            print("  Cache efficiency: A+ (Excellent)")
        elif speedup > 5:
            print("  Cache efficiency: A (Very Good)")
        elif speedup > 2:
            print("  Cache efficiency: B (Good)")
        else:
            print("  Cache efficiency: C (Needs Improvement)")
    
    print(f"\n✅ Performance testing completed successfully!")


def main():
    """Main performance testing function"""
    print("🚀 DSPy Agent RedDB Performance Test Suite")
    print("=" * 60)
    
    try:
        from dspy_agent.db import get_enhanced_data_manager, initialize_database
        
        # Initialize database
        print("Initializing database...")
        if not initialize_database():
            print("❌ Database initialization failed")
            return 1
        
        # Get data manager
        dm = get_enhanced_data_manager()
        
        # Generate test data
        print("Generating test data...")
        test_data = generate_test_data()
        print(f"Generated: {len(test_data['signatures'])} signatures, "
              f"{len(test_data['verifiers'])} verifiers, "
              f"{len(test_data['actions'])} actions, "
              f"{len(test_data['logs'])} logs")
        
        # Run all performance tests
        all_results = {}
        
        # Basic operations
        all_results["basic_ops"] = test_basic_operations(dm, test_data)
        
        # Query performance
        all_results["queries"] = test_query_performance(dm)
        
        # Cache performance
        all_results["cache"] = test_cache_performance(dm)
        
        # Concurrent access
        all_results["concurrent"] = test_concurrent_access(dm)
        
        # Analytics performance
        all_results["analytics"] = test_analytics_performance(dm)
        
        # Memory usage
        all_results["memory"] = test_memory_usage()
        
        # Generate comprehensive report
        generate_performance_report(all_results)
        
        # Save results to file
        results_file = Path(__file__).parent / "reddb_performance_results.json"
        with open(results_file, 'w') as f:
            # Convert any datetime objects to strings for JSON serialization
            json_results = {}
            for key, value in all_results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: str(v) if hasattr(v, 'isoformat') else v 
                                       for k, v in value.items()}
                else:
                    json_results[key] = str(value) if hasattr(value, 'isoformat') else value
            
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
