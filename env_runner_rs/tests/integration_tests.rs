//! Integration tests for the Rust Environment Runner
//! Tests the complete functionality of the environment runner system

use env_runner_rs::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

/// Test environment setup and teardown
#[tokio::test]
async fn test_environment_lifecycle() {
    let env = Environment::new()
        .with_memory_limit(1024 * 1024 * 1024) // 1GB
        .with_cpu_limit(4)
        .with_timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create environment");

    // Test environment is properly initialized
    assert!(env.is_initialized());
    assert_eq!(env.memory_limit(), 1024 * 1024 * 1024);
    assert_eq!(env.cpu_limit(), 4);
    assert_eq!(env.timeout(), Duration::from_secs(30));
}

/// Test workload classification system
#[tokio::test]
async fn test_workload_classification() {
    let classifier = WorkloadClassifier::new()
        .with_cpu_intensive_threshold(0.8)
        .with_memory_intensive_threshold(0.7)
        .with_io_intensive_threshold(0.6)
        .build();

    // Test CPU intensive task classification
    let cpu_task = Task {
        cpu_usage: 0.9,
        memory_usage: 0.3,
        io_usage: 0.2,
        duration: Duration::from_secs(10),
    };

    let classification = classifier.classify(&cpu_task);
    assert_eq!(classification, WorkloadClass::CpuIntensive);

    // Test memory intensive task classification
    let memory_task = Task {
        cpu_usage: 0.4,
        memory_usage: 0.8,
        io_usage: 0.3,
        duration: Duration::from_secs(5),
    };

    let classification = classifier.classify(&memory_task);
    assert_eq!(classification, WorkloadClass::MemoryIntensive);

    // Test I/O intensive task classification
    let io_task = Task {
        cpu_usage: 0.3,
        memory_usage: 0.4,
        io_usage: 0.9,
        duration: Duration::from_secs(15),
    };

    let classification = classifier.classify(&io_task);
    assert_eq!(classification, WorkloadClass::IoIntensive);
}

/// Test prefetch queue functionality
#[tokio::test]
async fn test_prefetch_queue() {
    let mut queue = PrefetchQueue::new()
        .with_capacity(1000)
        .with_prefetch_strategy(PrefetchStrategy::Predictive)
        .build();

    // Test queue operations
    let task1 = Task::new("task1", WorkloadClass::CpuIntensive);
    let task2 = Task::new("task2", WorkloadClass::MemoryIntensive);
    let task3 = Task::new("task3", WorkloadClass::IoIntensive);

    queue.enqueue(task1).await;
    queue.enqueue(task2).await;
    queue.enqueue(task3).await;

    assert_eq!(queue.size(), 3);
    assert!(!queue.is_empty());

    // Test prefetching
    let prefetched = queue.prefetch_resources(2).await;
    assert_eq!(prefetched.len(), 2);

    // Test dequeue
    let dequeued = queue.dequeue().await;
    assert!(dequeued.is_some());
    assert_eq!(queue.size(), 2);
}

/// Test high-throughput I/O operations
#[tokio::test]
async fn test_high_throughput_io() {
    let io_manager = IOManager::new()
        .with_buffer_size(64 * 1024) // 64KB
        .with_batch_size(1000)
        .with_connection_pool_size(100)
        .build();

    // Test batch I/O operations
    let data = vec![b"test data".to_vec(); 1000];
    let results = io_manager.batch_write(&data).await;

    assert_eq!(results.len(), 1000);
    assert!(results.iter().all(|r| r.is_ok()));

    // Test concurrent I/O
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let io_manager = io_manager.clone();
            tokio::spawn(async move {
                let data = vec![b"concurrent data".to_vec(); 100];
                io_manager.batch_write(&data).await
            })
        })
        .collect();

    let results: Vec<_> = futures::future::join_all(handles).await;
    assert_eq!(results.len(), 10);
    assert!(results.iter().all(|r| r.is_ok()));
}

/// Test resource monitoring and limits
#[tokio::test]
async fn test_resource_monitoring() {
    let env = Environment::new()
        .with_memory_limit(512 * 1024 * 1024) // 512MB
        .with_cpu_limit(2)
        .with_timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create environment");

    let monitor = ResourceMonitor::new(env.clone());

    // Test resource usage tracking
    let initial_usage = monitor.get_current_usage().await;
    assert!(initial_usage.memory_usage >= 0);
    assert!(initial_usage.cpu_usage >= 0);

    // Test resource limit enforcement
    let result = env
        .execute_task(
            || {
                // Simulate memory-intensive task
                let mut data = Vec::with_capacity(1024 * 1024 * 1024); // 1GB
                for i in 0..1024 * 1024 * 1024 {
                    data.push(i as u8);
                }
                Ok(())
            },
            WorkloadClass::MemoryIntensive,
        )
        .await;

    // Should fail due to memory limit
    assert!(result.is_err());
}

/// Test concurrent task execution
#[tokio::test]
async fn test_concurrent_execution() {
    let env = Environment::new()
        .with_memory_limit(1024 * 1024 * 1024) // 1GB
        .with_cpu_limit(8)
        .with_timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create environment");

    // Test concurrent task execution
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let env = env.clone();
            tokio::spawn(async move {
                env.execute_task(
                    || {
                        // Simulate CPU-intensive task
                        let mut sum = 0;
                        for j in 0..1000000 {
                            sum += j;
                        }
                        Ok(sum)
                    },
                    WorkloadClass::CpuIntensive,
                )
                .await
            })
        })
        .collect();

    let results: Vec<_> = futures::future::join_all(handles).await;
    assert_eq!(results.len(), 10);
    assert!(results.iter().all(|r| r.is_ok()));
}

/// Test error handling and recovery
#[tokio::test]
async fn test_error_handling() {
    let env = Environment::new()
        .with_memory_limit(1024 * 1024 * 1024) // 1GB
        .with_cpu_limit(4)
        .with_timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create environment");

    // Test task that panics
    let result = env
        .execute_task(
            || {
                panic!("Test panic");
            },
            WorkloadClass::CpuIntensive,
        )
        .await;

    assert!(result.is_err());

    // Test task that returns error
    let result = env
        .execute_task(
            || Err("Test error".to_string()),
            WorkloadClass::CpuIntensive,
        )
        .await;

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Test error");
}

/// Test performance benchmarks
#[tokio::test]
async fn test_performance_benchmarks() {
    let env = Environment::new()
        .with_memory_limit(1024 * 1024 * 1024) // 1GB
        .with_cpu_limit(8)
        .with_timeout(Duration::from_secs(60))
        .build()
        .expect("Failed to create environment");

    // Benchmark task execution time
    let start = std::time::Instant::now();

    let handles: Vec<_> = (0..100)
        .map(|i| {
            let env = env.clone();
            tokio::spawn(async move {
                env.execute_task(
                    || {
                        // Simulate work
                        let mut sum = 0;
                        for j in 0..100000 {
                            sum += j;
                        }
                        Ok(sum)
                    },
                    WorkloadClass::CpuIntensive,
                )
                .await
            })
        })
        .collect();

    let results: Vec<_> = futures::future::join_all(handles).await;
    let duration = start.elapsed();

    assert_eq!(results.len(), 100);
    assert!(results.iter().all(|r| r.is_ok()));

    // Performance should be reasonable (less than 10 seconds for 100 tasks)
    assert!(duration.as_secs() < 10);
    println!("Executed 100 tasks in {:?}", duration);
}

/// Test sandbox security
#[tokio::test]
async fn test_sandbox_security() {
    let env = Environment::new()
        .with_memory_limit(1024 * 1024 * 1024) // 1GB
        .with_cpu_limit(4)
        .with_timeout(Duration::from_secs(10))
        .with_sandbox_enabled(true)
        .build()
        .expect("Failed to create environment");

    // Test that sandbox prevents dangerous operations
    let result = env
        .execute_task(
            || {
                // Try to access system resources
                std::fs::read("/etc/passwd")
            },
            WorkloadClass::IoIntensive,
        )
        .await;

    // Should fail due to sandbox restrictions
    assert!(result.is_err());
}

/// Test resource cleanup
#[tokio::test]
async fn test_resource_cleanup() {
    let env = Environment::new()
        .with_memory_limit(1024 * 1024 * 1024) // 1GB
        .with_cpu_limit(4)
        .with_timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create environment");

    let initial_memory = env.get_memory_usage().await;

    // Execute memory-intensive task
    let result = env
        .execute_task(
            || {
                let mut data = Vec::with_capacity(100 * 1024 * 1024); // 100MB
                for i in 0..100 * 1024 * 1024 {
                    data.push(i as u8);
                }
                // Data should be automatically cleaned up
                Ok(())
            },
            WorkloadClass::MemoryIntensive,
        )
        .await;

    assert!(result.is_ok());

    // Wait for cleanup
    tokio::time::sleep(Duration::from_millis(100)).await;

    let final_memory = env.get_memory_usage().await;

    // Memory should be cleaned up (allowing for some variance)
    assert!(final_memory <= initial_memory + 10 * 1024 * 1024); // 10MB tolerance
}

/// Test workload-specific optimizations
#[tokio::test]
async fn test_workload_optimizations() {
    let env = Environment::new()
        .with_memory_limit(1024 * 1024 * 1024) // 1GB
        .with_cpu_limit(8)
        .with_timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create environment");

    // Test CPU-intensive task optimization
    let cpu_start = std::time::Instant::now();
    let cpu_result = env
        .execute_task(
            || {
                // CPU-intensive computation
                let mut sum = 0;
                for i in 0..10000000 {
                    sum += i;
                }
                Ok(sum)
            },
            WorkloadClass::CpuIntensive,
        )
        .await;
    let cpu_duration = cpu_start.elapsed();

    assert!(cpu_result.is_ok());
    println!("CPU-intensive task completed in {:?}", cpu_duration);

    // Test I/O-intensive task optimization
    let io_start = std::time::Instant::now();
    let io_result = env
        .execute_task(
            || {
                // I/O-intensive operation
                let mut data = Vec::new();
                for i in 0..1000000 {
                    data.push(format!("data_{}", i));
                }
                Ok(data.len())
            },
            WorkloadClass::IoIntensive,
        )
        .await;
    let io_duration = io_start.elapsed();

    assert!(io_result.is_ok());
    println!("I/O-intensive task completed in {:?}", io_duration);
}

/// Test monitoring and metrics
#[tokio::test]
async fn test_monitoring_metrics() {
    let env = Environment::new()
        .with_memory_limit(1024 * 1024 * 1024) // 1GB
        .with_cpu_limit(4)
        .with_timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create environment");

    let monitor = MetricsCollector::new(env.clone());

    // Execute some tasks
    for i in 0..10 {
        let _ = env
            .execute_task(
                || {
                    // Simulate work
                    let mut sum = 0;
                    for j in 0..100000 {
                        sum += j;
                    }
                    Ok(sum)
                },
                WorkloadClass::CpuIntensive,
            )
            .await;
    }

    // Check metrics
    let metrics = monitor.get_metrics().await;
    assert!(metrics.total_tasks >= 10);
    assert!(metrics.successful_tasks >= 10);
    assert!(metrics.failed_tasks == 0);
    assert!(metrics.average_execution_time > Duration::from_millis(0));
}
