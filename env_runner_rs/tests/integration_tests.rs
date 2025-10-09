use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use env_runner_rs::hardware::{recommended_inflight, AcceleratorInfo, HardwareSnapshot};
use env_runner_rs::metrics::EnvRunnerMetrics;
use env_runner_rs::streaming::{collect_assignment_headers, RunnerTaskAssignment};
use env_runner_rs::{PrefetchQueue, Runner, WorkItem, WorkloadClass};
use prost::bytes::Bytes;
use serde_json::json;

#[test]
fn prefetch_queue_respects_capacity_and_preference() {
    let queue = PrefetchQueue::new(1);

    queue.push(WorkItem {
        id: "cpu-1".into(),
        class: WorkloadClass::CpuShort,
        payload: "{}".into(),
    });

    queue.push(WorkItem {
        id: "gpu-1".into(),
        class: WorkloadClass::Gpu,
        payload: "{}".into(),
    });

    // Capacity for the CpuShort partition is 1, so this push is ignored.
    queue.push(WorkItem {
        id: "cpu-2".into(),
        class: WorkloadClass::CpuShort,
        payload: "{}".into(),
    });

    assert_eq!(queue.len(), 2);

    let first = queue
        .pop(&[
            WorkloadClass::Gpu,
            WorkloadClass::CpuShort,
            WorkloadClass::CpuLong,
        ])
        .expect("expected gpu work item");
    assert_eq!(first.id, "gpu-1");

    let second = queue
        .pop(&[
            WorkloadClass::CpuShort,
            WorkloadClass::CpuLong,
            WorkloadClass::Gpu,
        ])
        .expect("expected cpu work item");
    assert_eq!(second.id, "cpu-1");

    assert_eq!(queue.len(), 0);
}

#[test]
fn runner_processes_items_and_updates_metrics() {
    let queue = Arc::new(PrefetchQueue::new(16));
    let processed = Arc::new(Mutex::new(Vec::new()));
    let metrics = Arc::new(EnvRunnerMetrics::new());

    let runner = Runner {
        queue: queue.clone(),
        processed: processed.clone(),
        metrics: metrics.clone(),
    };

    let shutdown = Arc::new(AtomicBool::new(false));
    let handle = runner.start(shutdown.clone());

    let classes = [
        WorkloadClass::Gpu,
        WorkloadClass::CpuShort,
        WorkloadClass::CpuLong,
    ];

    for (idx, class) in classes.iter().enumerate() {
        queue.push(WorkItem {
            id: format!("task-{idx}"),
            class: class.clone(),
            payload: "{}".into(),
        });
    }

    let deadline = Instant::now() + Duration::from_secs(2);
    while metrics.get_stats().tasks_processed < 3 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }

    shutdown.store(true, Ordering::SeqCst);
    queue.close();
    handle.join().expect("runner thread panicked");

    let processed_ids = processed.lock().unwrap().clone();
    assert_eq!(processed_ids.len(), 3);
    assert!(processed_ids.contains(&"task-0".to_string()));
    assert!(processed_ids.contains(&"task-1".to_string()));
    assert!(processed_ids.contains(&"task-2".to_string()));

    let stats = metrics.get_stats();
    assert_eq!(stats.tasks_processed, 3);
    assert_eq!(stats.queue_depth, 0);
}

#[test]
fn recommended_inflight_prefers_accelerators() {
    let mut snapshot = HardwareSnapshot::default();
    snapshot.cpu.logical_cores = 8;
    snapshot.accelerators = vec![AcceleratorInfo {
        vendor: "nvidia".into(),
        model: "L4".into(),
        memory_mb: 24576,
        count: 2,
        compute_capability: Some("8.9".into()),
    }];
    assert_eq!(recommended_inflight(&snapshot), 8);

    snapshot.accelerators.clear();
    snapshot.cpu.logical_cores = 6;
    assert_eq!(recommended_inflight(&snapshot), 3);
}

#[test]
fn kafka_headers_include_workflow_metadata() {
    let assignment = RunnerTaskAssignment {
        task_id: "task-42".into(),
        tenant: "demo".into(),
        topic: "workflow.tasks".into(),
        payload: Bytes::from(
            serde_json::to_vec(&json!({
                "workflow_id": "wf-abc",
                "workflow_run_id": "run-xyz",
                "workflow_context": {"tenant": "alpha"}
            }))
            .expect("payload"),
        ),
        offset: 0,
        partition: "p0".into(),
    };
    let headers = collect_assignment_headers(&assignment, None);
    let mut map = std::collections::HashMap::new();
    for (key, value) in headers {
        map.insert(key, String::from_utf8(value).unwrap());
    }
    assert_eq!(map.get("workflow_id"), Some(&"wf-abc".to_string()));
    assert_eq!(map.get("workflow_run_id"), Some(&"run-xyz".to_string()));
    assert_eq!(map.get("workflow_tenant"), Some(&"alpha".to_string()));
}

#[test]
fn workflow_meta_uses_topic_fallback() {
    let assignment = RunnerTaskAssignment {
        task_id: "task-100".into(),
        tenant: String::new(),
        topic: "workflow::embed".into(),
        payload: Bytes::from_static(b"{}"),
        offset: 10,
        partition: "p1".into(),
    };
    let meta = extract_workflow_meta(&assignment);
    assert_eq!(meta.workflow_id, None);
    assert_eq!(meta.workflow_run_id, None);
    assert_eq!(meta.tenant, Some("workflow::embed".to_string()));

    let skill = derive_skill_from_assignment(&assignment).expect("skill fallback");
    assert_eq!(skill, "workflow::embed");
}

#[test]
fn headers_include_runner_metadata_when_available() {
    let mut assignment = RunnerTaskAssignment {
        task_id: "task-55".into(),
        tenant: "Studio".into(),
        topic: "embed.jobs".into(),
        payload: Bytes::from_static(b"{}"),
        offset: 0,
        partition: "p0".into(),
    };
    assignment.payload = Bytes::from(serde_json::to_vec(&json!({"workflow_id": "WF", "tenant": "Studio"})).unwrap());
    let mut snapshot = HardwareSnapshot::default();
    snapshot.accelerators = vec![AcceleratorInfo {
        vendor: "amd".into(),
        model: "MI300".into(),
        memory_mb: 131072,
        count: 1,
        compute_capability: None,
    }];
    let headers = collect_assignment_headers(&assignment, Some(&snapshot));
    let mut found = false;
    for (key, value) in headers {
        if key == "runner_hardware" {
            assert!(String::from_utf8(value).unwrap().contains("MI300"));
            found = true;
        }
    }
    assert!(found, "runner_hardware header should be present when accelerators exist");
}
