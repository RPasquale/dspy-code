use chrono;
use serde_json::json;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub mod infermesh;
pub mod metrics;
pub mod notify_watcher;

use crate::metrics::EnvRunnerMetrics;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkloadClass {
    CpuShort,
    CpuLong,
    Gpu,
}

#[derive(Clone, Debug)]
pub struct WorkItem {
    pub id: String,
    pub class: WorkloadClass,
    pub payload: String,
}

#[derive(Default)]
struct InnerQ {
    q_cpu_short: VecDeque<WorkItem>,
    q_cpu_long: VecDeque<WorkItem>,
    q_gpu: VecDeque<WorkItem>,
    closed: bool,
}

/// PrefetchQueue partitions items by workload class and supports bounded prefetching.
pub struct PrefetchQueue {
    inner: Mutex<InnerQ>,
    cv: Condvar,
    cap_per_class: usize,
}

impl PrefetchQueue {
    pub fn new(cap_per_class: usize) -> Self {
        Self {
            inner: Mutex::new(InnerQ::default()),
            cv: Condvar::new(),
            cap_per_class,
        }
    }

    pub fn push(&self, item: WorkItem) {
        let mut inner = self.inner.lock().unwrap();
        if inner.closed {
            return;
        }
        let q = match item.class {
            WorkloadClass::CpuShort => &mut inner.q_cpu_short,
            WorkloadClass::CpuLong => &mut inner.q_cpu_long,
            WorkloadClass::Gpu => &mut inner.q_gpu,
        };
        if q.len() < self.cap_per_class {
            q.push_back(item);
            self.cv.notify_one();
        }
    }

    pub fn pop(&self, class_preference: &[WorkloadClass]) -> Option<WorkItem> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if inner.closed
                && inner.q_cpu_short.is_empty()
                && inner.q_cpu_long.is_empty()
                && inner.q_gpu.is_empty()
            {
                return None;
            }
            for class in class_preference {
                let opt = match class {
                    WorkloadClass::CpuShort => inner.q_cpu_short.pop_front(),
                    WorkloadClass::CpuLong => inner.q_cpu_long.pop_front(),
                    WorkloadClass::Gpu => inner.q_gpu.pop_front(),
                };
                if opt.is_some() {
                    return opt;
                }
            }
            inner = self.cv.wait(inner).unwrap();
        }
    }

    pub fn close(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
        self.cv.notify_all();
    }

    pub fn len(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.q_cpu_short.len() + inner.q_cpu_long.len() + inner.q_gpu.len()
    }
}

/// Runner pulls from the queue and processes items according to class-specific policies.
pub struct Runner {
    pub queue: Arc<PrefetchQueue>,
    pub processed: Arc<Mutex<Vec<String>>>,
    pub metrics: Arc<EnvRunnerMetrics>,
}

impl Runner {
    pub fn start(self, shutdown: Arc<AtomicBool>) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let Runner {
                queue,
                processed,
                metrics,
            } = self;
            let prefs = [
                WorkloadClass::Gpu,
                WorkloadClass::CpuShort,
                WorkloadClass::CpuLong,
            ];
            let mut counts_gpu: u64 = 0;
            let mut counts_cpu_s: u64 = 0;
            let mut counts_cpu_l: u64 = 0;
            let base =
                std::env::var("ENV_QUEUE_DIR").unwrap_or_else(|_| "logs/env_queue".to_string());
            let metrics_path = PathBuf::from(base.clone()).join("runner_metrics.json");
            let logs_path = PathBuf::from(base).parent().map(|p| p.to_path_buf());
            let mut last_metrics = Instant::now();

            while !shutdown.load(Ordering::Relaxed) {
                match queue.pop(&prefs) {
                    Some(item) => {
                        let started = Instant::now();
                        // Simulate class-specific handling
                        match item.class {
                            WorkloadClass::Gpu => {
                                thread::sleep(Duration::from_millis(5));
                                counts_gpu += 1;
                            }
                            WorkloadClass::CpuShort => {
                                thread::sleep(Duration::from_millis(2));
                                counts_cpu_s += 1;
                            }
                            WorkloadClass::CpuLong => {
                                thread::sleep(Duration::from_millis(10));
                                counts_cpu_l += 1;
                            }
                        }
                        let elapsed = started.elapsed();
                        metrics.increment_tasks_processed();
                        metrics.record_task_duration(elapsed);
                        metrics.update_queue_depth(queue.len() as u64);

                        processed.lock().unwrap().push(item.id.clone());

                        // write event to logs/agent_action.jsonl (best-effort)
                        if let Some(logd) = &logs_path {
                            let evt = format!(
                                "{}\n",
                                json!({"topic":"agent.action","ts": chrono::Utc::now().timestamp(), "event": {"name":"env_task_done","id": item.id}})
                            );
                            let file = logd.join("agent_action.jsonl");
                            if let Ok(mut fh) =
                                OpenOptions::new().create(true).append(true).open(file)
                            {
                                let _ = fh.write_all(evt.as_bytes());
                            }
                        }

                        if last_metrics.elapsed() > Duration::from_secs(1) {
                            let metrics_snapshot = json!({
                                "gpu": counts_gpu, "cpu_short": counts_cpu_s, "cpu_long": counts_cpu_l,
                                "processed": processed.lock().unwrap().len(), "ts": chrono::Utc::now().timestamp(),
                            });
                            if let Ok(mut fh) = OpenOptions::new()
                                .create(true)
                                .write(true)
                                .truncate(true)
                                .open(&metrics_path)
                            {
                                let _ = fh.write_all(metrics_snapshot.to_string().as_bytes());
                            }
                            last_metrics = Instant::now();
                        }
                    }
                    None => {
                        break;
                    }
                }
            }

            metrics.update_queue_depth(queue.len() as u64);
        })
    }
}

/// Simple feeder that prefetches from a source iterator into the queue (bounded by capacity).
pub fn prefetch_from<I: Iterator<Item = WorkItem> + Send + 'static>(
    queue: Arc<PrefetchQueue>,
    iter: I,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        for item in iter {
            queue.push(item);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_partitions_and_pops_by_preference() {
        let q = Arc::new(PrefetchQueue::new(10));
        q.push(WorkItem {
            id: "a".into(),
            class: WorkloadClass::CpuLong,
            payload: "".into(),
        });
        q.push(WorkItem {
            id: "b".into(),
            class: WorkloadClass::Gpu,
            payload: "".into(),
        });
        q.push(WorkItem {
            id: "c".into(),
            class: WorkloadClass::CpuShort,
            payload: "".into(),
        });
        let first = q
            .pop(&[
                WorkloadClass::Gpu,
                WorkloadClass::CpuShort,
                WorkloadClass::CpuLong,
            ])
            .unwrap();
        assert_eq!(first.id, "b");
        let second = q
            .pop(&[WorkloadClass::CpuShort, WorkloadClass::CpuLong])
            .unwrap();
        assert_eq!(second.id, "c");
        let third = q.pop(&[WorkloadClass::CpuLong]).unwrap();
        assert_eq!(third.id, "a");
    }

    #[test]
    fn runner_processes_and_records() {
        let q = Arc::new(PrefetchQueue::new(100));
        let processed = Arc::new(Mutex::new(Vec::new()));
        let metrics = Arc::new(EnvRunnerMetrics::new());
        let runner = Runner {
            queue: q.clone(),
            processed: processed.clone(),
            metrics: metrics.clone(),
        };
        let shutdown = Arc::new(AtomicBool::new(false));
        let handle = runner.start(shutdown.clone());
        for i in 0..5 {
            q.push(WorkItem {
                id: format!("id-{i}"),
                class: if i % 2 == 0 {
                    WorkloadClass::Gpu
                } else {
                    WorkloadClass::CpuShort
                },
                payload: "".into(),
            });
        }
        // Wait for processing
        let start = Instant::now();
        while processed.lock().unwrap().len() < 5 {
            if start.elapsed() > Duration::from_secs(1) {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        // Terminate thread (test-only): process will block on pop, so detach
        assert!(processed.lock().unwrap().len() >= 5);
        shutdown.store(true, Ordering::SeqCst);
        q.close();
        let _ = handle.join();
    }
}
