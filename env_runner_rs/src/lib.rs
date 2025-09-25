use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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
}

/// PrefetchQueue partitions items by workload class and supports bounded prefetching.
pub struct PrefetchQueue {
    inner: Mutex<InnerQ>,
    cv: Condvar,
    cap_per_class: usize,
}

impl PrefetchQueue {
    pub fn new(cap_per_class: usize) -> Self {
        Self { inner: Mutex::new(InnerQ::default()), cv: Condvar::new(), cap_per_class }
    }

    pub fn push(&self, item: WorkItem) {
        let mut inner = self.inner.lock().unwrap();
        let q = match item.class { WorkloadClass::CpuShort => &mut inner.q_cpu_short, WorkloadClass::CpuLong => &mut inner.q_cpu_long, WorkloadClass::Gpu => &mut inner.q_gpu };
        if q.len() < self.cap_per_class { q.push_back(item); self.cv.notify_one(); }
    }

    pub fn pop(&self, class_preference: &[WorkloadClass]) -> Option<WorkItem> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            for class in class_preference {
                let opt = match class {
                    WorkloadClass::CpuShort => inner.q_cpu_short.pop_front(),
                    WorkloadClass::CpuLong => inner.q_cpu_long.pop_front(),
                    WorkloadClass::Gpu => inner.q_gpu.pop_front(),
                };
                if opt.is_some() { return opt; }
            }
            inner = self.cv.wait(inner).unwrap();
        }
    }
}

/// Runner pulls from the queue and processes items according to class-specific policies.
pub struct Runner {
    pub queue: Arc<PrefetchQueue>,
    pub processed: Arc<Mutex<Vec<String>>>,
}

impl Runner {
    pub fn start(self) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let prefs = [WorkloadClass::Gpu, WorkloadClass::CpuShort, WorkloadClass::CpuLong];
            loop {
                let item = self.queue.pop(&prefs).unwrap();
                // Simulate class-specific handling
                match item.class {
                    WorkloadClass::Gpu => thread::sleep(Duration::from_millis(5)),
                    WorkloadClass::CpuShort => thread::sleep(Duration::from_millis(2)),
                    WorkloadClass::CpuLong => thread::sleep(Duration::from_millis(10)),
                }
                self.processed.lock().unwrap().push(item.id);
            }
        })
    }
}

/// Simple feeder that prefetches from a source iterator into the queue (bounded by capacity).
pub fn prefetch_from<I: Iterator<Item = WorkItem> + Send + 'static>(queue: Arc<PrefetchQueue>, iter: I) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        for item in iter { queue.push(item); }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_partitions_and_pops_by_preference() {
        let q = Arc::new(PrefetchQueue::new(10));
        q.push(WorkItem { id: "a".into(), class: WorkloadClass::CpuLong, payload: "".into() });
        q.push(WorkItem { id: "b".into(), class: WorkloadClass::Gpu, payload: "".into() });
        q.push(WorkItem { id: "c".into(), class: WorkloadClass::CpuShort, payload: "".into() });
        let first = q.pop(&[WorkloadClass::Gpu, WorkloadClass::CpuShort, WorkloadClass::CpuLong]).unwrap();
        assert_eq!(first.id, "b");
        let second = q.pop(&[WorkloadClass::CpuShort, WorkloadClass::CpuLong]).unwrap();
        assert_eq!(second.id, "c");
        let third = q.pop(&[WorkloadClass::CpuLong]).unwrap();
        assert_eq!(third.id, "a");
    }

    #[test]
    fn runner_processes_and_records() {
        let q = Arc::new(PrefetchQueue::new(100));
        let processed = Arc::new(Mutex::new(Vec::new()));
        let runner = Runner { queue: q.clone(), processed: processed.clone() };
        let handle = runner.start();
        for i in 0..5 {
            q.push(WorkItem { id: format!("id-{i}"), class: if i % 2 == 0 { WorkloadClass::Gpu } else { WorkloadClass::CpuShort }, payload: "".into() });
        }
        // Wait for processing
        let start = Instant::now();
        while processed.lock().unwrap().len() < 5 {
            if start.elapsed() > Duration::from_secs(1) { break; }
            thread::sleep(Duration::from_millis(5));
        }
        // Terminate thread (test-only): process will block on pop, so detach
        assert!(processed.lock().unwrap().len() >= 5);
        let _ = handle.thread().id();
    }
}

