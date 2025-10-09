use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use notify::event::{ModifyKind, RemoveKind};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::time::interval;

use crate::metrics::EnvRunnerMetrics;
use crate::{PrefetchQueue, WorkItem, WorkloadClass};

/// File system watcher for queue directory changes
pub struct NotifyWatcher {
    _watcher: RecommendedWatcher,
    queue: Arc<PrefetchQueue>,
    metrics: Arc<EnvRunnerMetrics>,
    pend_dir: PathBuf,
    done_dir: PathBuf,
    event_rx: mpsc::UnboundedReceiver<notify::Result<Event>>,
}

impl NotifyWatcher {
    pub fn new(
        pend_dir: impl Into<PathBuf>,
        done_dir: impl Into<PathBuf>,
        queue: Arc<PrefetchQueue>,
        metrics: Arc<EnvRunnerMetrics>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let pend_dir = pend_dir.into();
        let done_dir = done_dir.into();

        fs::create_dir_all(&pend_dir)?;
        fs::create_dir_all(&done_dir)?;

        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let tx_clone = event_tx.clone();

        let mut watcher = notify::recommended_watcher(move |res| {
            if let Err(err) = tx_clone.send(res) {
                eprintln!("Failed to forward notify event: {err}");
            }
        })?;

        watcher.watch(&pend_dir, RecursiveMode::NonRecursive)?;
        watcher.watch(&done_dir, RecursiveMode::NonRecursive)?;

        Ok(Self {
            _watcher: watcher,
            queue,
            metrics,
            pend_dir,
            done_dir,
            event_rx,
        })
    }

    pub async fn run(
        &mut self,
        shutdown: &AtomicBool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut tick = interval(Duration::from_millis(250));
        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            tokio::select! {
                maybe_event = self.event_rx.recv() => {
                    if let Some(res) = maybe_event {
                        self.handle_event(res).await;
                    } else {
                        break;
                    }
                }
                _ = tick.tick() => {
                    self.scan_pending().await;
                }
            }
        }

        Ok(())
    }

    async fn handle_event(&self, res: notify::Result<Event>) {
        match res {
            Ok(event) => {
                if self.should_rescan(&event.kind) {
                    self.scan_pending().await;
                    return;
                }

                for path in event.paths {
                    if !path.starts_with(&self.pend_dir) {
                        continue;
                    }
                    if !is_json_file(&path) {
                        continue;
                    }
                    self.ingest_path(path).await;
                }
            }
            Err(err) => eprintln!("Notify watcher error: {err}"),
        }
    }

    fn should_rescan(&self, kind: &EventKind) -> bool {
        matches!(
            kind,
            EventKind::Remove(RemoveKind::Any)
                | EventKind::Remove(RemoveKind::File)
                | EventKind::Modify(ModifyKind::Metadata(_))
                | EventKind::Modify(ModifyKind::Name(_))
        )
    }

    async fn ingest_path(&self, path: PathBuf) {
        let pend_dir = self.pend_dir.clone();
        let done_dir = self.done_dir.clone();
        let queue = self.queue.clone();
        let metrics = self.metrics.clone();

        let _ = tokio::task::spawn_blocking(move || {
            process_file(&pend_dir, &done_dir, &path, queue, metrics);
        })
        .await;
    }

    async fn scan_pending(&self) {
        let pend_dir = self.pend_dir.clone();
        let done_dir = self.done_dir.clone();
        let queue = self.queue.clone();
        let metrics = self.metrics.clone();

        let _ = tokio::task::spawn_blocking(move || {
            if let Ok(entries) = fs::read_dir(&pend_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_file() || !is_json_file(&path) {
                        continue;
                    }
                    process_file(&pend_dir, &done_dir, &path, queue.clone(), metrics.clone());
                }
            }
            update_queue_depth(&pend_dir, &metrics);
        })
        .await;
    }
}

fn process_file(
    pend_dir: &Path,
    done_dir: &Path,
    path: &Path,
    queue: Arc<PrefetchQueue>,
    metrics: Arc<EnvRunnerMetrics>,
) {
    if !path.starts_with(pend_dir) {
        return;
    }

    let lock_path = path.with_extension("lock");
    if fs::rename(path, &lock_path).is_err() {
        return;
    }

    match read_task_file(&lock_path) {
        Some(item) => {
            queue.push(item);
            let done_path = done_dir.join(path.file_name().unwrap_or_default());
            if let Err(err) = fs::rename(&lock_path, done_path) {
                eprintln!("Failed to move processed file to done: {err}");
            }
        }
        None => {
            let _ = fs::remove_file(&lock_path);
        }
    }

    update_queue_depth(pend_dir, &metrics);
}

fn update_queue_depth(pend_dir: &Path, metrics: &Arc<EnvRunnerMetrics>) {
    let depth = fs::read_dir(pend_dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .filter(|entry| entry.path().is_file() && is_json_file(&entry.path()))
                .count() as u64
        })
        .unwrap_or(0);
    metrics.update_queue_depth(depth);
}

fn is_json_file(path: &Path) -> bool {
    matches!(path.extension().and_then(|ext| ext.to_str()), Some("json"))
}

fn read_task_file(path: &Path) -> Option<WorkItem> {
    let content = fs::read_to_string(path).ok()?;
    let task: TaskEnvelope = serde_json::from_str(&content).ok()?;

    let class_label = task.class.unwrap_or_else(|| "cpu_short".to_owned());
    let class = WorkloadClass::from_str(&class_label).unwrap_or(WorkloadClass::CpuShort);

    Some(WorkItem {
        id: task.id,
        class,
        payload: task.payload.to_string(),
    })
}

#[derive(Debug, Deserialize, Serialize)]
struct TaskEnvelope {
    id: String,
    class: Option<String>,
    payload: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_notify_watcher_basic() {
        let temp_dir = tempdir().unwrap();
        let pend_dir = temp_dir.path().join("pending");
        let done_dir = temp_dir.path().join("done");

        fs::create_dir_all(&pend_dir).unwrap();
        fs::create_dir_all(&done_dir).unwrap();

        let queue = Arc::new(PrefetchQueue::new(128));
        let metrics = Arc::new(EnvRunnerMetrics::new());
        let watcher = NotifyWatcher::new(
            pend_dir.clone(),
            done_dir.clone(),
            queue.clone(),
            metrics.clone(),
        )
        .unwrap();

        let shutdown = Arc::new(AtomicBool::new(false));
        let mut watcher_ref = watcher;
        let shutdown_handle = shutdown.clone();
        let task = tokio::spawn(async move {
            watcher_ref.run(shutdown_handle.as_ref()).await.unwrap();
        });

        // Create a test task file
        let task_file = pend_dir.join("test_task.json");
        let task_data = serde_json::json!({
            "id": "test_123",
            "class": "cpu_short",
            "payload": {"test": "data"}
        });

        fs::write(&task_file, serde_json::to_string(&task_data).unwrap()).unwrap();

        // Allow watcher to process the file
        tokio::time::sleep(Duration::from_millis(200)).await;

        shutdown.store(true, Ordering::SeqCst);
        queue.close();
        task.await.unwrap();

        assert_eq!(queue.len(), 1);
        assert_eq!(metrics.get_stats().queue_depth, 0);
    }
}
