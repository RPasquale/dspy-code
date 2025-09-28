use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use env_runner_rs::metrics::{EnvRunnerMetrics, MetricsServer};
use env_runner_rs::notify_watcher::NotifyWatcher;
use env_runner_rs::{PrefetchQueue, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let queue = Arc::new(PrefetchQueue::new(128));
    let processed = Arc::new(Mutex::new(Vec::new()));
    let metrics = Arc::new(EnvRunnerMetrics::new());
    let shutdown = Arc::new(AtomicBool::new(false));

    // Runner thread
    let runner = Runner {
        queue: queue.clone(),
        processed: processed.clone(),
        metrics: metrics.clone(),
    };
    let runner_handle = runner.start(shutdown.clone());

    // File queue directories
    let base_dir = std::env::var("ENV_QUEUE_DIR").unwrap_or_else(|_| "logs/env_queue".to_string());
    let pend_dir = Path::new(&base_dir).join("pending");
    let done_dir = Path::new(&base_dir).join("done");

    // Notify watcher
    let mut watcher = NotifyWatcher::new(
        pend_dir.clone(),
        done_dir.clone(),
        queue.clone(),
        metrics.clone(),
    )?;
    let watcher_shutdown = shutdown.clone();
    let watcher_task = tokio::spawn(async move {
        if let Err(err) = watcher.run(&watcher_shutdown).await {
            eprintln!("notify watcher exited with error: {err}");
        }
    });

    // Metrics server (also serves frontend API)
    let metrics_server = MetricsServer::new(metrics.clone(), 8083);
    let metrics_task = tokio::spawn(async move {
        if let Err(err) = metrics_server.start().await {
            eprintln!("metrics server error: {err}");
        }
    });

    // Wait for shutdown signal
    tokio::signal::ctrl_c().await?;
    shutdown.store(true, Ordering::SeqCst);
    queue.close();

    // Allow watcher task to exit
    let _ = watcher_task.await;
    metrics_task.abort();
    let _ = metrics_task.await;

    let _ = runner_handle.join();

    Ok(())
}
