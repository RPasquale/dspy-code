use crate::executor::execute_task;
use crate::hardware::HardwareSnapshot;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use warp::http::StatusCode;
use warp::Filter;

/// Metrics for the Rust environment runner
#[derive(Debug, Clone)]
pub struct EnvRunnerMetrics {
    pub tasks_processed: Arc<Mutex<u64>>,
    pub task_duration: Arc<Mutex<Vec<Duration>>>,
    pub queue_depth: Arc<Mutex<u64>>,
    pub gpu_utilization: Arc<Mutex<f64>>,
    pub errors_by_class: Arc<Mutex<HashMap<String, u64>>>,
    pub latency_p95: Arc<Mutex<Duration>>,
    pub start_time: Instant,
}

impl EnvRunnerMetrics {
    pub fn new() -> Self {
        Self {
            tasks_processed: Arc::new(Mutex::new(0)),
            task_duration: Arc::new(Mutex::new(Vec::new())),
            queue_depth: Arc::new(Mutex::new(0)),
            gpu_utilization: Arc::new(Mutex::new(0.0)),
            errors_by_class: Arc::new(Mutex::new(HashMap::new())),
            latency_p95: Arc::new(Mutex::new(Duration::from_millis(0))),
            start_time: Instant::now(),
        }
    }

    pub fn increment_tasks_processed(&self) {
        if let Ok(mut count) = self.tasks_processed.lock() {
            *count += 1;
        }
    }

    pub fn record_task_duration(&self, duration: Duration) {
        if let Ok(mut durations) = self.task_duration.lock() {
            durations.push(duration);
            if durations.len() > 1000 {
                durations.remove(0);
            }
            if durations.len() >= 20 {
                let mut sorted = durations.clone();
                sorted.sort();
                let p95_index = ((sorted.len() as f64) * 0.95).floor() as usize;
                if let Some(item) = sorted.get(p95_index.min(sorted.len() - 1)) {
                    if let Ok(mut p95) = self.latency_p95.lock() {
                        *p95 = *item;
                    }
                }
            }
        }
    }

    pub fn update_queue_depth(&self, depth: u64) {
        if let Ok(mut queue_depth) = self.queue_depth.lock() {
            *queue_depth = depth;
        }
    }

    pub fn update_gpu_utilization(&self, utilization: f64) {
        if let Ok(mut gpu_util) = self.gpu_utilization.lock() {
            *gpu_util = utilization;
        }
    }

    pub fn increment_error(&self, class: &str) {
        if let Ok(mut errors) = self.errors_by_class.lock() {
            *errors.entry(class.to_string()).or_insert(0) += 1;
        }
    }

    pub fn get_stats(&self) -> MetricsStats {
        let tasks_processed = self.tasks_processed.lock().map(|v| *v).unwrap_or(0);
        let queue_depth = self.queue_depth.lock().map(|v| *v).unwrap_or(0);
        let gpu_utilization = self.gpu_utilization.lock().map(|v| *v).unwrap_or(0.0);
        let latency_p95 = self
            .latency_p95
            .lock()
            .map(|v| *v)
            .unwrap_or(Duration::from_millis(0));

        let durations = self
            .task_duration
            .lock()
            .map(|v| v.clone())
            .unwrap_or_default();
        let avg_duration = if !durations.is_empty() {
            durations.iter().sum::<Duration>() / durations.len() as u32
        } else {
            Duration::from_millis(0)
        };

        let errors_by_class = self
            .errors_by_class
            .lock()
            .map(|v| v.clone())
            .unwrap_or_default();
        let total_errors: u64 = errors_by_class.values().sum();

        MetricsStats {
            tasks_processed,
            queue_depth,
            gpu_utilization,
            latency_p95_ms: latency_p95.as_millis() as u64,
            avg_duration_ms: avg_duration.as_millis() as u64,
            total_errors,
            errors_by_class,
            uptime_seconds: self.start_time.elapsed().as_secs(),
            hardware: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MetricsStats {
    pub tasks_processed: u64,
    pub queue_depth: u64,
    pub gpu_utilization: f64,
    pub latency_p95_ms: u64,
    pub avg_duration_ms: u64,
    pub total_errors: u64,
    pub errors_by_class: HashMap<String, u64>,
    pub uptime_seconds: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware: Option<HardwareSnapshot>,
}

impl Default for EnvRunnerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
pub struct TaskRequest {
    pub id: String,
    #[serde(default)]
    pub class: Option<String>,
    pub payload: Value,
}

#[derive(Debug, Serialize)]
pub struct TaskExecutionResponse {
    pub id: String,
    pub embeddings: Vec<Vec<f64>>,
    pub latency_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

/// HTTP metrics and task execution server for the Rust environment runner.
pub struct MetricsServer {
    metrics: Arc<EnvRunnerMetrics>,
    hardware: Arc<RwLock<HardwareSnapshot>>,
    port: u16,
    shutdown: Arc<Notify>,
}

impl MetricsServer {
    pub fn new(
        metrics: Arc<EnvRunnerMetrics>,
        hardware: Arc<RwLock<HardwareSnapshot>>,
        port: u16,
    ) -> Self {
        Self {
            metrics,
            hardware,
            port,
            shutdown: Arc::new(Notify::new()),
        }
    }

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let metrics = self.metrics.clone();
        let metrics_for_prometheus = self.metrics.clone();
        let hardware_for_metrics = self.hardware.clone();
        let hardware_for_prometheus = self.hardware.clone();
        let hardware_for_endpoint = self.hardware.clone();
        let shutdown = self.shutdown.clone();

        let health = warp::path("health")
            .map(|| warp::reply::json(&serde_json::json!({ "status": "healthy" })))
            .boxed();

        let metrics_endpoint = warp::path("metrics")
            .map(move || {
                let mut stats = metrics.get_stats();
                if let Ok(snapshot) = hardware_for_metrics.read() {
                    stats.hardware = Some(snapshot.clone());
                }
                warp::reply::json(&stats)
            })
            .boxed();

        let prometheus = warp::path("prometheus")
            .map(move || {
                let stats = metrics_for_prometheus.get_stats();
                let gpu_total: u64 = hardware_for_prometheus
                    .read()
                    .map(|snapshot| {
                        snapshot
                            .accelerators
                            .iter()
                            .map(|acc| acc.count as u64)
                            .sum()
                    })
                    .unwrap_or_default();
                format!(
                    "# HELP env_runner_tasks_processed_total Total number of tasks processed\n\
                 # TYPE env_runner_tasks_processed_total counter\n\
                 env_runner_tasks_processed_total {}\n\
                 \n\
                 # HELP env_runner_queue_depth Current queue depth\n\
                 # TYPE env_runner_queue_depth gauge\n\
                 env_runner_queue_depth {}\n\
                 \n\
                 # HELP env_runner_gpu_utilization GPU utilization percentage\n\
                 # TYPE env_runner_gpu_utilization gauge\n\
                 env_runner_gpu_utilization {}\n\
                 \n\
                 # HELP env_runner_latency_p95_seconds P95 latency in seconds\n\
                 # TYPE env_runner_latency_p95_seconds gauge\n\
                 env_runner_latency_p95_seconds {}\n\
                 \n\
                 # HELP env_runner_avg_duration_seconds Average task duration in seconds\n\
                 # TYPE env_runner_avg_duration_seconds gauge\n\
                 env_runner_avg_duration_seconds {}\n\
                 \n\
                 # HELP env_runner_errors_total Total number of errors\n\
                 # TYPE env_runner_errors_total counter\n\
                 env_runner_errors_total {}\n\
                 \n\
                 # HELP env_runner_uptime_seconds Uptime in seconds\n\
                 # TYPE env_runner_uptime_seconds gauge\n\
                 env_runner_uptime_seconds {}\n\
                 \n\
                 # HELP env_runner_gpu_total_total Reported GPU devices\n\
                 # TYPE env_runner_gpu_total_total gauge\n\
                 env_runner_gpu_total_total {}\n",
                    stats.tasks_processed,
                    stats.queue_depth,
                    stats.gpu_utilization,
                    stats.latency_p95_ms as f64 / 1000.0,
                    stats.avg_duration_ms as f64 / 1000.0,
                    stats.total_errors,
                    stats.uptime_seconds,
                    gpu_total
                )
            })
            .boxed();

        let hardware = warp::path("hardware")
            .map(move || {
                let snapshot = hardware_for_endpoint
                    .read()
                    .map(|snapshot| snapshot.clone())
                    .unwrap_or_default();
                warp::reply::json(&snapshot)
            })
            .boxed();

        let tasks = warp::path("tasks")
            .and(warp::path("execute"))
            .and(warp::post())
            .and(with_metrics(self.metrics.clone()))
            .and(warp::body::json())
            .and_then(handle_task_request)
            .boxed();

        let routes = health
            .or(metrics_endpoint)
            .or(prometheus)
            .or(hardware)
            .or(tasks)
            .with(warp::log("env_runner"));

        let (_, server) = warp::serve(routes).bind_with_graceful_shutdown(
            ([0, 0, 0, 0], self.port),
            async move {
                shutdown.notified().await;
            },
        );

        server.await;

        Ok(())
    }

    pub async fn shutdown(&self) {
        self.shutdown.notify_waiters();
    }
}

fn with_metrics(
    metrics: Arc<EnvRunnerMetrics>,
) -> impl Filter<Extract = (Arc<EnvRunnerMetrics>,), Error = Infallible> + Clone {
    warp::any().map(move || metrics.clone())
}

async fn handle_task_request(
    metrics: Arc<EnvRunnerMetrics>,
    request: TaskRequest,
) -> Result<impl warp::Reply, warp::Rejection> {
    let class = request
        .class
        .clone()
        .unwrap_or_else(|| "cpu_short".to_string());

    let started = Instant::now();
    match execute_task(&request.id, &class, &request.payload).await {
        Ok(result) if result.success => {
            metrics.increment_tasks_processed();
            metrics.record_task_duration(result.duration);
            metrics.update_queue_depth(0);

            if class.to_ascii_lowercase().contains("gpu") {
                metrics.update_gpu_utilization(0.85);
            }

            let response = TaskExecutionResponse {
                id: request.id,
                embeddings: Vec::new(),
                latency_ms: started.elapsed().as_secs_f64() * 1000.0,
                metadata: Some(result.metadata),
            };

            let body = warp::reply::json(&response);
            Ok(warp::reply::with_status(body, StatusCode::OK))
        }
        Ok(result) => {
            metrics.increment_error(&class);
            let body = warp::reply::json(&ErrorResponse {
                error: result
                    .error
                    .unwrap_or_else(|| "task execution failed".into()),
            });
            Ok(warp::reply::with_status(
                body,
                StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
        Err(err) => {
            metrics.increment_error(&class);
            let body = warp::reply::json(&ErrorResponse {
                error: err.to_string(),
            });
            Ok(warp::reply::with_status(
                body,
                StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }
}
