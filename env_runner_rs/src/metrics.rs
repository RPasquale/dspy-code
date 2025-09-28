use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

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

            // Keep only last 1000 durations to prevent memory growth
            if durations.len() > 1000 {
                durations.remove(0);
            }

            // Update P95 latency
            if durations.len() >= 20 {
                let mut sorted = durations.clone();
                sorted.sort();
                let p95_index = (sorted.len() as f64 * 0.95) as usize;
                if let Ok(mut p95) = self.latency_p95.lock() {
                    *p95 = sorted[p95_index];
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
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsStats {
    pub tasks_processed: u64,
    pub queue_depth: u64,
    pub gpu_utilization: f64,
    pub latency_p95_ms: u64,
    pub avg_duration_ms: u64,
    pub total_errors: u64,
    pub errors_by_class: HashMap<String, u64>,
    pub uptime_seconds: u64,
}

impl Default for EnvRunnerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// HTTP metrics server for the Rust environment runner
pub struct MetricsServer {
    metrics: Arc<EnvRunnerMetrics>,
    port: u16,
}

impl MetricsServer {
    pub fn new(metrics: Arc<EnvRunnerMetrics>, port: u16) -> Self {
        Self { metrics, port }
    }

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use warp::Filter;

        let metrics = self.metrics.clone();
        let metrics_for_prometheus = self.metrics.clone();

        // Health check endpoint
        let health = warp::path("health")
            .map(|| warp::reply::json(&serde_json::json!({"status": "healthy"})));

        // Metrics endpoint
        let metrics_endpoint = warp::path("metrics").map(move || {
            let stats = metrics.get_stats();
            warp::reply::json(&stats)
        });

        // Prometheus format endpoint
        let prometheus = warp::path("prometheus").map(move || {
            let stats = metrics_for_prometheus.get_stats();
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
                     env_runner_uptime_seconds {}\n",
                stats.tasks_processed,
                stats.queue_depth,
                stats.gpu_utilization,
                stats.latency_p95_ms as f64 / 1000.0,
                stats.avg_duration_ms as f64 / 1000.0,
                stats.total_errors,
                stats.uptime_seconds
            )
        });

        let routes = health.or(metrics_endpoint).or(prometheus);

        println!("Starting metrics server on port {}", self.port);
        warp::serve(routes).run(([0, 0, 0, 0], self.port)).await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_metrics_basic() {
        let metrics = EnvRunnerMetrics::new();

        // Test initial state
        let stats = metrics.get_stats();
        assert_eq!(stats.tasks_processed, 0);
        assert_eq!(stats.queue_depth, 0);
        assert_eq!(stats.total_errors, 0);

        // Test incrementing
        metrics.increment_tasks_processed();
        metrics.update_queue_depth(5);
        metrics.increment_error("test_class");

        let stats = metrics.get_stats();
        assert_eq!(stats.tasks_processed, 1);
        assert_eq!(stats.queue_depth, 5);
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.errors_by_class.get("test_class"), Some(&1));
    }

    #[test]
    fn test_task_duration_recording() {
        let metrics = EnvRunnerMetrics::new();

        // Record some durations
        metrics.record_task_duration(Duration::from_millis(100));
        metrics.record_task_duration(Duration::from_millis(200));
        metrics.record_task_duration(Duration::from_millis(300));

        let stats = metrics.get_stats();
        assert!(stats.avg_duration_ms > 0);
        assert!(stats.latency_p95_ms > 0);
    }

    #[test]
    fn test_gpu_utilization() {
        let metrics = EnvRunnerMetrics::new();

        metrics.update_gpu_utilization(75.5);

        let stats = metrics.get_stats();
        assert_eq!(stats.gpu_utilization, 75.5);
    }
}
