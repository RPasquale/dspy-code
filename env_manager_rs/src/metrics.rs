//! Prometheus metrics for the environment manager.

use prometheus::{
    CounterVec, Encoder, Gauge, HistogramOpts, HistogramVec, Opts, Registry, TextEncoder,
};

/// MetricsRegistry wraps a Prometheus registry and provides typed metric accessors.
#[derive(Clone)]
pub struct MetricsRegistry {
    registry: Registry,

    // Service lifecycle metrics
    pub service_start_count: CounterVec,
    pub service_stop_count: CounterVec,
    pub failed_service_starts: CounterVec,
    pub active_services: Gauge,

    // Performance metrics
    pub service_health_check_duration_seconds: HistogramVec,
    pub docker_api_duration_seconds: HistogramVec,
}

impl MetricsRegistry {
    /// Create a new metrics registry with all metrics registered.
    pub fn new() -> anyhow::Result<Self> {
        let registry = Registry::new();

        // Service start counter
        let service_start_count = CounterVec::new(
            Opts::new(
                "env_manager_service_start_total",
                "Total number of service start attempts",
            ),
            &["service_name", "status"],
        )?;
        registry.register(Box::new(service_start_count.clone()))?;

        // Service stop counter
        let service_stop_count = CounterVec::new(
            Opts::new(
                "env_manager_service_stop_total",
                "Total number of service stop attempts",
            ),
            &["service_name", "status"],
        )?;
        registry.register(Box::new(service_stop_count.clone()))?;

        // Failed service starts
        let failed_service_starts = CounterVec::new(
            Opts::new(
                "env_manager_failed_service_starts_total",
                "Total number of failed service start attempts",
            ),
            &["service_name", "reason"],
        )?;
        registry.register(Box::new(failed_service_starts.clone()))?;

        // Active services gauge
        let active_services = Gauge::new(
            "env_manager_active_services",
            "Number of currently active services",
        )?;
        registry.register(Box::new(active_services.clone()))?;

        // Health check duration histogram
        let service_health_check_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "env_manager_service_health_check_duration_seconds",
                "Time taken to perform service health checks",
            )
            .buckets(vec![0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]),
            &["service_name", "status"],
        )?;
        registry.register(Box::new(service_health_check_duration_seconds.clone()))?;

        // Docker API duration histogram
        let docker_api_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "env_manager_docker_api_duration_seconds",
                "Time taken for Docker API calls",
            )
            .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]),
            &["operation", "status"],
        )?;
        registry.register(Box::new(docker_api_duration_seconds.clone()))?;

        Ok(Self {
            registry,
            service_start_count,
            service_stop_count,
            failed_service_starts,
            active_services,
            service_health_check_duration_seconds,
            docker_api_duration_seconds,
        })
    }

    /// Gather all metrics and encode them in Prometheus text format.
    pub fn gather(&self) -> anyhow::Result<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new().expect("Failed to create default metrics registry")
    }
}
