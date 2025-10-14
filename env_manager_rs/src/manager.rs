use crate::config::Config;
use crate::container::ContainerManager;
use crate::health::HealthChecker;
use crate::metrics::MetricsRegistry;
use crate::retry::{retry_with_backoff, RetryConfig};
use crate::service_registry::{ServiceDefinition, ServiceRegistry};
use anyhow::{Context, Result};
use bollard::Docker;
// PortBinding is now in service_registry
// use bollard::models::PortBinding;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, Semaphore};
use tracing::{error, info, warn};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ServiceState {
    pub name: String,
    pub container_id: Option<String>,
    pub status: ServiceStatus,
    pub started_at: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ServiceStatus {
    Stopped,
    Starting,
    Running,
    Unhealthy,
    Failed(String),
}

impl ServiceStatus {
    pub fn as_str(&self) -> &str {
        match self {
            ServiceStatus::Stopped => "stopped",
            ServiceStatus::Starting => "starting",
            ServiceStatus::Running => "running",
            ServiceStatus::Unhealthy => "unhealthy",
            ServiceStatus::Failed(_) => "failed",
        }
    }
}

pub struct EnvManager {
    container_manager: Arc<ContainerManager>,
    health_checker: Arc<HealthChecker>,
    registry: Arc<ServiceRegistry>,
    state: Arc<RwLock<HashMap<String, ServiceState>>>,
    metrics: Arc<MetricsRegistry>,
    config: Arc<Config>,
}

impl EnvManager {
    pub async fn new(
        docker: Docker,
        metrics: Arc<MetricsRegistry>,
        config: Arc<Config>,
    ) -> Result<Self> {
        let container_manager = Arc::new(ContainerManager::new(docker));
        let health_checker = Arc::new(HealthChecker::new());
        let registry = Arc::new(ServiceRegistry::new());
        let state = Arc::new(RwLock::new(HashMap::new()));

        // Initialize state for all services
        {
            let mut state_guard = state.write().await;
            for service in registry.get_all() {
                state_guard.insert(
                    service.name.clone(),
                    ServiceState {
                        name: service.name.clone(),
                        container_id: None,
                        status: ServiceStatus::Stopped,
                        started_at: None,
                    },
                );
            }
        }

        Ok(Self {
            container_manager,
            health_checker,
            registry,
            state,
            metrics,
            config,
        })
    }

    /// Start all services in dependency order
    #[allow(dead_code)]
    pub async fn start_all_services(&self, parallel: bool) -> Result<()> {
        self.start_services(&[], parallel).await
    }

    /// Start selected services. Empty list => all services.
    pub async fn start_services(&self, service_names: &[String], parallel: bool) -> Result<()> {
        info!(
            "Starting services (parallel: {}, selection: {:?})",
            parallel, service_names
        );

        let filter: Option<HashSet<String>> = if service_names.is_empty() {
            None
        } else {
            Some(service_names.iter().map(|s| s.to_string()).collect())
        };

        let services = self.registry.get_startup_order();
        let services: Vec<&ServiceDefinition> = services
            .into_iter()
            .filter(|service| {
                filter
                    .as_ref()
                    .map(|set| set.contains(&service.name))
                    .unwrap_or(true)
            })
            .collect();

        if services.is_empty() {
            if let Some(filter) = filter {
                warn!("No matching services found for {:?}", filter);
            }
            return Ok(());
        }

        if parallel {
            // Group services by dependency level for parallel startup
            self.start_services_parallel(services).await
        } else {
            // Sequential startup
            for service in services {
                self.start_service(service).await?;
            }
            Ok(())
        }
    }

    /// Start a single service
    pub async fn start_service(&self, service: &ServiceDefinition) -> Result<String> {
        let effective_service = self.apply_overrides(service);
        info!("Starting service: {}", effective_service.name);

        // Update state to starting
        self.update_service_status(&effective_service.name, ServiceStatus::Starting)
            .await;

        // Ports are already in the correct format (HashMap<String, Vec<PortBinding>>)
        let port_bindings = effective_service.ports.clone();

        // Start container (track Docker API duration)
        let docker_start = Instant::now();
        let container_manager = Arc::clone(&self.container_manager);
        let image = effective_service.image.clone();
        let environment = effective_service.environment.clone();
        let volumes = effective_service.volumes.clone();
        let network = effective_service.network.clone();
        let service_name = effective_service.name.clone();

        let start_result = retry_with_backoff(
            &format!("start_container_{}", service_name),
            RetryConfig::default(),
            || {
                let cm = container_manager.clone();
                let env_vars = environment.clone();
                let port_bindings = port_bindings.clone();
                let volumes = volumes.clone();
                let network = network.clone();
                let image = image.clone();
                let name = service_name.clone();
                async move {
                    cm.create_and_start(
                        &name,
                        &image,
                        env_vars,
                        port_bindings.clone(),
                        volumes.clone(),
                        network.as_deref(),
                    )
                    .await
                }
            },
        )
        .await;

        let container_id = match start_result {
            Ok(id) => {
                let duration = docker_start.elapsed().as_secs_f64();
                self.metrics
                    .docker_api_duration_seconds
                    .with_label_values(&["create_and_start", "success"])
                    .observe(duration);
                id
            }
            Err(e) => {
                let duration = docker_start.elapsed().as_secs_f64();
                self.metrics
                    .docker_api_duration_seconds
                    .with_label_values(&["create_and_start", "error"])
                    .observe(duration);

                let error_msg = format!("Failed to start {}: {}", service_name, e);
                error!("{}", error_msg);

                // Track failed start
                self.metrics
                    .failed_service_starts
                    .with_label_values(&[&service_name, "docker_error"])
                    .inc();

                self.update_service_status(&service_name, ServiceStatus::Failed(error_msg.clone()))
                    .await;
                return Err(e);
            }
        };

        // Update state with container ID
        {
            let mut state = self.state.write().await;
            if let Some(service_state) = state.get_mut(&service_name) {
                service_state.container_id = Some(container_id.clone());
                service_state.started_at = Some(chrono::Utc::now().timestamp());
            }
        }

        // Wait for health check
        if effective_service.health_check_url.is_some() || effective_service.name == "redis" {
            let health_start = Instant::now();
            match self
                .health_checker
                .wait_for_health(
                    &service_name,
                    effective_service.health_check_url.as_deref(),
                    self.config.health_check_max_attempts,
                    self.config.health_check_timeout_secs,
                )
                .await
            {
                Ok(_) => {
                    let health_duration = health_start.elapsed().as_secs_f64();
                    self.metrics
                        .service_health_check_duration_seconds
                        .with_label_values(&[&service_name, "success"])
                        .observe(health_duration);

                    info!("✓ Service {} is healthy", service_name);
                    self.update_service_status(&service_name, ServiceStatus::Running)
                        .await;
                }
                Err(e) => {
                    let health_duration = health_start.elapsed().as_secs_f64();
                    self.metrics
                        .service_health_check_duration_seconds
                        .with_label_values(&[&service_name, "failed"])
                        .observe(health_duration);

                    self.metrics
                        .failed_service_starts
                        .with_label_values(&[&service_name, "health_check_failed"])
                        .inc();

                    warn!("Service {} health check failed: {}", service_name, e);
                    self.update_service_status(&service_name, ServiceStatus::Unhealthy)
                        .await;

                    if effective_service.required {
                        self.metrics
                            .service_start_count
                            .with_label_values(&[&service_name, "failed"])
                            .inc();
                        return Err(e);
                    }
                }
            }
        } else {
            // No health check, assume running
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            self.update_service_status(&service_name, ServiceStatus::Running)
                .await;
        }

        // Track successful start
        self.metrics
            .service_start_count
            .with_label_values(&[&service_name, "success"])
            .inc();

        self.update_active_services_gauge().await;

        info!("✓ Service {} started successfully", service_name);
        Ok(container_id)
    }

    /// Stop a service
    pub async fn stop_service(&self, service_name: &str, timeout: Option<i64>) -> Result<()> {
        info!("Stopping service: {}", service_name);

        let container_id = {
            let state = self.state.read().await;
            state
                .get(service_name)
                .and_then(|s| s.container_id.clone())
                .ok_or_else(|| {
                    anyhow::anyhow!("Service {} not found or not started", service_name)
                })?
        };

        let result = self.container_manager.stop(&container_id, timeout).await;

        match result {
            Ok(_) => {
                self.metrics
                    .service_stop_count
                    .with_label_values(&[service_name, "success"])
                    .inc();
                {
                    let mut state = self.state.write().await;
                    if let Some(service_state) = state.get_mut(service_name) {
                        service_state.status = ServiceStatus::Stopped;
                        service_state.container_id = None;
                        service_state.started_at = None;
                    }
                }
                self.update_active_services_gauge().await;
                Ok(())
            }
            Err(e) => {
                self.metrics
                    .service_stop_count
                    .with_label_values(&[service_name, "error"])
                    .inc();
                Err(e)
            }
        }
    }

    /// Stop all services
    pub async fn stop_all_services(&self, timeout: Option<i64>) -> Result<()> {
        self.stop_services(&[], timeout).await
    }

    /// Stop selected services. Empty list => all services.
    pub async fn stop_services(
        &self,
        service_names: &[String],
        timeout: Option<i64>,
    ) -> Result<()> {
        info!("Stopping all services");

        let filter: Option<HashSet<String>> = if service_names.is_empty() {
            None
        } else {
            Some(service_names.iter().map(|s| s.to_string()).collect())
        };

        let services: Vec<String> = {
            let state = self.state.read().await;
            state
                .keys()
                .filter(|name| {
                    filter
                        .as_ref()
                        .map(|set| set.contains(*name))
                        .unwrap_or(true)
                })
                .cloned()
                .collect()
        };

        for service_name in services {
            if let Err(e) = self.stop_service(&service_name, timeout).await {
                warn!("Failed to stop {}: {}", service_name, e);
            }
        }

        Ok(())
    }

    /// Get status of all services
    pub async fn get_services_status(&self) -> HashMap<String, ServiceState> {
        let status = self.state.read().await.clone();

        // Update active services gauge
        let active_count = status
            .values()
            .filter(|s| s.status == ServiceStatus::Running)
            .count();
        self.metrics.active_services.set(active_count as f64);

        status
    }

    /// Get status of a specific service
    #[allow(dead_code)]
    pub async fn get_service_status(&self, service_name: &str) -> Option<ServiceState> {
        self.state.read().await.get(service_name).cloned()
    }

    async fn update_service_status(&self, service_name: &str, status: ServiceStatus) {
        let mut state = self.state.write().await;
        if let Some(service_state) = state.get_mut(service_name) {
            service_state.status = status;
        }
    }

    async fn start_services_parallel(&self, services: Vec<&ServiceDefinition>) -> Result<()> {
        use tokio::task::JoinSet;

        let semaphore = if self.config.max_concurrent_starts == 0 {
            None
        } else {
            Some(Arc::new(Semaphore::new(self.config.max_concurrent_starts)))
        };

        // Group by depth (number of dependencies)
        let mut depth_groups: HashMap<usize, Vec<&ServiceDefinition>> = HashMap::new();
        for service in services {
            let depth = self.calculate_depth(service, 0);
            depth_groups
                .entry(depth)
                .or_insert_with(Vec::new)
                .push(service);
        }

        // Start each depth group in parallel
        let mut depths: Vec<_> = depth_groups.keys().cloned().collect();
        depths.sort();

        for depth in depths {
            if let Some(group) = depth_groups.get(&depth) {
                info!(
                    "Starting services at depth {}: {:?}",
                    depth,
                    group.iter().map(|s| &s.name).collect::<Vec<_>>()
                );

                let mut join_set = JoinSet::new();

                for service in group {
                    let service_clone = (*service).clone();
                    let manager = self.clone();
                    let sem = semaphore.clone();

                    join_set.spawn(async move {
                        let _permit = if let Some(sem) = sem {
                            Some(
                                sem.acquire_owned()
                                    .await
                                    .context("failed to acquire concurrency slot")?,
                            )
                        } else {
                            None
                        };
                        manager.start_service(&service_clone).await
                    });
                }

                // Wait for all services in this group to start
                while let Some(result) = join_set.join_next().await {
                    match result {
                        Ok(Ok(_)) => {}
                        Ok(Err(e)) => {
                            error!("Service failed to start: {}", e);
                            // Continue with other services
                        }
                        Err(e) => {
                            error!("Task join error: {}", e);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn calculate_depth(&self, service: &ServiceDefinition, current_depth: usize) -> usize {
        if service.depends_on.is_empty() {
            return current_depth;
        }

        let mut max_depth = current_depth;
        for dep_name in &service.depends_on {
            if let Some(dep_service) = self.registry.get(dep_name) {
                let depth = self.calculate_depth(dep_service, current_depth + 1);
                max_depth = max_depth.max(depth);
            }
        }
        max_depth
    }

    /// Get service definition by name
    pub fn get_service_definition(&self, name: &str) -> Option<&ServiceDefinition> {
        self.registry.get(name)
    }

    /// Stream logs from a container
    pub async fn stream_container_logs(
        &self,
        container_id: &str,
        follow: bool,
        tail: i32,
        since_timestamp: i64,
    ) -> Result<impl futures_util::Stream<Item = Result<(String, String, i64)>>> {
        use bollard::container::LogsOptions;
        use futures_util::StreamExt;

        let opts = LogsOptions::<String> {
            follow,
            stdout: true,
            stderr: true,
            tail: if tail > 0 {
                tail.to_string()
            } else {
                "all".to_string()
            },
            since: since_timestamp,
            ..Default::default()
        };

        let stream = self.container_manager.docker.logs(container_id, Some(opts));

        // Transform the stream to extract log messages
        Ok(stream.map(move |log_result| {
            log_result
                .map(|log_output| {
                    use bollard::container::LogOutput;
                    let timestamp = chrono::Utc::now().timestamp();
                    match log_output {
                        LogOutput::StdOut { message } => (
                            "stdout".to_string(),
                            String::from_utf8_lossy(&message).to_string(),
                            timestamp,
                        ),
                        LogOutput::StdErr { message } => (
                            "stderr".to_string(),
                            String::from_utf8_lossy(&message).to_string(),
                            timestamp,
                        ),
                        LogOutput::StdIn { message } => (
                            "stdin".to_string(),
                            String::from_utf8_lossy(&message).to_string(),
                            timestamp,
                        ),
                        LogOutput::Console { message } => (
                            "console".to_string(),
                            String::from_utf8_lossy(&message).to_string(),
                            timestamp,
                        ),
                    }
                })
                .map_err(|e| anyhow::anyhow!("Log error: {}", e))
        }))
    }

    fn apply_overrides(&self, service: &ServiceDefinition) -> ServiceDefinition {
        let mut effective = service.clone();
        if let Some(override_cfg) = self.config.get_service_override(&service.name) {
            if let Some(required) = override_cfg.required {
                effective.required = required;
            }
            if let Some(url) = &override_cfg.health_check_url {
                effective.health_check_url = Some(url.clone());
            }
            if let Some(env_overrides) = &override_cfg.environment {
                effective.environment.extend(env_overrides.iter().cloned());
            }
        }
        effective
    }

    async fn update_active_services_gauge(&self) {
        let state = self.state.read().await;
        let active = state
            .values()
            .filter(|svc| svc.status == ServiceStatus::Running)
            .count();
        self.metrics.active_services.set(active as f64);
    }
}

impl Clone for EnvManager {
    fn clone(&self) -> Self {
        Self {
            container_manager: Arc::clone(&self.container_manager),
            health_checker: Arc::clone(&self.health_checker),
            registry: Arc::clone(&self.registry),
            state: Arc::clone(&self.state),
            metrics: Arc::clone(&self.metrics),
            config: Arc::clone(&self.config),
        }
    }
}
