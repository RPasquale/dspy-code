use crate::container::ContainerManager;
use crate::health::HealthChecker;
use crate::service_registry::{ServiceDefinition, ServiceRegistry};
use anyhow::Result;
use bollard::Docker;
// PortBinding is now in service_registry
// use bollard::models::PortBinding;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, error, warn};

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
}

impl EnvManager {
    pub async fn new(docker: Docker) -> Result<Self> {
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
        })
    }

    /// Start all services in dependency order
    pub async fn start_all_services(&self, parallel: bool) -> Result<()> {
        info!("Starting all services (parallel: {})", parallel);

        let services = self.registry.get_startup_order();

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
        info!("Starting service: {}", service.name);

        // Update state to starting
        self.update_service_status(&service.name, ServiceStatus::Starting)
            .await;

        // Ports are already in the correct format (HashMap<String, Vec<PortBinding>>)
        let port_bindings = service.ports.clone();

        // Start container
        let container_id = match self
            .container_manager
            .create_and_start(
                &service.name,
                &service.image,
                service.environment.clone(),
                port_bindings,
                service.volumes.clone(),
                service.network.as_deref(),
            )
            .await
        {
            Ok(id) => id,
            Err(e) => {
                let error_msg = format!("Failed to start {}: {}", service.name, e);
                error!("{}", error_msg);
                self.update_service_status(&service.name, ServiceStatus::Failed(error_msg.clone()))
                    .await;
                return Err(e);
            }
        };

        // Update state with container ID
        {
            let mut state = self.state.write().await;
            if let Some(service_state) = state.get_mut(&service.name) {
                service_state.container_id = Some(container_id.clone());
                service_state.started_at = Some(chrono::Utc::now().timestamp());
            }
        }

        // Wait for health check
        if service.health_check_url.is_some() || service.name == "redis" {
            match self
                .health_checker
                .wait_for_health(
                    &service.name,
                    service.health_check_url.as_deref(),
                    30, // max attempts
                )
                .await
            {
                Ok(_) => {
                    info!("✓ Service {} is healthy", service.name);
                    self.update_service_status(&service.name, ServiceStatus::Running)
                        .await;
                }
                Err(e) => {
                    warn!("Service {} health check failed: {}", service.name, e);
                    self.update_service_status(&service.name, ServiceStatus::Unhealthy)
                        .await;
                    
                    if service.required {
                        return Err(e);
                    }
                }
            }
        } else {
            // No health check, assume running
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            self.update_service_status(&service.name, ServiceStatus::Running)
                .await;
        }

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
                .ok_or_else(|| anyhow::anyhow!("Service {} not found or not started", service_name))?
        };

        self.container_manager.stop(&container_id, timeout).await?;
        self.update_service_status(service_name, ServiceStatus::Stopped)
            .await;

        Ok(())
    }

    /// Stop all services
    pub async fn stop_all_services(&self, timeout: Option<i64>) -> Result<()> {
        info!("Stopping all services");

        let services: Vec<String> = {
            let state = self.state.read().await;
            state.keys().cloned().collect()
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
        self.state.read().await.clone()
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

        // Group by depth (number of dependencies)
        let mut depth_groups: HashMap<usize, Vec<&ServiceDefinition>> = HashMap::new();
        for service in services {
            let depth = self.calculate_depth(service, 0);
            depth_groups.entry(depth).or_insert_with(Vec::new).push(service);
        }

        // Start each depth group in parallel
        let mut depths: Vec<_> = depth_groups.keys().cloned().collect();
        depths.sort();

        for depth in depths {
            if let Some(group) = depth_groups.get(&depth) {
                info!("Starting services at depth {}: {:?}", depth, group.iter().map(|s| &s.name).collect::<Vec<_>>());

                let mut join_set = JoinSet::new();

                for service in group {
                    let service_clone = (*service).clone();
                    let manager = self.clone();

                    join_set.spawn(async move {
                        manager.start_service(&service_clone).await
                    });
                }

                // Wait for all services in this group to start
                while let Some(result) = join_set.join_next().await {
                    match result {
                        Ok(Ok(_)) => {},
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

    pub fn clone(&self) -> Self {
        Self {
            container_manager: Arc::clone(&self.container_manager),
            health_checker: Arc::clone(&self.health_checker),
            registry: Arc::clone(&self.registry),
            state: Arc::clone(&self.state),
        }
    }

    /// Get service definition by name
    pub fn get_service_definition(&self, name: &str) -> Option<&ServiceDefinition> {
        self.registry.get(name)
    }
}

