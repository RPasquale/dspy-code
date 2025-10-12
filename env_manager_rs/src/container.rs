use anyhow::{Context, Result};
use bollard::Docker;
use bollard::container::{
    Config, CreateContainerOptions, StartContainerOptions, StopContainerOptions,
    RemoveContainerOptions, ListContainersOptions, Stats, StatsOptions,
};
use bollard::models::{ContainerStateStatusEnum, HostConfig, PortBinding};
use futures_util::StreamExt;
use std::collections::HashMap;
use std::default::Default;
use tracing::{info, error};

pub struct ContainerManager {
    docker: Docker,
}

impl ContainerManager {
    pub fn new(docker: Docker) -> Self {
        Self { docker }
    }

    /// Create and start a container with the given configuration
    pub async fn create_and_start(
        &self,
        name: &str,
        image: &str,
        env_vars: Vec<String>,
        ports: HashMap<String, Vec<PortBinding>>,
        volumes: Vec<String>,
        network: Option<&str>,
    ) -> Result<String> {
        info!("Creating container: {}", name);

        // Check if container already exists
        if let Ok(Some(container_id)) = self.get_container_id(name).await {
            info!("Container {} already exists with ID: {}", name, container_id);
            
            // Check if it's running
            if self.is_running(&container_id).await? {
                info!("Container {} is already running", name);
                return Ok(container_id);
            }
            
            // Start existing container
            info!("Starting existing container {}", name);
            self.start(&container_id).await?;
            return Ok(container_id);
        }

        // Pull image if not present
        self.ensure_image(image).await?;

        // Create container configuration
        // Convert ports HashMap to the correct type expected by Bollard
        let port_bindings: HashMap<String, Option<Vec<PortBinding>>> = ports
            .into_iter()
            .map(|(k, v)| (k, Some(v)))
            .collect();
        
        let host_config = HostConfig {
            port_bindings: Some(port_bindings),
            binds: Some(volumes),
            network_mode: network.map(|n| n.to_string()),
            ..Default::default()
        };

        let config = Config {
            image: Some(image.to_string()),
            env: Some(env_vars),
            host_config: Some(host_config),
            ..Default::default()
        };

        // Create container
        let create_options = CreateContainerOptions { name, ..Default::default() };
        let container = self.docker
            .create_container(Some(create_options), config)
            .await
            .with_context(|| format!("Failed to create container {}", name))?;

        let container_id = container.id;
        info!("Container {} created with ID: {}", name, container_id);

        // Start container
        self.start(&container_id).await?;

        Ok(container_id)
    }

    /// Start a container
    pub async fn start(&self, container_id: &str) -> Result<()> {
        info!("Starting container: {}", container_id);
        self.docker
            .start_container(container_id, None::<StartContainerOptions<String>>)
            .await
            .with_context(|| format!("Failed to start container {}", container_id))?;
        info!("Container {} started", container_id);
        Ok(())
    }

    /// Stop a container gracefully
    pub async fn stop(&self, container_id: &str, timeout: Option<i64>) -> Result<()> {
        info!("Stopping container: {}", container_id);
        let options = StopContainerOptions {
            t: timeout.unwrap_or(10) as i64,
        };
        self.docker
            .stop_container(container_id, Some(options))
            .await
            .with_context(|| format!("Failed to stop container {}", container_id))?;
        info!("Container {} stopped", container_id);
        Ok(())
    }

    /// Remove a container
    #[allow(dead_code)]
    pub async fn remove(&self, container_id: &str, force: bool) -> Result<()> {
        info!("Removing container: {}", container_id);
        let options = RemoveContainerOptions {
            force,
            v: true, // Remove volumes
            ..Default::default()
        };
        self.docker
            .remove_container(container_id, Some(options))
            .await
            .with_context(|| format!("Failed to remove container {}", container_id))?;
        info!("Container {} removed", container_id);
        Ok(())
    }

    /// Check if a container is running
    pub async fn is_running(&self, container_id: &str) -> Result<bool> {
        let inspect = self.docker
            .inspect_container(container_id, None)
            .await
            .with_context(|| format!("Failed to inspect container {}", container_id))?;

        Ok(inspect
            .state
            .and_then(|s| s.status)
            .map(|status| status == ContainerStateStatusEnum::RUNNING)
            .unwrap_or(false))
    }

    /// Get container ID by name
    pub async fn get_container_id(&self, name: &str) -> Result<Option<String>> {
        let mut filters = HashMap::new();
        filters.insert("name".to_string(), vec![name.to_string()]);

        let options = ListContainersOptions {
            all: true,
            filters,
            ..Default::default()
        };

        let containers = self.docker.list_containers(Some(options)).await?;

        Ok(containers.first().map(|c| c.id.clone().unwrap_or_default()))
    }

    /// Get container stats (CPU, memory usage)
    #[allow(dead_code)]
    pub async fn get_stats(&self, container_id: &str) -> Result<Stats> {
        let options = StatsOptions {
            stream: false,
            one_shot: true,
        };

        let mut stats_stream = self.docker.stats(container_id, Some(options));
        
        if let Some(stats_result) = stats_stream.next().await {
            return stats_result.with_context(|| format!("Failed to get stats for {}", container_id));
        }

        Err(anyhow::anyhow!("No stats available for container {}", container_id))
    }

    /// Ensure image is pulled
    async fn ensure_image(&self, image: &str) -> Result<()> {
        use bollard::image::ListImagesOptions;
        
        let mut filters = HashMap::new();
        filters.insert("reference".to_string(), vec![image.to_string()]);
        
        let options = ListImagesOptions {
            filters,
            ..Default::default()
        };

        let images = self.docker.list_images(Some(options)).await?;

        if images.is_empty() {
            info!("Pulling image: {}", image);
            self.pull_image(image).await?;
        } else {
            info!("Image {} already present", image);
        }

        Ok(())
    }

    /// Pull a Docker image
    pub async fn pull_image(&self, image: &str) -> Result<()> {
        use bollard::image::CreateImageOptions;

        let options = Some(CreateImageOptions {
            from_image: image,
            ..Default::default()
        });

        let mut stream = self.docker.create_image(options, None, None);

        while let Some(result) = stream.next().await {
            match result {
                Ok(info) => {
                    if let Some(status) = info.status {
                        info!("Image pull: {}", status);
                    }
                }
                Err(e) => {
                    error!("Error pulling image: {}", e);
                    return Err(e.into());
                }
            }
        }

        info!("Successfully pulled image: {}", image);
        Ok(())
    }
}

