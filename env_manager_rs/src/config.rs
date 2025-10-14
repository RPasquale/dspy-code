use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;

/// Configuration for the environment manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// gRPC server address
    pub grpc_addr: String,

    /// HTTP metrics server address
    pub metrics_http_addr: String,

    /// Docker host URL
    pub docker_host: Option<String>,

    /// Maximum concurrent service starts
    pub max_concurrent_starts: usize,

    /// Health check timeout (seconds)
    pub health_check_timeout_secs: u64,

    /// Health check retry attempts
    pub health_check_max_attempts: u32,

    /// Enable detailed logging
    pub verbose_logging: bool,

    /// Service-specific overrides
    pub service_overrides: Vec<ServiceOverride>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceOverride {
    pub name: String,
    pub required: Option<bool>,
    pub health_check_url: Option<String>,
    pub environment: Option<Vec<String>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            grpc_addr: "0.0.0.0:50100".to_string(),
            metrics_http_addr: "0.0.0.0:50101".to_string(),
            docker_host: None,
            max_concurrent_starts: 5,
            health_check_timeout_secs: 60,
            health_check_max_attempts: 30,
            verbose_logging: false,
            service_overrides: vec![],
        }
    }
}

impl Config {
    /// Load configuration from environment variables and optional config file
    pub fn load() -> Result<Self> {
        let mut config = Self::default();

        // Load from config file if exists
        if let Ok(config_path) = env::var("ENV_MANAGER_CONFIG") {
            if Path::new(&config_path).exists() {
                let contents = fs::read_to_string(&config_path)?;
                config = toml::from_str(&contents)?;
            }
        }

        // Override with environment variables
        if let Ok(addr) = env::var("ENV_MANAGER_GRPC_ADDR") {
            config.grpc_addr = addr;
        }

        if let Ok(addr) = env::var("ENV_MANAGER_METRICS_ADDR") {
            config.metrics_http_addr = addr;
        }

        if let Ok(docker_host) = env::var("DOCKER_HOST") {
            config.docker_host = Some(docker_host);
        }

        if let Ok(max_concurrent) = env::var("ENV_MANAGER_MAX_CONCURRENT") {
            if let Ok(n) = max_concurrent.parse() {
                config.max_concurrent_starts = n;
            }
        }

        if let Ok(timeout) = env::var("ENV_MANAGER_HEALTH_TIMEOUT") {
            if let Ok(n) = timeout.parse() {
                config.health_check_timeout_secs = n;
            }
        }

        if let Ok(verbose) = env::var("ENV_MANAGER_VERBOSE") {
            config.verbose_logging = verbose == "1" || verbose.to_lowercase() == "true";
        }

        Ok(config)
    }

    /// Get service override if it exists
    pub fn get_service_override(&self, service_name: &str) -> Option<&ServiceOverride> {
        self.service_overrides
            .iter()
            .find(|o| o.name == service_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.grpc_addr, "0.0.0.0:50100");
        assert_eq!(config.max_concurrent_starts, 5);
    }

    #[test]
    fn test_load_from_env() {
        env::set_var("ENV_MANAGER_GRPC_ADDR", "127.0.0.1:9999");
        env::set_var("ENV_MANAGER_VERBOSE", "true");

        let config = Config::load().unwrap();
        assert_eq!(config.grpc_addr, "127.0.0.1:9999");
        assert!(config.verbose_logging);

        env::remove_var("ENV_MANAGER_GRPC_ADDR");
        env::remove_var("ENV_MANAGER_VERBOSE");
    }
}
