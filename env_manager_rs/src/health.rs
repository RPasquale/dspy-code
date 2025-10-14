use anyhow::Result;
use reqwest::Client;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn};

pub struct HealthChecker {
    client: Client,
}

impl HealthChecker {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();

        Self { client }
    }

    /// Check HTTP health endpoint
    pub async fn check_http(&self, url: &str) -> bool {
        match self.client.get(url).send().await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }

    /// Check Redis health using PING command
    pub async fn check_redis(&self, host: &str, port: u16) -> bool {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpStream;

        let addr = format!("{}:{}", host, port);
        match TcpStream::connect(&addr).await {
            Ok(mut stream) => {
                // Send PING command
                if stream.write_all(b"*1\r\n$4\r\nPING\r\n").await.is_err() {
                    return false;
                }

                // Read response
                let mut buf = [0u8; 128];
                match stream.read(&mut buf).await {
                    Ok(_) => {
                        // Check for PONG response
                        let response = String::from_utf8_lossy(&buf);
                        response.contains("PONG")
                    }
                    Err(_) => false,
                }
            }
            Err(_) => false,
        }
    }

    /// Wait for service to become healthy with exponential backoff
    pub async fn wait_for_health(
        &self,
        service_name: &str,
        check_url: Option<&str>,
        max_attempts: u32,
        timeout_secs: u64,
    ) -> Result<()> {
        info!("Waiting for {} to become healthy", service_name);

        let mut attempt = 0;
        let mut delay = Duration::from_millis(500);
        let deadline = if timeout_secs == 0 {
            None
        } else {
            Some(std::time::Instant::now() + Duration::from_secs(timeout_secs))
        };

        while attempt < max_attempts {
            attempt += 1;

            let healthy = if let Some(url) = check_url {
                // Special handling for Redis
                if service_name == "redis" {
                    self.check_redis("localhost", 6379).await
                } else {
                    self.check_http(url).await
                }
            } else {
                // If no health check URL, assume healthy after a few attempts
                attempt > 3
            };

            if healthy {
                info!("✓ {} is healthy", service_name);
                return Ok(());
            }

            if let Some(deadline) = deadline {
                if std::time::Instant::now() >= deadline {
                    break;
                }
            }

            warn!(
                "Health check failed for {} (attempt {}/{}), retrying in {:?}",
                service_name, attempt, max_attempts, delay
            );

            sleep(delay).await;

            // Exponential backoff, max 10 seconds
            delay = std::cmp::min(delay * 2, Duration::from_secs(10));
        }

        Err(anyhow::anyhow!(
            "Service {} failed to become healthy after {} attempts",
            service_name,
            max_attempts
        ))
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}
