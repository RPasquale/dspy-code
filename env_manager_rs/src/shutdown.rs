use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::signal;
use tracing::{info, warn};

/// Shutdown coordinator for graceful termination
#[derive(Clone)]
pub struct ShutdownCoordinator {
    shutdown_flag: Arc<AtomicBool>,
}

impl ShutdownCoordinator {
    pub fn new() -> Self {
        Self {
            shutdown_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Check if shutdown has been requested
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_flag.load(Ordering::Relaxed)
    }

    /// Request shutdown
    pub fn request_shutdown(&self) {
        info!("Shutdown requested");
        self.shutdown_flag.store(true, Ordering::Relaxed);
    }

    /// Wait for shutdown signal (SIGTERM or SIGINT)
    pub async fn wait_for_signal(&self) -> Result<()> {
        #[cfg(unix)]
        {
            let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())?;
            let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())?;

            tokio::select! {
                _ = sigterm.recv() => {
                    info!("Received SIGTERM signal");
                }
                _ = sigint.recv() => {
                    info!("Received SIGINT signal");
                }
            }
        }

        #[cfg(not(unix))]
        {
            signal::ctrl_c().await?;
            info!("Received Ctrl+C signal");
        }

        self.request_shutdown();
        Ok(())
    }

    /// Perform graceful shutdown with timeout
    pub async fn graceful_shutdown<F, Fut>(&self, cleanup: F, timeout_secs: u64) -> Result<()>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        info!("🛑 Starting graceful shutdown (timeout: {}s)", timeout_secs);

        let cleanup_result =
            tokio::time::timeout(tokio::time::Duration::from_secs(timeout_secs), cleanup()).await;

        match cleanup_result {
            Ok(Ok(())) => {
                info!("✓ Cleanup completed successfully");
                Ok(())
            }
            Ok(Err(e)) => {
                warn!("⚠ Cleanup completed with errors: {}", e);
                Err(e)
            }
            Err(_) => {
                warn!("⚠ Cleanup timed out after {}s", timeout_secs);
                Err(anyhow::anyhow!("Shutdown timeout"))
            }
        }
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_shutdown_flag() {
        let coordinator = ShutdownCoordinator::new();
        assert!(!coordinator.is_shutdown_requested());

        coordinator.request_shutdown();
        assert!(coordinator.is_shutdown_requested());
    }

    #[tokio::test]
    async fn test_graceful_shutdown_success() {
        let coordinator = ShutdownCoordinator::new();

        let result = coordinator.graceful_shutdown(|| async { Ok(()) }, 5).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_graceful_shutdown_timeout() {
        let coordinator = ShutdownCoordinator::new();

        let result = coordinator
            .graceful_shutdown(
                || async {
                    tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
                    Ok(())
                },
                1, // 1 second timeout
            )
            .await;

        assert!(result.is_err());
    }
}
