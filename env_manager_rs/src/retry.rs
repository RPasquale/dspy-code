use anyhow::Result;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 500,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Retry a function with exponential backoff
pub async fn retry_with_backoff<F, Fut, T>(
    operation_name: &str,
    config: RetryConfig,
    mut f: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut attempt = 0;
    let mut delay_ms = config.initial_delay_ms;

    loop {
        attempt += 1;

        match f().await {
            Ok(result) => {
                if attempt > 1 {
                    debug!(
                        "{} succeeded on attempt {}/{}",
                        operation_name, attempt, config.max_attempts
                    );
                }
                return Ok(result);
            }
            Err(e) => {
                if attempt >= config.max_attempts {
                    warn!(
                        "{} failed after {} attempts: {}",
                        operation_name, attempt, e
                    );
                    return Err(e);
                }

                warn!(
                    "{} failed on attempt {}/{}: {}, retrying in {}ms",
                    operation_name, attempt, config.max_attempts, e, delay_ms
                );

                sleep(Duration::from_millis(delay_ms)).await;

                // Exponential backoff
                delay_ms = ((delay_ms as f64) * config.backoff_multiplier) as u64;
                delay_ms = delay_ms.min(config.max_delay_ms);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_success_first_attempt() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay_ms: 10,
            max_delay_ms: 1000,
            backoff_multiplier: 2.0,
        };

        let result =
            retry_with_backoff("test", config, || async { Ok::<i32, anyhow::Error>(42) }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let config = RetryConfig {
            max_attempts: 3,
            initial_delay_ms: 10,
            max_delay_ms: 1000,
            backoff_multiplier: 2.0,
        };

        let result = retry_with_backoff("test", config, move || {
            let attempts = attempts_clone.clone();
            async move {
                let count = attempts.fetch_add(1, Ordering::SeqCst);
                if count < 2 {
                    Err(anyhow::anyhow!("attempt {}", count))
                } else {
                    Ok::<i32, anyhow::Error>(42)
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_max_attempts_exceeded() {
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let config = RetryConfig {
            max_attempts: 3,
            initial_delay_ms: 10,
            max_delay_ms: 1000,
            backoff_multiplier: 2.0,
        };

        let result = retry_with_backoff("test", config, move || {
            let attempts = attempts_clone.clone();
            async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err::<i32, anyhow::Error>(anyhow::anyhow!("always fails"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }
}
