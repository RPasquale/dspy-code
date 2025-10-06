use std::sync::Arc;

use env_runner_rs::infermesh::{InferMeshClient, InferMeshConfig};
use env_runner_rs::metrics::{EnvRunnerMetrics, MetricsServer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let metrics = Arc::new(EnvRunnerMetrics::new());
    let infermesh_config = config_from_env();
    let client = Arc::new(InferMeshClient::new(infermesh_config)?);

    let server = MetricsServer::new(
        metrics.clone(),
        client.clone(),
        std::env::var("METRICS_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8083),
    );
    let server_task = tokio::spawn(async move {
        if let Err(err) = server.start().await {
            eprintln!("metrics server error: {err}");
        }
    });

    tokio::signal::ctrl_c().await?;
    server.shutdown().await;
    let _ = server_task.await;

    Ok(())
}

fn config_from_env() -> InferMeshConfig {
    let mut config = InferMeshConfig::default();
    if let Ok(base) = std::env::var("INFERMESH_BASE_URL") {
        config.base_url = base;
    }
    if let Ok(key) = std::env::var("INFERMESH_API_KEY") {
        if !key.is_empty() {
            config.api_key = Some(key);
        }
    }
    if let Ok(model) = std::env::var("INFERMESH_MODEL") {
        if !model.is_empty() {
            config.model = model;
        }
    }
    if let Ok(max) = std::env::var("MAX_CONCURRENT_REQUESTS") {
        if let Ok(val) = max.parse() {
            config.max_concurrent_requests = val;
        }
    }
    if let Ok(batch) = std::env::var("BATCH_SIZE") {
        if let Ok(val) = batch.parse() {
            config.batch_size = val;
        }
    }
    if let Ok(wait) = std::env::var("MAX_WAIT_TIME_MS") {
        if let Ok(val) = wait.parse() {
            config.max_wait_time_ms = val;
        }
    }
    if let Ok(timeout) = std::env::var("REQUEST_TIMEOUT_SECS") {
        if let Ok(val) = timeout.parse() {
            config.timeout_secs = val;
        }
    }
    if let Ok(pool) = std::env::var("CONNECTION_POOL_SIZE") {
        if let Ok(val) = pool.parse() {
            config.connection_pool_size = val;
        }
    }
    config
}
