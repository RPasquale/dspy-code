mod config;
mod container;
mod grpc_server;
mod health;
mod http_server;
mod manager;
mod metrics;
mod retry;
mod service_registry;
mod shutdown;
mod pb {
    tonic::include_proto!("env_manager.v1");
}

use anyhow::Result;
use std::sync::Arc;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        .with_line_number(true)
        .init();

    info!(
        "🚀 Starting DSPy Environment Manager v{}",
        env!("CARGO_PKG_VERSION")
    );

    // Load configuration
    let config = Arc::new(match config::Config::load() {
        Ok(cfg) => cfg,
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            return Err(e);
        }
    });

    info!("📋 Configuration:");
    info!("  gRPC Address: {}", config.grpc_addr);
    info!("  HTTP Metrics Address: {}", config.metrics_http_addr);
    info!(
        "  Docker Host: {}",
        config.docker_host.as_deref().unwrap_or("default")
    );
    info!("  Max Concurrent Starts: {}", config.max_concurrent_starts);
    info!(
        "  Health Check Timeout: {}s",
        config.health_check_timeout_secs
    );
    info!("  Verbose Logging: {}", config.verbose_logging);

    // Initialize Docker client
    let docker = match &config.docker_host {
        Some(host) => {
            info!("🐳 Connecting to Docker at {}", host);
            bollard::Docker::connect_with_socket(&host, 120, bollard::API_DEFAULT_VERSION)?
        }
        None => {
            info!("🐳 Connecting to default Docker socket");
            bollard::Docker::connect_with_local_defaults()?
        }
    };

    // Test Docker connection
    match docker.ping().await {
        Ok(_) => info!("✓ Docker connection successful"),
        Err(e) => {
            error!("✗ Failed to connect to Docker: {}", e);
            error!("Please ensure Docker is running and accessible");
            return Err(e.into());
        }
    }

    // Initialize shutdown coordinator
    let shutdown = Arc::new(shutdown::ShutdownCoordinator::new());
    let shutdown_signal = shutdown.clone();

    // Spawn signal handler task
    tokio::spawn(async move {
        if let Err(e) = shutdown_signal.wait_for_signal().await {
            error!("Error waiting for shutdown signal: {}", e);
        }
    });

    // Initialize metrics registry
    let metrics = Arc::new(metrics::MetricsRegistry::new()?);
    info!("📊 Metrics registry initialized");

    // Initialize manager with metrics and configuration
    let manager =
        Arc::new(manager::EnvManager::new(docker, metrics.clone(), config.clone()).await?);

    // Start gRPC server
    let grpc_server = grpc_server::GrpcServer::new(manager.clone());
    let server_addr = config.grpc_addr.parse()?;

    info!("🌐 Starting gRPC server on {}", config.grpc_addr);

    let grpc_handle = tokio::spawn(async move { grpc_server.serve(server_addr).await });

    // Start HTTP metrics server
    let http_server =
        http_server::HttpMetricsServer::new(metrics.clone(), config.metrics_http_addr.parse()?);

    let http_handle = tokio::spawn(async move {
        if let Err(e) = http_server.serve().await {
            error!("HTTP metrics server error: {}", e);
        }
    });

    // Wait for shutdown signal
    while !shutdown.is_shutdown_requested() {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    info!("🛑 Shutdown signal received, cleaning up...");

    // Perform graceful shutdown
    let manager_cleanup = manager.clone();
    shutdown
        .graceful_shutdown(
            || async move {
                info!("Stopping all services...");
                if let Err(e) = manager_cleanup.stop_all_services(Some(10)).await {
                    error!("Error stopping services: {}", e);
                }
                Ok(())
            },
            30, // 30 second timeout
        )
        .await?;

    // Wait for servers to finish (they should be cancelled by now)
    let _ = tokio::time::timeout(tokio::time::Duration::from_secs(5), grpc_handle).await;

    let _ = tokio::time::timeout(tokio::time::Duration::from_secs(5), http_handle).await;

    info!("✓ Shutdown complete");
    Ok(())
}
