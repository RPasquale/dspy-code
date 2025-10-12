mod container;
mod health;
mod manager;
mod service_registry;
mod grpc_server;
mod pb {
    tonic::include_proto!("env_manager.v1");
}

use anyhow::Result;
use std::env;
use tracing::{info, error};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        .init();

    info!("Starting DSPy Environment Manager v{}", env!("CARGO_PKG_VERSION"));

    // Get configuration from environment
    let grpc_addr = env::var("ENV_MANAGER_GRPC_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:50100".to_string());
    
    let docker_host = env::var("DOCKER_HOST")
        .ok();

    info!("Configuration:");
    info!("  gRPC Address: {}", grpc_addr);
    info!("  Docker Host: {}", docker_host.as_deref().unwrap_or("default"));

    // Initialize Docker client
    let docker = match docker_host {
        Some(host) => {
            info!("Connecting to Docker at {}", host);
            bollard::Docker::connect_with_socket(&host, 120, bollard::API_DEFAULT_VERSION)?
        }
        None => {
            info!("Connecting to default Docker socket");
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

    // Initialize manager
    let manager = manager::EnvManager::new(docker).await?;
    
    // Start gRPC server
    let grpc_server = grpc_server::GrpcServer::new(manager);
    
    info!("Starting gRPC server on {}", grpc_addr);
    grpc_server.serve(grpc_addr.parse()?).await?;

    Ok(())
}

