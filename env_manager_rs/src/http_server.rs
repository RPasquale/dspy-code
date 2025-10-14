//! HTTP server for exposing metrics and health endpoints.

use crate::metrics::MetricsRegistry;
use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::info;

/// HTTP server for metrics and health endpoints.
pub struct HttpMetricsServer {
    metrics: Arc<MetricsRegistry>,
    addr: SocketAddr,
}

impl HttpMetricsServer {
    /// Create a new HTTP metrics server.
    pub fn new(metrics: Arc<MetricsRegistry>, addr: SocketAddr) -> Self {
        Self { metrics, addr }
    }

    /// Start serving HTTP endpoints.
    pub async fn serve(self) -> Result<()> {
        let app = Router::new()
            .route("/metrics", get(metrics_handler))
            .route("/health", get(health_handler))
            .with_state(self.metrics);

        info!("📊 HTTP metrics server listening on {}", self.addr);

        let listener = tokio::net::TcpListener::bind(self.addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

/// Handler for /metrics endpoint - returns Prometheus-formatted metrics.
async fn metrics_handler(
    State(metrics): State<Arc<MetricsRegistry>>,
) -> Result<Response, AppError> {
    let metrics_text = metrics.gather()?;
    Ok((
        StatusCode::OK,
        [("Content-Type", "text/plain; version=0.0.4")],
        metrics_text,
    )
        .into_response())
}

/// Handler for /health endpoint - returns JSON health status.
async fn health_handler() -> Json<serde_json::Value> {
    Json(json!({
        "status": "healthy",
        "service": "env-manager",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

/// Error wrapper for Axum handlers.
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Internal error: {}", self.0),
        )
            .into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
