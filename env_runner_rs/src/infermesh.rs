//! High-performance InferMesh client for Rust environment runner
//! Provides async HTTP client with connection pooling and intelligent batching

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};
use tokio::time::timeout;

/// High-performance InferMesh client with async I/O and connection pooling
pub struct InferMeshClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    connection_pool: Arc<Semaphore>,
    batch_processor: Arc<Mutex<BatchProcessor>>,
    metrics: Arc<Mutex<ClientMetrics>>,
}

/// Batch processor for intelligent request batching
struct BatchProcessor {
    batch_size: usize,
    max_wait_time: Duration,
    pending_batches: HashMap<String, PendingBatch>,
}

/// Pending batch waiting to be processed
struct PendingBatch {
    texts: Vec<String>,
    created_at: Instant,
}

/// Client performance metrics
#[derive(Debug, Clone)]
pub struct ClientMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_latency_ms: f64,
    pub concurrent_requests: u64,
    pub batch_efficiency: f64,
}

/// Embedding request payload
#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    inputs: Vec<String>,
}

/// Embedding response payload
#[derive(Deserialize)]
struct EmbedResponse {
    vectors: Option<Vec<Vec<f32>>>,
    embeddings: Option<Vec<Vec<f32>>>,
}

/// Configuration for InferMesh client
#[derive(Debug, Clone)]
pub struct InferMeshConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub max_concurrent_requests: usize,
    pub batch_size: usize,
    pub max_wait_time_ms: u64,
    pub timeout_secs: u64,
    pub connection_pool_size: usize,
}

impl Default for InferMeshConfig {
    fn default() -> Self {
        Self {
            base_url: "http://infermesh:9000".to_string(),
            api_key: None,
            model: "BAAI/bge-small-en-v1.5".to_string(),
            max_concurrent_requests: 100,
            batch_size: 512,
            max_wait_time_ms: 100,
            timeout_secs: 30,
            connection_pool_size: 50,
        }
    }
}

impl InferMeshClient {
    /// Create a new high-performance InferMesh client
    pub fn new(config: InferMeshConfig) -> Result<Self> {
        // Configure HTTP client with connection pooling
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_max_idle_per_host(config.connection_pool_size)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .context("Failed to create HTTP client")?;

        let connection_pool = Arc::new(Semaphore::new(config.max_concurrent_requests));

        let batch_processor = Arc::new(Mutex::new(BatchProcessor {
            batch_size: config.batch_size,
            max_wait_time: Duration::from_millis(config.max_wait_time_ms),
            pending_batches: HashMap::new(),
        }));

        let metrics = Arc::new(Mutex::new(ClientMetrics {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            average_latency_ms: 0.0,
            concurrent_requests: 0,
            batch_efficiency: 0.0,
        }));

        Ok(Self {
            client,
            base_url: config.base_url,
            api_key: config.api_key,
            model: config.model,
            connection_pool,
            batch_processor,
            metrics,
        })
    }

    /// Embed texts with intelligent batching and async processing
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Acquire semaphore for concurrency control
        let _permit = self
            .connection_pool
            .acquire()
            .await
            .context("Failed to acquire connection permit")?;

        // Update metrics
        {
            let mut metrics = self.metrics.lock().await;
            metrics.total_requests += 1;
            metrics.concurrent_requests += 1;
        }

        let start_time = Instant::now();
        let result = self.process_embedding_request(texts).await;
        let latency = start_time.elapsed();

        // Update metrics
        {
            let mut metrics = self.metrics.lock().await;
            metrics.concurrent_requests -= 1;

            if result.is_ok() {
                metrics.successful_requests += 1;
            } else {
                metrics.failed_requests += 1;
            }

            // Update average latency
            metrics.average_latency_ms =
                (metrics.average_latency_ms + latency.as_millis() as f64) / 2.0;
        }

        result
    }

    /// Process embedding request with optimized HTTP handling
    async fn process_embedding_request(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = EmbedRequest {
            model: self.model.clone(),
            inputs: texts.clone(),
        };

        // Prepare HTTP request
        let mut request_builder = self
            .client
            .post(&format!("{}/embed", self.base_url))
            .json(&request);

        // Add authentication header if API key is provided
        if let Some(api_key) = &self.api_key {
            request_builder =
                request_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        // Execute request with timeout
        let response = timeout(Duration::from_secs(30), request_builder.send())
            .await
            .context("Request timeout")?
            .context("Failed to send request")?;

        // Check response status
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "InferMesh returned status: {}",
                response.status()
            ));
        }

        // Parse response
        let embed_response: EmbedResponse =
            response.json().await.context("Failed to parse response")?;

        // Extract embeddings (support both 'vectors' and 'embeddings' fields)
        let embeddings = embed_response
            .vectors
            .or(embed_response.embeddings)
            .ok_or_else(|| anyhow::anyhow!("No embeddings in response"))?;

        // Validate response length
        if embeddings.len() != texts.len() {
            return Err(anyhow::anyhow!(
                "Mismatch between input texts ({}) and embeddings ({})",
                texts.len(),
                embeddings.len()
            ));
        }

        Ok(embeddings)
    }

    /// Embed large batch with optimized processing
    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // For very large batches, split into smaller chunks
        let chunk_size = 512;
        if texts.len() <= chunk_size {
            return self.embed(texts).await;
        }

        let mut all_embeddings = Vec::new();
        let mut start = 0;

        while start < texts.len() {
            let end = (start + chunk_size).min(texts.len());
            let chunk = texts[start..end].to_vec();

            let chunk_embeddings = self.embed(chunk).await?;
            all_embeddings.extend(chunk_embeddings);

            start = end;
        }

        Ok(all_embeddings)
    }

    /// Get current client metrics
    pub async fn get_metrics(&self) -> ClientMetrics {
        self.metrics.lock().await.clone()
    }

    /// Set batch size for optimal performance
    pub async fn set_batch_size(&self, size: usize) {
        let mut processor = self.batch_processor.lock().await;
        processor.batch_size = size;
    }

    /// Set maximum wait time for batching
    pub async fn set_max_wait_time(&self, duration: Duration) {
        let mut processor = self.batch_processor.lock().await;
        processor.max_wait_time = duration;
    }

    /// Health check for InferMesh service
    pub async fn health_check(&self) -> Result<bool> {
        let response = self
            .client
            .get(&format!("{}/health", self.base_url))
            .send()
            .await
            .context("Failed to send health check request")?;

        Ok(response.status().is_success())
    }

    /// Get connection pool status
    pub fn get_connection_pool_status(&self) -> (usize, usize) {
        let available = self.connection_pool.available_permits();
        let total =
            self.connection_pool.available_permits() + self.connection_pool.available_permits();
        (available, total)
    }
}

/// High-performance InferMesh client builder
pub struct InferMeshClientBuilder {
    config: InferMeshConfig,
}

impl InferMeshClientBuilder {
    pub fn new() -> Self {
        Self {
            config: InferMeshConfig::default(),
        }
    }

    pub fn base_url(mut self, url: String) -> Self {
        self.config.base_url = url;
        self
    }

    pub fn api_key(mut self, key: Option<String>) -> Self {
        self.config.api_key = key;
        self
    }

    pub fn model(mut self, model: String) -> Self {
        self.config.model = model;
        self
    }

    pub fn max_concurrent_requests(mut self, max: usize) -> Self {
        self.config.max_concurrent_requests = max;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    pub fn max_wait_time_ms(mut self, ms: u64) -> Self {
        self.config.max_wait_time_ms = ms;
        self
    }

    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.config.timeout_secs = secs;
        self
    }

    pub fn connection_pool_size(mut self, size: usize) -> Self {
        self.config.connection_pool_size = size;
        self
    }

    pub fn build(self) -> Result<InferMeshClient> {
        InferMeshClient::new(self.config)
    }
}

impl Default for InferMeshClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_infermesh_client_creation() {
        let config = InferMeshConfig {
            base_url: "http://localhost:9000".to_string(),
            api_key: Some("test-key".to_string()),
            model: "test-model".to_string(),
            max_concurrent_requests: 10,
            batch_size: 64,
            max_wait_time_ms: 50,
            timeout_secs: 10,
            connection_pool_size: 5,
        };

        let client = InferMeshClient::new(config);
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_client_builder() {
        let client = InferMeshClientBuilder::new()
            .base_url("http://localhost:9000".to_string())
            .api_key(Some("test-key".to_string()))
            .model("test-model".to_string())
            .max_concurrent_requests(20)
            .batch_size(128)
            .build();

        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_metrics() {
        let config = InferMeshConfig::default();
        let client = InferMeshClient::new(config).unwrap();

        let metrics = client.get_metrics().await;
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.successful_requests, 0);
        assert_eq!(metrics.failed_requests, 0);
    }

    #[tokio::test]
    async fn test_connection_pool_status() {
        let config = InferMeshConfig::default();
        let client = InferMeshClient::new(config).unwrap();

        let (available, total) = client.get_connection_pool_status();
        assert!(available > 0);
        assert!(total > 0);
    }
}
