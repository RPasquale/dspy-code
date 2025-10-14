use crate::manager::EnvManager;
use crate::pb::env_manager_service_server::{EnvManagerService, EnvManagerServiceServer};
use crate::pb::*;
use futures_util::StreamExt;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::{wrappers::ReceiverStream, Stream};
use tonic::{transport::Server, Request, Response, Status};
use tracing::{error, info};

pub struct GrpcServer {
    manager: Arc<EnvManager>,
}

impl GrpcServer {
    pub fn new(manager: Arc<EnvManager>) -> Self {
        Self { manager }
    }

    pub async fn serve(self, addr: SocketAddr) -> anyhow::Result<()> {
        info!("gRPC server listening on {}", addr);

        Server::builder()
            .add_service(EnvManagerServiceServer::new(self))
            .serve(addr)
            .await
            .map_err(|e| anyhow::anyhow!("gRPC server error: {}", e))?;

        Ok(())
    }
}

#[tonic::async_trait]
impl EnvManagerService for GrpcServer {
    type StartServicesStream =
        Pin<Box<dyn Stream<Item = Result<ServiceStatusUpdate, Status>> + Send>>;
    type StreamHealthStream = Pin<Box<dyn Stream<Item = Result<HealthUpdate, Status>> + Send>>;
    type PullImagesStream = Pin<Box<dyn Stream<Item = Result<ImagePullProgress, Status>> + Send>>;
    type StreamLogsStream = Pin<Box<dyn Stream<Item = Result<LogEntry, Status>> + Send>>;

    async fn start_services(
        &self,
        request: Request<StartServicesRequest>,
    ) -> Result<Response<Self::StartServicesStream>, Status> {
        let req = request.into_inner();
        info!("gRPC: Starting services (parallel: {})", req.parallel);

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let manager = Arc::clone(&self.manager);
        let selection = req.service_names.clone();

        tokio::spawn(async move {
            // Send initial status updates
            let services = if req.service_names.is_empty() {
                manager
                    .get_services_status()
                    .await
                    .keys()
                    .cloned()
                    .collect()
            } else {
                req.service_names.clone()
            };

            for service_name in &services {
                let _ = tx
                    .send(Ok(ServiceStatusUpdate {
                        service_name: service_name.clone(),
                        status: "starting".to_string(),
                        message: format!("Preparing to start {}", service_name),
                        progress: 0,
                    }))
                    .await;
            }

            // Start services
            if let Err(e) = manager.start_services(&selection, req.parallel).await {
                error!("Failed to start services: {}", e);
                for service_name in &services {
                    let _ = tx
                        .send(Ok(ServiceStatusUpdate {
                            service_name: service_name.clone(),
                            status: "failed".to_string(),
                            message: e.to_string(),
                            progress: 0,
                        }))
                        .await;
                }
                return;
            }

            // Send final status updates
            let final_status = manager.get_services_status().await;
            for (service_name, state) in final_status {
                let _ = tx
                    .send(Ok(ServiceStatusUpdate {
                        service_name: service_name.clone(),
                        status: state.status.as_str().to_string(),
                        message: format!("{} is {}", service_name, state.status.as_str()),
                        progress: 100,
                    }))
                    .await;
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn stop_services(
        &self,
        request: Request<StopServicesRequest>,
    ) -> Result<Response<StopServicesResponse>, Status> {
        let req = request.into_inner();
        info!("gRPC: Stopping services {:?}", req.service_names);

        let timeout = if req.timeout_seconds > 0 {
            Some(req.timeout_seconds as i64)
        } else {
            None
        };

        let targets = req.service_names.clone();
        match self.manager.stop_services(&targets, timeout).await {
            Ok(_) => {
                let mut results = std::collections::HashMap::new();
                if targets.is_empty() {
                    results.insert("all".to_string(), "stopped".to_string());
                } else {
                    for name in targets {
                        results.insert(name, "stopped".to_string());
                    }
                }

                Ok(Response::new(StopServicesResponse {
                    success: true,
                    service_results: results,
                }))
            }
            Err(e) => {
                error!("Failed to stop services: {}", e);
                Ok(Response::new(StopServicesResponse {
                    success: false,
                    service_results: std::collections::HashMap::new(),
                }))
            }
        }
    }

    async fn restart_service(
        &self,
        request: Request<RestartServiceRequest>,
    ) -> Result<Response<RestartServiceResponse>, Status> {
        let req = request.into_inner();
        info!("gRPC: Restarting service: {}", req.service_name);

        let timeout = if req.timeout_seconds > 0 {
            Some(req.timeout_seconds as i64)
        } else {
            None
        };

        // Stop the service
        if let Err(e) = self.manager.stop_service(&req.service_name, timeout).await {
            return Ok(Response::new(RestartServiceResponse {
                success: false,
                message: format!("Failed to stop service: {}", e),
            }));
        }

        // Get service definition and restart
        if let Some(service_def) = self.manager.get_service_definition(&req.service_name) {
            match self.manager.start_service(service_def).await {
                Ok(_) => Ok(Response::new(RestartServiceResponse {
                    success: true,
                    message: format!("Service {} restarted successfully", req.service_name),
                })),
                Err(e) => Ok(Response::new(RestartServiceResponse {
                    success: false,
                    message: format!("Failed to start service: {}", e),
                })),
            }
        } else {
            Ok(Response::new(RestartServiceResponse {
                success: false,
                message: format!("Service {} not found", req.service_name),
            }))
        }
    }

    async fn get_services_status(
        &self,
        _request: Request<GetServicesStatusRequest>,
    ) -> Result<Response<ServicesStatusResponse>, Status> {
        let status = self.manager.get_services_status().await;

        let services: std::collections::HashMap<String, ServiceStatus> = status
            .iter()
            .map(|(name, state)| {
                (
                    name.clone(),
                    ServiceStatus {
                        name: name.clone(),
                        status: state.status.as_str().to_string(),
                        container_id: state.container_id.clone().unwrap_or_default(),
                        ports: vec![],
                        health_checks: std::collections::HashMap::new(),
                        started_at: state.started_at.unwrap_or(0),
                        resource_usage: None,
                    },
                )
            })
            .collect();

        Ok(Response::new(ServicesStatusResponse { services }))
    }

    async fn stream_health(
        &self,
        request: Request<StreamHealthRequest>,
    ) -> Result<Response<Self::StreamHealthStream>, Status> {
        let req = request.into_inner();
        let interval = std::time::Duration::from_secs(req.interval_seconds.max(1) as u64);

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let manager = Arc::clone(&self.manager);

        tokio::spawn(async move {
            loop {
                let status = manager.get_services_status().await;

                for (service_name, state) in status {
                    if !req.service_names.is_empty() && !req.service_names.contains(&service_name) {
                        continue;
                    }

                    let healthy = matches!(state.status, crate::manager::ServiceStatus::Running);

                    let _ = tx
                        .send(Ok(HealthUpdate {
                            service_name: service_name.clone(),
                            healthy,
                            message: state.status.as_str().to_string(),
                            timestamp: chrono::Utc::now().timestamp(),
                            details: std::collections::HashMap::new(),
                        }))
                        .await;
                }

                tokio::time::sleep(interval).await;
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn get_resource_availability(
        &self,
        _request: Request<GetResourceAvailabilityRequest>,
    ) -> Result<Response<ResourceAvailabilityResponse>, Status> {
        // TODO: Implement actual resource detection
        Ok(Response::new(ResourceAvailabilityResponse {
            available: true,
            capacity: 10,
            in_use: 0,
            hardware: Some(HardwareInfo {
                cpu_cores: num_cpus::get() as i32,
                memory_bytes: 8589934592, // 8GB placeholder
                gpus: vec![],
            }),
        }))
    }

    async fn execute_task(
        &self,
        request: Request<ExecuteTaskRequest>,
    ) -> Result<Response<ExecuteTaskResponse>, Status> {
        let req = request.into_inner();
        info!("gRPC: Executing task: {}", req.task_id);

        // This would forward to env_runner_rs for actual execution
        // For now, return a placeholder response
        Ok(Response::new(ExecuteTaskResponse {
            success: true,
            result: req.payload.clone(),
            error: String::new(),
            latency_ms: 0.0,
        }))
    }

    async fn pull_images(
        &self,
        request: Request<PullImagesRequest>,
    ) -> Result<Response<Self::PullImagesStream>, Status> {
        let req = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        let images = if req.image_names.is_empty() {
            vec![
                "redis:7-alpine".to_string(),
                "dspy/reddb:latest".to_string(),
            ]
        } else {
            req.image_names
        };

        tokio::spawn(async move {
            for image in images {
                let _ = tx
                    .send(Ok(ImagePullProgress {
                        image_name: image.clone(),
                        status: "downloading".to_string(),
                        current: 0,
                        total: 100,
                        progress_percent: 0,
                    }))
                    .await;

                // Simulate progress
                for progress in [25, 50, 75, 100] {
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                    let _ = tx
                        .send(Ok(ImagePullProgress {
                            image_name: image.clone(),
                            status: if progress == 100 {
                                "complete"
                            } else {
                                "downloading"
                            }
                            .to_string(),
                            current: progress,
                            total: 100,
                            progress_percent: progress as i32,
                        }))
                        .await;
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn stream_logs(
        &self,
        request: Request<StreamLogsRequest>,
    ) -> Result<Response<Self::StreamLogsStream>, Status> {
        let req = request.into_inner();

        if req.service_name.is_empty() {
            return Err(Status::invalid_argument("service_name is required"));
        }

        info!("gRPC: Streaming logs for service: {}", req.service_name);

        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let manager = Arc::clone(&self.manager);
        let service_name = req.service_name.clone();

        tokio::spawn(async move {
            // Get the service status to find container_id
            let status = manager.get_services_status().await;

            if let Some(state) = status.get(&service_name) {
                if let Some(container_id) = &state.container_id {
                    // Stream logs from the container
                    match manager
                        .stream_container_logs(
                            container_id,
                            req.follow,
                            req.tail,
                            req.since_timestamp,
                        )
                        .await
                    {
                        Ok(mut log_stream) => {
                            while let Some(log_result) = log_stream.next().await {
                                match log_result {
                                    Ok((stream_type, message, timestamp)) => {
                                        if tx
                                            .send(Ok(LogEntry {
                                                service_name: service_name.clone(),
                                                container_id: container_id.clone(),
                                                stream: stream_type,
                                                message,
                                                timestamp,
                                            }))
                                            .await
                                            .is_err()
                                        {
                                            break; // Client disconnected
                                        }
                                    }
                                    Err(e) => {
                                        error!("Error reading logs: {}", e);
                                        let _ = tx
                                            .send(Err(Status::internal(format!(
                                                "Log error: {}",
                                                e
                                            ))))
                                            .await;
                                        break;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to stream logs: {}", e);
                            let _ = tx
                                .send(Err(Status::internal(format!(
                                    "Failed to stream logs: {}",
                                    e
                                ))))
                                .await;
                        }
                    }
                } else {
                    let _ = tx
                        .send(Err(Status::not_found("Container not found for service")))
                        .await;
                }
            } else {
                let _ = tx
                    .send(Err(Status::not_found(format!(
                        "Service {} not found",
                        service_name
                    ))))
                    .await;
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}
