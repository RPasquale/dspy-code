// @generated
/// Generated client implementations.
pub mod env_manager_service_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    #[derive(Debug, Clone)]
    pub struct EnvManagerServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl EnvManagerServiceClient<tonic::transport::Channel> {
        /// Attempt to create a new client by connecting to a given endpoint.
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> EnvManagerServiceClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::BoxBody>,
        T::Error: Into<StdError>,
        T::ResponseBody: Body<Data = Bytes> + Send + 'static,
        <T::ResponseBody as Body>::Error: Into<StdError> + Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_origin(inner: T, origin: Uri) -> Self {
            let inner = tonic::client::Grpc::with_origin(inner, origin);
            Self { inner }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> EnvManagerServiceClient<InterceptedService<T, F>>
        where
            F: tonic::service::Interceptor,
            T::ResponseBody: Default,
            T: tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
                Response = http::Response<
                    <T as tonic::client::GrpcService<tonic::body::BoxBody>>::ResponseBody,
                >,
            >,
            <T as tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
            >>::Error: Into<StdError> + Send + Sync,
        {
            EnvManagerServiceClient::new(InterceptedService::new(inner, interceptor))
        }
        /// Compress requests with the given encoding.
        ///
        /// This requires the server to support it otherwise it might respond with an
        /// error.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.send_compressed(encoding);
            self
        }
        /// Enable decompressing responses.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.accept_compressed(encoding);
            self
        }
        /// Limits the maximum size of a decoded message.
        ///
        /// Default: `4MB`
        #[must_use]
        pub fn max_decoding_message_size(mut self, limit: usize) -> Self {
            self.inner = self.inner.max_decoding_message_size(limit);
            self
        }
        /// Limits the maximum size of an encoded message.
        ///
        /// Default: `usize::MAX`
        #[must_use]
        pub fn max_encoding_message_size(mut self, limit: usize) -> Self {
            self.inner = self.inner.max_encoding_message_size(limit);
            self
        }
        pub async fn start_services(
            &mut self,
            request: impl tonic::IntoRequest<super::StartServicesRequest>,
        ) -> std::result::Result<
            tonic::Response<tonic::codec::Streaming<super::ServiceStatusUpdate>>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/env_manager.v1.EnvManagerService/StartServices",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new("env_manager.v1.EnvManagerService", "StartServices"),
                );
            self.inner.server_streaming(req, path, codec).await
        }
        pub async fn stop_services(
            &mut self,
            request: impl tonic::IntoRequest<super::StopServicesRequest>,
        ) -> std::result::Result<
            tonic::Response<super::StopServicesResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/env_manager.v1.EnvManagerService/StopServices",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new("env_manager.v1.EnvManagerService", "StopServices"),
                );
            self.inner.unary(req, path, codec).await
        }
        pub async fn restart_service(
            &mut self,
            request: impl tonic::IntoRequest<super::RestartServiceRequest>,
        ) -> std::result::Result<
            tonic::Response<super::RestartServiceResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/env_manager.v1.EnvManagerService/RestartService",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new("env_manager.v1.EnvManagerService", "RestartService"),
                );
            self.inner.unary(req, path, codec).await
        }
        pub async fn get_services_status(
            &mut self,
            request: impl tonic::IntoRequest<super::GetServicesStatusRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ServicesStatusResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/env_manager.v1.EnvManagerService/GetServicesStatus",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new(
                        "env_manager.v1.EnvManagerService",
                        "GetServicesStatus",
                    ),
                );
            self.inner.unary(req, path, codec).await
        }
        pub async fn stream_health(
            &mut self,
            request: impl tonic::IntoRequest<super::StreamHealthRequest>,
        ) -> std::result::Result<
            tonic::Response<tonic::codec::Streaming<super::HealthUpdate>>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/env_manager.v1.EnvManagerService/StreamHealth",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new("env_manager.v1.EnvManagerService", "StreamHealth"),
                );
            self.inner.server_streaming(req, path, codec).await
        }
        pub async fn get_resource_availability(
            &mut self,
            request: impl tonic::IntoRequest<super::GetResourceAvailabilityRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ResourceAvailabilityResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/env_manager.v1.EnvManagerService/GetResourceAvailability",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new(
                        "env_manager.v1.EnvManagerService",
                        "GetResourceAvailability",
                    ),
                );
            self.inner.unary(req, path, codec).await
        }
        pub async fn execute_task(
            &mut self,
            request: impl tonic::IntoRequest<super::ExecuteTaskRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ExecuteTaskResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/env_manager.v1.EnvManagerService/ExecuteTask",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new("env_manager.v1.EnvManagerService", "ExecuteTask"),
                );
            self.inner.unary(req, path, codec).await
        }
        pub async fn pull_images(
            &mut self,
            request: impl tonic::IntoRequest<super::PullImagesRequest>,
        ) -> std::result::Result<
            tonic::Response<tonic::codec::Streaming<super::ImagePullProgress>>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/env_manager.v1.EnvManagerService/PullImages",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new("env_manager.v1.EnvManagerService", "PullImages"),
                );
            self.inner.server_streaming(req, path, codec).await
        }
        pub async fn stream_logs(
            &mut self,
            request: impl tonic::IntoRequest<super::StreamLogsRequest>,
        ) -> std::result::Result<
            tonic::Response<tonic::codec::Streaming<super::LogEntry>>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/env_manager.v1.EnvManagerService/StreamLogs",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(
                    GrpcMethod::new("env_manager.v1.EnvManagerService", "StreamLogs"),
                );
            self.inner.server_streaming(req, path, codec).await
        }
    }
}
/// Generated server implementations.
pub mod env_manager_service_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with EnvManagerServiceServer.
    #[async_trait]
    pub trait EnvManagerService: Send + Sync + 'static {
        /// Server streaming response type for the StartServices method.
        type StartServicesStream: tonic::codegen::tokio_stream::Stream<
                Item = std::result::Result<super::ServiceStatusUpdate, tonic::Status>,
            >
            + Send
            + 'static;
        async fn start_services(
            &self,
            request: tonic::Request<super::StartServicesRequest>,
        ) -> std::result::Result<
            tonic::Response<Self::StartServicesStream>,
            tonic::Status,
        >;
        async fn stop_services(
            &self,
            request: tonic::Request<super::StopServicesRequest>,
        ) -> std::result::Result<
            tonic::Response<super::StopServicesResponse>,
            tonic::Status,
        >;
        async fn restart_service(
            &self,
            request: tonic::Request<super::RestartServiceRequest>,
        ) -> std::result::Result<
            tonic::Response<super::RestartServiceResponse>,
            tonic::Status,
        >;
        async fn get_services_status(
            &self,
            request: tonic::Request<super::GetServicesStatusRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ServicesStatusResponse>,
            tonic::Status,
        >;
        /// Server streaming response type for the StreamHealth method.
        type StreamHealthStream: tonic::codegen::tokio_stream::Stream<
                Item = std::result::Result<super::HealthUpdate, tonic::Status>,
            >
            + Send
            + 'static;
        async fn stream_health(
            &self,
            request: tonic::Request<super::StreamHealthRequest>,
        ) -> std::result::Result<
            tonic::Response<Self::StreamHealthStream>,
            tonic::Status,
        >;
        async fn get_resource_availability(
            &self,
            request: tonic::Request<super::GetResourceAvailabilityRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ResourceAvailabilityResponse>,
            tonic::Status,
        >;
        async fn execute_task(
            &self,
            request: tonic::Request<super::ExecuteTaskRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ExecuteTaskResponse>,
            tonic::Status,
        >;
        /// Server streaming response type for the PullImages method.
        type PullImagesStream: tonic::codegen::tokio_stream::Stream<
                Item = std::result::Result<super::ImagePullProgress, tonic::Status>,
            >
            + Send
            + 'static;
        async fn pull_images(
            &self,
            request: tonic::Request<super::PullImagesRequest>,
        ) -> std::result::Result<tonic::Response<Self::PullImagesStream>, tonic::Status>;
        /// Server streaming response type for the StreamLogs method.
        type StreamLogsStream: tonic::codegen::tokio_stream::Stream<
                Item = std::result::Result<super::LogEntry, tonic::Status>,
            >
            + Send
            + 'static;
        async fn stream_logs(
            &self,
            request: tonic::Request<super::StreamLogsRequest>,
        ) -> std::result::Result<tonic::Response<Self::StreamLogsStream>, tonic::Status>;
    }
    #[derive(Debug)]
    pub struct EnvManagerServiceServer<T: EnvManagerService> {
        inner: Arc<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
        max_decoding_message_size: Option<usize>,
        max_encoding_message_size: Option<usize>,
    }
    impl<T: EnvManagerService> EnvManagerServiceServer<T> {
        pub fn new(inner: T) -> Self {
            Self::from_arc(Arc::new(inner))
        }
        pub fn from_arc(inner: Arc<T>) -> Self {
            Self {
                inner,
                accept_compression_encodings: Default::default(),
                send_compression_encodings: Default::default(),
                max_decoding_message_size: None,
                max_encoding_message_size: None,
            }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> InterceptedService<Self, F>
        where
            F: tonic::service::Interceptor,
        {
            InterceptedService::new(Self::new(inner), interceptor)
        }
        /// Enable decompressing requests with the given encoding.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.accept_compression_encodings.enable(encoding);
            self
        }
        /// Compress responses with the given encoding, if the client supports it.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.send_compression_encodings.enable(encoding);
            self
        }
        /// Limits the maximum size of a decoded message.
        ///
        /// Default: `4MB`
        #[must_use]
        pub fn max_decoding_message_size(mut self, limit: usize) -> Self {
            self.max_decoding_message_size = Some(limit);
            self
        }
        /// Limits the maximum size of an encoded message.
        ///
        /// Default: `usize::MAX`
        #[must_use]
        pub fn max_encoding_message_size(mut self, limit: usize) -> Self {
            self.max_encoding_message_size = Some(limit);
            self
        }
    }
    impl<T, B> tonic::codegen::Service<http::Request<B>> for EnvManagerServiceServer<T>
    where
        T: EnvManagerService,
        B: Body + Send + 'static,
        B::Error: Into<StdError> + Send + 'static,
    {
        type Response = http::Response<tonic::body::BoxBody>;
        type Error = std::convert::Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;
        fn poll_ready(
            &mut self,
            _cx: &mut Context<'_>,
        ) -> Poll<std::result::Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, req: http::Request<B>) -> Self::Future {
            match req.uri().path() {
                "/env_manager.v1.EnvManagerService/StartServices" => {
                    #[allow(non_camel_case_types)]
                    struct StartServicesSvc<T: EnvManagerService>(pub Arc<T>);
                    impl<
                        T: EnvManagerService,
                    > tonic::server::ServerStreamingService<super::StartServicesRequest>
                    for StartServicesSvc<T> {
                        type Response = super::ServiceStatusUpdate;
                        type ResponseStream = T::StartServicesStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::StartServicesRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as EnvManagerService>::start_services(&inner, request)
                                    .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = StartServicesSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/env_manager.v1.EnvManagerService/StopServices" => {
                    #[allow(non_camel_case_types)]
                    struct StopServicesSvc<T: EnvManagerService>(pub Arc<T>);
                    impl<
                        T: EnvManagerService,
                    > tonic::server::UnaryService<super::StopServicesRequest>
                    for StopServicesSvc<T> {
                        type Response = super::StopServicesResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::StopServicesRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as EnvManagerService>::stop_services(&inner, request)
                                    .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = StopServicesSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/env_manager.v1.EnvManagerService/RestartService" => {
                    #[allow(non_camel_case_types)]
                    struct RestartServiceSvc<T: EnvManagerService>(pub Arc<T>);
                    impl<
                        T: EnvManagerService,
                    > tonic::server::UnaryService<super::RestartServiceRequest>
                    for RestartServiceSvc<T> {
                        type Response = super::RestartServiceResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::RestartServiceRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as EnvManagerService>::restart_service(&inner, request)
                                    .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = RestartServiceSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/env_manager.v1.EnvManagerService/GetServicesStatus" => {
                    #[allow(non_camel_case_types)]
                    struct GetServicesStatusSvc<T: EnvManagerService>(pub Arc<T>);
                    impl<
                        T: EnvManagerService,
                    > tonic::server::UnaryService<super::GetServicesStatusRequest>
                    for GetServicesStatusSvc<T> {
                        type Response = super::ServicesStatusResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::GetServicesStatusRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as EnvManagerService>::get_services_status(
                                        &inner,
                                        request,
                                    )
                                    .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = GetServicesStatusSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/env_manager.v1.EnvManagerService/StreamHealth" => {
                    #[allow(non_camel_case_types)]
                    struct StreamHealthSvc<T: EnvManagerService>(pub Arc<T>);
                    impl<
                        T: EnvManagerService,
                    > tonic::server::ServerStreamingService<super::StreamHealthRequest>
                    for StreamHealthSvc<T> {
                        type Response = super::HealthUpdate;
                        type ResponseStream = T::StreamHealthStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::StreamHealthRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as EnvManagerService>::stream_health(&inner, request)
                                    .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = StreamHealthSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/env_manager.v1.EnvManagerService/GetResourceAvailability" => {
                    #[allow(non_camel_case_types)]
                    struct GetResourceAvailabilitySvc<T: EnvManagerService>(pub Arc<T>);
                    impl<
                        T: EnvManagerService,
                    > tonic::server::UnaryService<super::GetResourceAvailabilityRequest>
                    for GetResourceAvailabilitySvc<T> {
                        type Response = super::ResourceAvailabilityResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                super::GetResourceAvailabilityRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as EnvManagerService>::get_resource_availability(
                                        &inner,
                                        request,
                                    )
                                    .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = GetResourceAvailabilitySvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/env_manager.v1.EnvManagerService/ExecuteTask" => {
                    #[allow(non_camel_case_types)]
                    struct ExecuteTaskSvc<T: EnvManagerService>(pub Arc<T>);
                    impl<
                        T: EnvManagerService,
                    > tonic::server::UnaryService<super::ExecuteTaskRequest>
                    for ExecuteTaskSvc<T> {
                        type Response = super::ExecuteTaskResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ExecuteTaskRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as EnvManagerService>::execute_task(&inner, request)
                                    .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = ExecuteTaskSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/env_manager.v1.EnvManagerService/PullImages" => {
                    #[allow(non_camel_case_types)]
                    struct PullImagesSvc<T: EnvManagerService>(pub Arc<T>);
                    impl<
                        T: EnvManagerService,
                    > tonic::server::ServerStreamingService<super::PullImagesRequest>
                    for PullImagesSvc<T> {
                        type Response = super::ImagePullProgress;
                        type ResponseStream = T::PullImagesStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::PullImagesRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as EnvManagerService>::pull_images(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = PullImagesSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/env_manager.v1.EnvManagerService/StreamLogs" => {
                    #[allow(non_camel_case_types)]
                    struct StreamLogsSvc<T: EnvManagerService>(pub Arc<T>);
                    impl<
                        T: EnvManagerService,
                    > tonic::server::ServerStreamingService<super::StreamLogsRequest>
                    for StreamLogsSvc<T> {
                        type Response = super::LogEntry;
                        type ResponseStream = T::StreamLogsStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::StreamLogsRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as EnvManagerService>::stream_logs(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = StreamLogsSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                _ => {
                    Box::pin(async move {
                        Ok(
                            http::Response::builder()
                                .status(200)
                                .header("grpc-status", tonic::Code::Unimplemented as i32)
                                .header(
                                    http::header::CONTENT_TYPE,
                                    tonic::metadata::GRPC_CONTENT_TYPE,
                                )
                                .body(empty_body())
                                .unwrap(),
                        )
                    })
                }
            }
        }
    }
    impl<T: EnvManagerService> Clone for EnvManagerServiceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
                max_decoding_message_size: self.max_decoding_message_size,
                max_encoding_message_size: self.max_encoding_message_size,
            }
        }
    }
    impl<T: EnvManagerService> tonic::server::NamedService
    for EnvManagerServiceServer<T> {
        const NAME: &'static str = "env_manager.v1.EnvManagerService";
    }
}
