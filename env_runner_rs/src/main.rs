use std::fs;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use base64::engine::general_purpose::STANDARD as Base64;
use base64::Engine;
use chrono::Utc;
use env_runner_rs::hardware::{detect, recommended_inflight, HardwareSnapshot};
use env_runner_rs::metrics::{EnvRunnerMetrics, MetricsServer};
use env_runner_rs::pb;
use env_runner_rs::streaming::{
    collect_assignment_headers, derive_skill_from_assignment, extract_workflow_meta,
};
use futures::StreamExt;
use pb::mesh::mesh_data_client::MeshDataClient;
use pb::mesh::{AckRequest, Received, SubscribeRequest};
use pb::runner::stream_supervisor_client::StreamSupervisorClient;
use pb::runner::worker_to_supervisor::Msg as WorkerMsg;
use pb::runner::{
    CreditReport, TaskAck, TaskAssignment as ProtoTaskAssignment, WorkerHello, WorkerToSupervisor,
};
use prost::Message;
use rdkafka::message::{Header, OwnedHeaders};
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::ClientConfig;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{mpsc, Semaphore};
use tokio::time::sleep;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize)]
struct MeshServiceConfig {
    id: u64,
    endpoint: String,
    #[serde(default)]
    domain: String,
}

#[derive(Clone, Debug)]
struct WorkerConfig {
    worker_id: String,
    kafka_brokers: String,
    output_topic: String,
    supervisor_addr: String,
    max_inflight: usize,
    mesh_endpoint: Option<String>,
    mesh_node_id: Option<u64>,
    mesh_domain: Option<String>,
    rl_results_topic: Option<String>,
    reddb_url: Option<String>,
    reddb_namespace: String,
    reddb_stream: String,
    reddb_token: Option<String>,
}

struct RedDBSink {
    client: reqwest::Client,
    url: String,
    token: Option<String>,
}

impl RedDBSink {
    fn new(base_url: &str, namespace: &str, stream: &str, token: Option<String>) -> Result<Self> {
        let base = base_url.trim_end_matches('/');
        let url = format!("{}/api/streams/{}/{}/append", base, namespace, stream);
        Ok(Self {
            client: reqwest::Client::new(),
            url,
            token,
        })
    }

    async fn publish(&self, assignment: &ProtoTaskAssignment) -> Result<()> {
        let payload_b64 = Base64.encode(&assignment.payload);
        let body = json!({
            "task_id": assignment.task_id,
            "tenant": assignment.tenant,
            "topic": assignment.topic,
            "partition": assignment.partition,
            "offset": assignment.offset,
            "payload_b64": payload_b64,
            "ts": Utc::now().to_rfc3339(),
        });

        let mut request = self.client.post(&self.url).json(&body);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await.context("send RedDB append request")?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "reddb append failed: status={} body={}",
                status,
                text
            ));
        }
        Ok(())
    }
}

#[derive(Default)]
struct WorkerTelemetry {
    max_inflight: u64,
    active: AtomicU64,
    processed: AtomicU64,
    failed: AtomicU64,
}

impl WorkerTelemetry {
    fn new(max_inflight: usize) -> Self {
        Self {
            max_inflight: max_inflight as u64,
            ..Self::default()
        }
    }

    fn on_start(&self) -> u64 {
        self.active.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn on_finish(&self, success: bool) -> (u64, u64, u64) {
        if success {
            self.processed.fetch_add(1, Ordering::SeqCst);
        } else {
            self.failed.fetch_add(1, Ordering::SeqCst);
        }
        let active = self.active.fetch_sub(1, Ordering::SeqCst).saturating_sub(1);
        let processed = self.processed.load(Ordering::SeqCst);
        let failed = self.failed.load(Ordering::SeqCst);
        (active, processed, failed)
    }

    fn queue_depth(&self, active: u64) -> u64 {
        self.max_inflight.saturating_sub(active)
    }
}

#[allow(dead_code)]
fn spawn_hardware_refresh(hardware: Arc<RwLock<HardwareSnapshot>>) {
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(60)).await;
            let snapshot = detect();
            if let Ok(mut guard) = hardware.write() {
                *guard = snapshot;
            }
        }
    });
}

struct ProcessingContext {
    producer: Arc<FutureProducer>,
    output_topic: String,
    ack_client: Arc<tokio::sync::Mutex<MeshDataClient<tonic::transport::Channel>>>,
    node_id: u64,
    worker_id: String,
    semaphore: Arc<Semaphore>,
    tx_supervisor: mpsc::Sender<WorkerToSupervisor>,
    telemetry: Arc<WorkerTelemetry>,
    reddb: Option<Arc<RedDBSink>>,
    rl_topic: Option<String>,
    hardware: Arc<RwLock<HardwareSnapshot>>,
    max_credits: u32,
    credit_granularity: u32,
    last_credit: AtomicU32,
}

impl ProcessingContext {
    async fn handle_received(self: Arc<Self>, received: Received) {
        let permit = match self.semaphore.clone().acquire_owned().await {
            Ok(permit) => permit,
            Err(err) => {
                error!(error = %err, "failed to acquire semaphore permit");
                return;
            }
        };

        let active = self.telemetry.on_start();
        let queue_depth = self.telemetry.queue_depth(active);
        debug!(worker = %self.worker_id, active_inflight = active, queue_depth, "accepted mesh assignment");

        let ctx = self.clone();
        tokio::spawn(async move {
            let start = Instant::now();
            let mut assignment: Option<ProtoTaskAssignment> = None;
            let mut error_msg: Option<String> = None;
            let mut success = false;

            match ProtoTaskAssignment::decode(received.payload.as_ref()) {
                Ok(decoded) => {
                    assignment = Some(decoded);
                }
                Err(err) => {
                    error_msg = Some(format!("decode assignment: {err}"));
                }
            }

            if let Some(ref assignment) = assignment {
                if let Err(err) = ctx.publish_assignment(assignment).await {
                    error_msg = Some(err.to_string());
                } else {
                    success = true;
                }
            }

            if let Err(err) = ctx.ack_mesh(received.delivery_id.clone()).await {
                let ack_err = format!("mesh ack failed: {err}");
                error!(delivery_id = %received.delivery_id, error = %err, "failed to ack mesh message");
                let previous = error_msg.take();
                error_msg = Some(match previous {
                    Some(existing) => format!("{}; {}", existing, ack_err),
                    None => ack_err,
                });
                success = false;
            }

            if let Some(task) = assignment.as_ref() {
                if let Err(err) = ctx
                    .send_ack(
                        task.task_id.clone(),
                        success,
                        error_msg.clone().unwrap_or_default(),
                    )
                    .await
                {
                    error!(task_id = %task.task_id, error = %err, "failed to send supervisor ack");
                }
            } else if let Some(err) = error_msg.as_ref() {
                warn!(delivery_id = %received.delivery_id, error = %err, "missing task_id in mesh payload; supervisor ack skipped");
            }

            drop(permit);
            if let Err(err) = ctx.send_credit().await {
                error!(error = %err, "failed to send credit update");
            }

            let latency = start.elapsed();
            let (active_after, processed, failed) = ctx.telemetry.on_finish(success);
            if success {
                info!(
                    worker = %ctx.worker_id,
                    latency_ms = latency.as_secs_f64() * 1_000.0,
                    active_inflight = active_after,
                    processed_total = processed,
                    "processed mesh assignment"
                );
            } else {
                error!(
                    worker = %ctx.worker_id,
                    latency_ms = latency.as_secs_f64() * 1_000.0,
                    active_inflight = active_after,
                    processed_total = processed,
                    failed_total = failed,
                    error = ?error_msg,
                    "mesh assignment failed"
                );
            }
            debug!(
                worker = %ctx.worker_id,
                available_credits = ctx.semaphore.available_permits(),
                "credit window refreshed"
            );
        });
    }

    async fn publish_assignment(&self, assignment: &ProtoTaskAssignment) -> Result<()> {
        let skill_hint = derive_skill_from_assignment(assignment);
        let hardware_snapshot = self.hardware.read().ok().map(|guard| guard.clone());
        let kafka = publish_assignment_to_topic(
            &self.producer,
            &self.output_topic,
            assignment,
            skill_hint.as_deref(),
            hardware_snapshot.as_ref(),
        );
        let reddb = async {
            if let Some(sink) = &self.reddb {
                sink.publish(assignment).await
            } else {
                Ok(())
            }
        };

        let (kafka_res, reddb_res) = tokio::join!(kafka, reddb);
        kafka_res?;
        reddb_res?;

        if let Some(topic) = &self.rl_topic {
            if topic == &self.output_topic {
                return Ok(());
            }
            if let Err(err) = publish_assignment_to_topic(
                &self.producer,
                topic,
                assignment,
                skill_hint.as_deref(),
                hardware_snapshot.as_ref(),
            )
            .await
            {
                warn!(%topic, error = %err, "rl topic publish failed");
            }
        }
        Ok(())
    }

    async fn ack_mesh(&self, delivery_id: String) -> Result<()> {
        let mut client = self.ack_client.lock().await;
        client
            .ack(tonic::Request::new(AckRequest {
                delivery_id,
                node_id: self.node_id,
            }))
            .await
            .context("mesh ack")?;
        Ok(())
    }

    async fn send_ack(&self, task_id: String, success: bool, error: String) -> Result<()> {
        self.tx_supervisor
            .send(WorkerToSupervisor {
                msg: Some(WorkerMsg::Ack(TaskAck {
                    task_id,
                    success,
                    error,
                })),
            })
            .await
            .context("send supervisor ack")
    }

    async fn send_credit(&self) -> Result<()> {
        let available = self.semaphore.available_permits() as u32;
        let previous = self.last_credit.swap(available, Ordering::SeqCst);
        if available == previous {
            return Ok(());
        }
        if available < previous
            && previous - available < self.credit_granularity
            && available != 0
            && available != self.max_credits
        {
            return Ok(());
        }
        self.tx_supervisor
            .send(WorkerToSupervisor {
                msg: Some(WorkerMsg::Credit(CreditReport {
                    worker_id: self.worker_id.clone(),
                    credits: available,
                })),
            })
            .await
            .context("send credit update")
    }
}

impl WorkerConfig {
    fn from_env() -> Self {
        let worker_id =
            std::env::var("WORKER_ID").unwrap_or_else(|_| format!("worker-{}", Uuid::new_v4()));
        let kafka_brokers =
            std::env::var("KAFKA_BROKERS").unwrap_or_else(|_| "broker:9092".to_string());
        let output_topic =
            std::env::var("OUTPUT_TOPIC").unwrap_or_else(|_| "features.events.demo".to_string());
        let supervisor_addr = std::env::var("SUPERVISOR_GRPC_ADDR")
            .unwrap_or_else(|_| "http://127.0.0.1:7000".to_string());
        let max_inflight = std::env::var("MAX_INFLIGHT")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .unwrap_or(4);
        let mesh_endpoint_raw = std::env::var("MESH_ENDPOINT")
            .ok()
            .filter(|s| !s.is_empty());
        let mesh_node_id = std::env::var("MESH_NODE_ID")
            .ok()
            .and_then(|val| val.parse::<u64>().ok());
        let mesh_domain_env = std::env::var("MESH_DOMAIN").ok().filter(|s| !s.is_empty());
        let mesh_services_file = std::env::var("MESH_SERVICES_FILE")
            .ok()
            .filter(|s| !s.is_empty());
        let mesh_services = if let Some(path) = mesh_services_file.as_ref() {
            match fs::read_to_string(path) {
                Ok(raw) => match serde_json::from_str::<Vec<MeshServiceConfig>>(&raw) {
                    Ok(list) => list,
                    Err(err) => {
                        warn!(file = %path, "failed to parse mesh services file: {err}");
                        Vec::new()
                    }
                },
                Err(err) => {
                    warn!(file = %path, "failed to read mesh services file: {err}");
                    Vec::new()
                }
            }
        } else {
            std::env::var("MESH_SERVICES_JSON")
                .ok()
                .and_then(|raw| serde_json::from_str::<Vec<MeshServiceConfig>>(&raw).ok())
                .unwrap_or_default()
        };

        let mut mesh_endpoint = mesh_endpoint_raw.clone();
        let mut mesh_domain = mesh_domain_env.clone();

        if mesh_endpoint.is_none() && !mesh_services.is_empty() {
            if let Some(node_id) = mesh_node_id {
                if let Some(entry) = mesh_services.iter().find(|svc| svc.id == node_id) {
                    mesh_endpoint = Some(entry.endpoint.clone());
                    if mesh_domain.is_none() && !entry.domain.is_empty() {
                        mesh_domain = Some(entry.domain.clone());
                    }
                }
            }
            if mesh_endpoint.is_none() {
                let entry = &mesh_services[0];
                mesh_endpoint = Some(entry.endpoint.clone());
                if mesh_domain.is_none() && !entry.domain.is_empty() {
                    mesh_domain = Some(entry.domain.clone());
                }
            }
        }

        let rl_results_topic = std::env::var("RL_RESULTS_TOPIC")
            .ok()
            .filter(|s| !s.is_empty());

        let reddb_url = std::env::var("REDDB_URL").ok().filter(|s| !s.is_empty());
        let reddb_namespace =
            std::env::var("REDDB_NAMESPACE").unwrap_or_else(|_| "dspy".to_string());
        let reddb_stream =
            std::env::var("REDDB_STREAM").unwrap_or_else(|_| "mesh_results".to_string());
        let reddb_token = std::env::var("REDDB_TOKEN").ok().filter(|s| !s.is_empty());

        Self {
            worker_id,
            kafka_brokers,
            output_topic,
            supervisor_addr,
            max_inflight,
            mesh_endpoint,
            mesh_node_id,
            mesh_domain,
            rl_results_topic,
            reddb_url,
            reddb_namespace,
            reddb_stream,
            reddb_token,
        }
    }
}

fn create_producer(cfg: &WorkerConfig) -> Result<FutureProducer> {
    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", &cfg.kafka_brokers)
        .set("message.timeout.ms", "30000")
        .set("queue.buffering.max.ms", "10")
        .set("compression.type", "lz4")
        .create()
        .context("failed to create kafka producer")?;
    Ok(producer)
}

async fn publish_assignment_to_topic(
    producer: &FutureProducer,
    topic: &str,
    assignment: &ProtoTaskAssignment,
    skill: Option<&str>,
    hardware: Option<&HardwareSnapshot>,
) -> Result<()> {
    let payload = assignment.payload.clone().to_vec();
    let task_id = assignment.task_id.clone();
    let meta = extract_workflow_meta(assignment);

    let mut record = FutureRecord::to(topic).key(&task_id).payload(&payload);
    let mut headers = OwnedHeaders::new();
    if !assignment.tenant.is_empty() {
        headers = headers.insert(Header {
            key: "tenant",
            value: Some(assignment.tenant.as_bytes()),
        });
    } else if let Some(tenant) = meta.tenant.as_ref() {
        headers = headers.insert(Header {
            key: "tenant",
            value: Some(tenant.as_bytes()),
        });
    }
    if !assignment.topic.is_empty() {
        headers = headers.insert(Header {
            key: "source_topic",
            value: Some(assignment.topic.as_bytes()),
        });
    }
    if let Some(skill) = skill {
        if !skill.is_empty() {
            headers = headers.insert(Header {
                key: "skill",
                value: Some(skill.as_bytes()),
            });
        }
    }
    let extra_headers = collect_assignment_headers(assignment, hardware);
    for (key, value) in extra_headers.iter() {
        headers = headers.insert(Header {
            key,
            value: Some(value.as_slice()),
        });
    }
    record = record.headers(headers);

    producer
        .send(record, Duration::from_secs(5))
        .await
        .map(|_| ())
        .map_err(|(err, _)| anyhow!("producer send failed: {err}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .compact()
        .init();

    let hardware_state = Arc::new(RwLock::new(detect()));
    spawn_hardware_refresh(hardware_state.clone());

    let metrics_state = Arc::new(EnvRunnerMetrics::new());
    let metrics_port = std::env::var("ENV_RUNNER_HTTP_PORT")
        .ok()
        .and_then(|val| val.parse::<u16>().ok())
        .unwrap_or(8083);
    let metrics_server = Arc::new(MetricsServer::new(
        metrics_state.clone(),
        hardware_state.clone(),
        metrics_port,
    ));
    let server_handle = metrics_server.clone();
    tokio::spawn(async move {
        if let Err(err) = server_handle.start().await {
            error!(port = metrics_port, error = %err, "metrics server terminated");
        }
    });

    let supervisor_enabled = std::env::var("ENABLE_SUPERVISOR")
        .map(|v| v == "1")
        .unwrap_or(false);

    if supervisor_enabled {
        let mut cfg = WorkerConfig::from_env();
        if cfg.max_inflight == 0 || std::env::var("MAX_INFLIGHT").is_err() {
            if let Ok(snapshot) = hardware_state.read() {
                cfg.max_inflight = recommended_inflight(&snapshot);
            }
        }

        if cfg.mesh_endpoint.is_some() {
            run_mesh_worker(cfg, hardware_state.clone()).await
        } else {
            run_kafka_worker(cfg, hardware_state.clone()).await
        }
    } else {
        info!(
            port = metrics_port,
            "env-runner started in HTTP execution mode"
        );
        tokio::signal::ctrl_c()
            .await
            .context("await ctrl-c signal")?;
        metrics_server.shutdown().await;
        Ok(())
    }
}

async fn run_kafka_worker(
    cfg: WorkerConfig,
    hardware: Arc<RwLock<HardwareSnapshot>>,
) -> Result<()> {
    info!(worker = %cfg.worker_id, transport = "kafka", max_inflight = cfg.max_inflight);

    let shared_cfg = Arc::new(cfg);
    let producer = Arc::new(create_producer(&shared_cfg)?);
    let (tx, rx) = mpsc::channel::<WorkerToSupervisor>(shared_cfg.max_inflight * 4);
    let outbound = ReceiverStream::new(rx);

    let mut client = StreamSupervisorClient::connect(shared_cfg.supervisor_addr.clone())
        .await
        .context("connect to supervisor")?;
    let mut stream = client
        .open_stream(outbound)
        .await
        .context("register with supervisor")?
        .into_inner();

    tx.send(WorkerToSupervisor {
        msg: Some(WorkerMsg::Hello(WorkerHello {
            worker_id: shared_cfg.worker_id.clone(),
            max_inflight: shared_cfg.max_inflight as u32,
            version: env!("CARGO_PKG_VERSION").to_string(),
            mesh_enabled: false,
            mesh_node_id: 0,
            mesh_domain: String::new(),
        })),
    })
    .await
    .ok();

    tx.send(WorkerToSupervisor {
        msg: Some(WorkerMsg::Credit(CreditReport {
            worker_id: shared_cfg.worker_id.clone(),
            credits: shared_cfg.max_inflight as u32,
        })),
    })
    .await
    .ok();

    let semaphore = Arc::new(Semaphore::new(shared_cfg.max_inflight));
    let tx_main = tx.clone();

    while let Some(message) = stream
        .next()
        .await
        .transpose()
        .context("receive from supervisor")?
    {
        if let Some(pb::runner::supervisor_to_worker::Msg::Assignment(assignment)) = message.msg {
            let permit = match semaphore.clone().try_acquire_owned() {
                Ok(permit) => permit,
                Err(_) => {
                    warn!("assignment arrived with no available permits");
                    continue;
                }
            };

            let tx_inner = tx_main.clone();
            let semaphore_inner = semaphore.clone();
            let cfg_inner = shared_cfg.clone();
            let producer_inner = producer.clone();
            let hardware_inner = hardware.clone();

            tokio::spawn(async move {
                let skill_hint = derive_skill_from_assignment(&assignment);
                let task_id = assignment.task_id.clone();
                let hardware_snapshot = hardware_inner.read().ok().map(|guard| guard.clone());
                let result = publish_assignment_to_topic(
                    &producer_inner,
                    &cfg_inner.output_topic,
                    &assignment,
                    skill_hint.as_deref(),
                    hardware_snapshot.as_ref(),
                )
                .await;
                if let Some(topic) = cfg_inner.rl_results_topic.as_ref() {
                    if topic != &cfg_inner.output_topic {
                        if let Err(err) = publish_assignment_to_topic(
                            &producer_inner,
                            topic,
                            &assignment,
                            skill_hint.as_deref(),
                            hardware_snapshot.as_ref(),
                        )
                        .await
                        {
                            warn!(%task_id, rl_topic = %topic, error = %err, "failed to publish rl sample");
                        }
                    }
                }
                match &result {
                    Ok(_) => info!(%task_id, transport = "kafka", "processed assignment"),
                    Err(err) => {
                        error!(%task_id, transport = "kafka", "processing failed: {err}")
                    }
                }

                let error_msg = result
                    .as_ref()
                    .err()
                    .map(|err| err.to_string())
                    .unwrap_or_default();

                if let Err(err) = tx_inner
                    .send(WorkerToSupervisor {
                        msg: Some(WorkerMsg::Ack(TaskAck {
                            task_id: task_id.clone(),
                            success: result.is_ok(),
                            error: error_msg,
                        })),
                    })
                    .await
                {
                    error!(%task_id, "failed to send ack: {err}");
                }

                drop(permit);
                let available = semaphore_inner.available_permits() as u32;
                if let Err(err) = tx_inner
                    .send(WorkerToSupervisor {
                        msg: Some(WorkerMsg::Credit(CreditReport {
                            worker_id: cfg_inner.worker_id.clone(),
                            credits: available,
                        })),
                    })
                    .await
                {
                    error!("failed to send credit update: {err}");
                }
            });
        }
    }

    info!(transport = "kafka", "supervisor stream closed");
    Ok(())
}

async fn run_mesh_worker(cfg: WorkerConfig, hardware: Arc<RwLock<HardwareSnapshot>>) -> Result<()> {
    let endpoint = cfg
        .mesh_endpoint
        .clone()
        .ok_or_else(|| anyhow::anyhow!("MESH_ENDPOINT must be set"))?;
    let domain = cfg
        .mesh_domain
        .clone()
        .unwrap_or_else(|| "default".to_string());
    let node_id = cfg
        .mesh_node_id
        .or_else(|| cfg.worker_id.parse::<u64>().ok())
        .unwrap_or_else(|| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            cfg.worker_id.hash(&mut hasher);
            hasher.finish()
        });

    info!(worker = %cfg.worker_id, transport = "mesh", %endpoint, %node_id, %domain, "starting mesh worker");

    let producer = Arc::new(create_producer(&cfg)?);
    let semaphore = Arc::new(Semaphore::new(cfg.max_inflight));

    let (tx, rx) = mpsc::channel::<WorkerToSupervisor>(cfg.max_inflight * 4);
    let outbound = ReceiverStream::new(rx);

    let mut supervisor_client = StreamSupervisorClient::connect(cfg.supervisor_addr.clone())
        .await
        .context("connect mesh supervisor control")?;
    let mut control_stream = supervisor_client
        .open_stream(outbound)
        .await
        .context("register mesh control stream")?
        .into_inner();

    let tx_control = tx.clone();
    tx_control
        .send(WorkerToSupervisor {
            msg: Some(WorkerMsg::Hello(WorkerHello {
                worker_id: cfg.worker_id.clone(),
                max_inflight: cfg.max_inflight as u32,
                version: env!("CARGO_PKG_VERSION").to_string(),
                mesh_enabled: true,
                mesh_node_id: node_id,
                mesh_domain: domain.clone(),
            })),
        })
        .await
        .ok();

    tx_control
        .send(WorkerToSupervisor {
            msg: Some(WorkerMsg::Credit(CreditReport {
                worker_id: cfg.worker_id.clone(),
                credits: cfg.max_inflight as u32,
            })),
        })
        .await
        .ok();

    tokio::spawn(async move {
        while let Some(msg) = control_stream.next().await {
            match msg {
                Ok(_payload) => {
                    // Reserved for future supervisor directives in mesh mode.
                }
                Err(err) => {
                    warn!(error = %err, "control stream closed");
                    break;
                }
            }
        }
    });

    let mut subscribe_client = MeshDataClient::connect(endpoint.clone())
        .await
        .context("connect mesh subscribe client")?;
    let ack_client = Arc::new(tokio::sync::Mutex::new(
        MeshDataClient::connect(endpoint.clone())
            .await
            .context("connect mesh ack client")?,
    ));

    let reddb_sink = if let Some(url) = cfg.reddb_url.as_ref() {
        match RedDBSink::new(
            url,
            &cfg.reddb_namespace,
            &cfg.reddb_stream,
            cfg.reddb_token.clone(),
        ) {
            Ok(sink) => {
                info!(worker = %cfg.worker_id, stream = %cfg.reddb_stream, "configured RedDB sink");
                Some(Arc::new(sink))
            }
            Err(err) => {
                warn!(worker = %cfg.worker_id, error = %err, "failed to configure RedDB sink; continuing without RedDB");
                None
            }
        }
    } else {
        None
    };

    let telemetry = Arc::new(WorkerTelemetry::new(cfg.max_inflight));
    let max_credits = cfg.max_inflight as u32;
    let credit_granularity = std::cmp::max(1, max_credits / 4);
    let context = Arc::new(ProcessingContext {
        producer: producer.clone(),
        output_topic: cfg.output_topic.clone(),
        ack_client: ack_client.clone(),
        node_id,
        worker_id: cfg.worker_id.clone(),
        semaphore: semaphore.clone(),
        tx_supervisor: tx.clone(),
        telemetry: telemetry.clone(),
        reddb: reddb_sink,
        rl_topic: cfg.rl_results_topic.clone(),
        hardware: hardware.clone(),
        max_credits,
        credit_granularity,
        last_credit: AtomicU32::new(max_credits),
    });

    let (sub_tx, sub_rx) = mpsc::channel::<SubscribeRequest>(1);
    sub_tx
        .send(SubscribeRequest {
            node_id,
            domain: domain.clone(),
        })
        .await
        .expect("subscribe request send");
    let _subscription_tx = sub_tx; // keep sender alive for stream lifecycle

    let mut inbound = subscribe_client
        .subscribe(tonic::Request::new(ReceiverStream::new(sub_rx)))
        .await
        .context("mesh subscribe")?
        .into_inner();

    while let Some(message) = inbound
        .message()
        .await
        .map_err(|status| anyhow!(status.to_string()))
        .context("mesh receive")?
    {
        context.clone().handle_received(message).await;
    }

    info!(transport = "mesh", "mesh subscription closed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_tracks_counts() {
        let telemetry = WorkerTelemetry::new(2);
        assert_eq!(telemetry.on_start(), 1);
        let (active_after, processed, failed) = telemetry.on_finish(true);
        assert_eq!(active_after, 0);
        assert_eq!(processed, 1);
        assert_eq!(failed, 0);

        telemetry.on_start();
        let (_, processed, failed) = telemetry.on_finish(false);
        assert_eq!(processed, 1);
        assert_eq!(failed, 1);
    }
}
