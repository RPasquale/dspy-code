use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::process::Command;
use tokio::time::timeout;
use tokio::time::Instant as TokioInstant;
use tracing::warn;

const STDOUT_TAIL_BYTES: usize = 4096;
const STDERR_TAIL_BYTES: usize = 4096;

#[derive(Debug, Deserialize, Clone, Default)]
pub struct TaskSpec {
    #[serde(default)]
    pub workflow_id: Option<String>,
    #[serde(default)]
    pub tenant: Option<String>,
    pub execution: Option<ExecutionSpec>,
    #[serde(default)]
    pub artifacts: Option<ArtifactsSpec>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ExecutionSpec {
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub mounts: Vec<MountSpec>,
    #[serde(default)]
    pub resources: Option<ResourceSpec>,
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
    #[serde(default)]
    pub working_dir: Option<String>,
    #[serde(default)]
    pub container_workdir: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ResourceSpec {
    #[serde(default)]
    pub cpu_millicores: Option<u64>,
    #[serde(default)]
    pub mem_bytes: Option<u64>,
    #[serde(default)]
    pub gpu_count: Option<u64>,
    #[serde(default)]
    pub gpu_class: Option<String>,
    #[serde(default)]
    pub gpu_memory_gb: Option<u64>,
    #[serde(default)]
    pub storage_gb: Option<u64>,
    #[serde(default)]
    pub working_dir: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct MountSpec {
    pub host_path: String,
    pub mount_path: String,
    #[serde(default)]
    pub read_only: bool,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ArtifactsSpec {
    #[serde(default)]
    pub inputs: Vec<ArtifactEntry>,
    #[serde(default)]
    pub outputs: Vec<ArtifactEntry>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ArtifactEntry {
    pub path: String,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug)]
pub struct ExecutionResult {
    pub success: bool,
    pub duration: Duration,
    pub metadata: Value,
    pub error: Option<String>,
}

pub async fn execute_task(task_id: &str, class: &str, payload: &Value) -> Result<ExecutionResult> {
    let spec: TaskSpec =
        serde_json::from_value(payload.clone()).context("parse execution payload")?;
    let execution = spec
        .execution
        .clone()
        .ok_or_else(|| anyhow!("payload.execution section is required"))?;
    if execution.command.is_empty() {
        bail!("payload.execution.command must contain at least one element");
    }

    let tenant = spec
        .tenant
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("default");

    let base_tasks = std::env::var("DSPY_TASK_BASE").unwrap_or_else(|_| "/tmp/dspy-tasks".into());
    let working_dir = execution
        .working_dir
        .clone()
        .unwrap_or_else(|| format!("{}/{}/{}", base_tasks, tenant, task_id));
    fs::create_dir_all(&working_dir)
        .await
        .with_context(|| format!("create working dir {}", working_dir))?;

    let log_root = std::env::var("DSPY_LOG_DIR").unwrap_or_else(|_| "/var/log/dspy/tasks".into());
    let log_dir = PathBuf::from(log_root).join(tenant);
    fs::create_dir_all(&log_dir)
        .await
        .with_context(|| format!("create log dir {}", log_dir.display()))?;
    let stdout_path = log_dir.join(format!("{task_id}-stdout.log"));
    let stderr_path = log_dir.join(format!("{task_id}-stderr.log"));

    for mount in execution.mounts.iter() {
        if let Err(err) = fs::create_dir_all(&mount.host_path).await {
            warn!(
                path = %mount.host_path,
                error = %err,
                "failed to ensure mount directory exists"
            );
        }
    }

    let start_wall = Instant::now();
    let start_ts: DateTime<Utc> = Utc::now();
    let mut command = build_command(task_id, &execution, &working_dir)?;

    command.envs(&execution.env);
    if execution.image.is_none() {
        command.current_dir(&working_dir);
    }
    let timeout_duration = Duration::from_secs(execution.timeout_seconds.unwrap_or(1800));
    let start_exec = TokioInstant::now();
    let output = match timeout(timeout_duration, command.output()).await {
        Ok(result) => result.context("spawn task process")?,
        Err(_) => {
            return Ok(build_timeout_result(
                task_id,
                class,
                &stdout_path,
                &stderr_path,
                &working_dir,
                &spec,
                start_ts,
                start_wall.elapsed(),
                timeout_duration,
            )
            .await?);
        }
    };
    let duration = start_exec.elapsed();
    let exit_code = output.status.code();

    fs::write(&stdout_path, &output.stdout)
        .await
        .with_context(|| format!("write stdout log {}", stdout_path.display()))?;
    fs::write(&stderr_path, &output.stderr)
        .await
        .with_context(|| format!("write stderr log {}", stderr_path.display()))?;

    let stdout_tail = tail_to_string(&output.stdout, STDOUT_TAIL_BYTES);
    let stderr_tail = tail_to_string(&output.stderr, STDERR_TAIL_BYTES);
    let runner_output = parse_result_marker(&output.stdout);

    let outputs_meta =
        collect_outputs_metadata(spec.artifacts.as_ref().map(|a| &a.outputs)).await?;
    let mut outputs_summary = Map::new();
    for entry in outputs_meta.iter() {
        if let Some(obj) = entry.as_object() {
            if let Some(path_str) = obj.get("path").and_then(|v| v.as_str()) {
                let path = Path::new(path_str);
                if path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|s| s.eq_ignore_ascii_case("json"))
                    .unwrap_or(false)
                {
                    match fs::read_to_string(path).await {
                        Ok(content) => {
                            if let Ok(parsed) = serde_json::from_str::<Value>(&content) {
                                let key = path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .map(|n| n.to_string())
                                    .unwrap_or_else(|| path_str.to_string());
                                outputs_summary.insert(key, parsed);
                            }
                        }
                        Err(err) => {
                            warn!(path = %path.display(), error = %err, "failed to read JSON artifact");
                        }
                    }
                }
            }
        }
    }

    let progress = json!([
        {
            "status": "started",
            "timestamp": start_ts.to_rfc3339(),
            "progress": 0
        },
        {
            "status": if output.status.success() { "completed" } else { "failed" },
            "timestamp": Utc::now().to_rfc3339(),
            "progress": 100
        }
    ]);

    let metadata = json!({
        "status": if output.status.success() { "completed" } else { "failed" },
        "exit_code": exit_code,
        "duration_ms": duration.as_millis(),
        "tenant": spec.tenant,
        "workflow_id": spec.workflow_id,
        "result": {
            "outputs": outputs_meta,
            "metrics": {
                "wall_clock_ms": duration.as_millis(),
                "task_class": class,
            },
            "outputs_summary": Value::Object(outputs_summary),
        },
        "logs": {
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        },
        "progress": progress,
        "working_dir": working_dir,
        "runner_output": runner_output,
    });

    Ok(ExecutionResult {
        success: output.status.success(),
        duration: start_wall.elapsed(),
        metadata,
        error: if output.status.success() {
            None
        } else {
            Some(String::from_utf8_lossy(&output.stderr).trim().to_string())
        },
    })
}

fn build_command(task_id: &str, execution: &ExecutionSpec, working_dir: &str) -> Result<Command> {
    if let Some(image) = execution.image.as_ref() {
        let docker_bin = std::env::var("DOCKER_BIN").unwrap_or_else(|_| "docker".into());
        let mut args: Vec<String> = vec![
            "run".into(),
            "--rm".into(),
            "--name".into(),
            format!("dspy-task-{}", task_id),
        ];

        if let Some(resources) = execution.resources.as_ref() {
            if let Some(cpu) = resources.cpu_millicores {
                let cpus = (cpu as f64 / 1000.0).max(0.1);
                args.push("--cpus".into());
                args.push(format!("{:.3}", cpus));
            }
            if let Some(mem) = resources.mem_bytes {
                args.push("--memory".into());
                args.push(format!("{}b", mem));
            }
            if let Some(gpu) = resources.gpu_count {
                if gpu > 0 {
                    args.push("--gpus".into());
                    args.push(format!(
                        "device={}",
                        std::env::var("DSPY_GPU_DEVICE").unwrap_or_else(|_| "0".into())
                    ));
                }
            }
        }

        for mount in execution.mounts.iter() {
            let mut spec = format!("{}:{}", mount.host_path, mount.mount_path);
            if mount.read_only {
                spec.push_str(":ro");
            }
            args.push("-v".into());
            args.push(spec);
        }

        for (key, value) in execution.env.iter() {
            args.push("-e".into());
            args.push(format!("{}={}", key, value));
        }

        if let Some(container_dir) = execution
            .container_workdir
            .clone()
            .or_else(|| execution.working_dir.clone())
        {
            args.push("--workdir".into());
            args.push(container_dir);
        }

        args.push(image.clone());
        args.extend(execution.command.iter().cloned());

        let mut command = Command::new(docker_bin);
        command.args(args);
        command.current_dir(working_dir);
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        Ok(command)
    } else {
        let mut iter = execution.command.iter();
        let program = iter
            .next()
            .ok_or_else(|| anyhow!("execution.command requires at least one element"))?;
        let mut command = Command::new(program);
        command.args(iter.cloned());
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        Ok(command)
    }
}

async fn build_timeout_result(
    task_id: &str,
    class: &str,
    stdout_path: &Path,
    stderr_path: &Path,
    working_dir: &str,
    spec: &TaskSpec,
    start_ts: DateTime<Utc>,
    duration: Duration,
    timeout: Duration,
) -> Result<ExecutionResult> {
    let message = format!(
        "task {task_id} timed out after {} seconds",
        timeout.as_secs()
    );
    fs::write(stdout_path, b"").await.ok();
    fs::write(stderr_path, message.as_bytes()).await.ok();

    let metadata = json!({
        "status": "failed",
        "exit_code": null,
        "duration_ms": duration.as_millis(),
        "tenant": spec.tenant,
        "workflow_id": spec.workflow_id,
        "result": {
            "outputs": [],
            "metrics": {
                "wall_clock_ms": duration.as_millis(),
                "task_class": class,
            }
        },
        "logs": {
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "stdout_tail": "",
            "stderr_tail": message,
        },
        "progress": [
            { "status": "started", "timestamp": start_ts.to_rfc3339(), "progress": 0 },
            { "status": "failed", "timestamp": Utc::now().to_rfc3339(), "progress": 100 }
        ],
        "working_dir": working_dir,
        "timeout_seconds": timeout.as_secs(),
    });

    Ok(ExecutionResult {
        success: false,
        duration,
        metadata,
        error: Some(message),
    })
}

async fn collect_outputs_metadata(outputs: Option<&Vec<ArtifactEntry>>) -> Result<Vec<Value>> {
    let mut collected = Vec::new();
    if let Some(list) = outputs {
        for entry in list.iter() {
            let path = Path::new(&entry.path);
            let metadata = match fs::metadata(path).await {
                Ok(meta) if meta.is_file() => meta,
                _ => continue,
            };
            let size = metadata.len();
            let checksum = compute_checksum(path).await?;
            collected.push(json!({
                "path": entry.path,
                "type": entry.r#type,
                "name": entry.name,
                "size_bytes": size,
                "checksum": format!("sha256:{checksum}")
            }));
        }
    }
    Ok(collected)
}

async fn compute_checksum(path: &Path) -> Result<String> {
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || -> Result<String> {
        let mut file = std::fs::File::open(&path)
            .with_context(|| format!("open output file {}", path.display()))?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        Ok(hex::encode(hasher.finalize()))
    })
    .await
    .context("join checksum task")?
}

fn tail_to_string(data: &[u8], max_len: usize) -> String {
    if data.len() <= max_len {
        return String::from_utf8_lossy(data).to_string();
    }
    let tail = &data[data.len() - max_len..];
    String::from_utf8_lossy(tail).to_string()
}

fn parse_result_marker(stdout: &[u8]) -> Option<Value> {
    let text = String::from_utf8_lossy(stdout);
    let marker = "__DSPY_RESULT__:";
    for line in text.lines().rev() {
        let trimmed = line.trim();
        if trimmed.starts_with(marker) {
            let payload = trimmed[marker.len()..].trim();
            if payload.is_empty() {
                return None;
            }
            return serde_json::from_str(payload).ok();
        }
    }
    None
}
