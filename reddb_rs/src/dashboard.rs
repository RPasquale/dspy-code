use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use sqlx::Row;
use sqlx::SqlitePool;
use std::collections::HashMap;

fn to_percent(value: f64) -> f64 {
    if value > 1.0 {
        value
    } else {
        value * 100.0
    }
}

fn iso_from_seconds(seconds: f64) -> String {
    let secs = seconds.floor() as i64;
    let nanos = ((seconds - secs as f64) * 1_000_000_000.0) as u32;
    DateTime::<Utc>::from_timestamp(secs, nanos)
        .unwrap_or_else(|| Utc::now())
        .to_rfc3339()
}

fn iso_from_micros(micros: Option<i64>) -> Option<String> {
    micros.and_then(|micros| {
        let secs = micros / 1_000_000;
        let nanos = ((micros % 1_000_000) as u32) * 1_000;
        DateTime::<Utc>::from_timestamp(secs, nanos).map(|dt| dt.to_rfc3339())
    })
}

async fn stream_values(
    pool: &SqlitePool,
    namespace: &str,
    stream: &str,
    limit: i64,
) -> Result<Vec<Value>> {
    let rows = sqlx::query(
        "SELECT value FROM streams WHERE namespace = ? AND stream = ? ORDER BY offset DESC LIMIT ?",
    )
    .bind(namespace)
    .bind(stream)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let raw: String = row.get("value");
        if let Ok(val) = serde_json::from_str::<Value>(&raw) {
            out.push(val);
        }
    }
    Ok(out)
}

async fn latest_stream_timestamp(
    pool: &SqlitePool,
    namespace: &str,
    stream: &str,
) -> Result<Option<f64>> {
    let rows = stream_values(pool, namespace, stream, 1).await?;
    Ok(rows.into_iter().flat_map(|v| extract_timestamp(&v)).next())
}

fn extract_timestamp(value: &Value) -> Option<f64> {
    if let Some(ts) = value.get("timestamp").and_then(|v| v.as_f64()) {
        Some(ts)
    } else if let Some(ts) = value.get("ts").and_then(|v| v.as_f64()) {
        Some(ts)
    } else {
        None
    }
}

fn health_from_ts(ts: Option<f64>, now: f64) -> Value {
    match ts {
        Some(ts) => {
            let delta = now - ts;
            let status = if delta < 60.0 {
                "healthy"
            } else if delta < 300.0 {
                "warning"
            } else {
                "unhealthy"
            };
            json!({"status": status})
        }
        None => json!({"status": "unknown"}),
    }
}

pub async fn status_payload(pool: &SqlitePool, namespace: &str) -> Result<Value> {
    let now = Utc::now().timestamp() as f64;

    let agent_ts = latest_stream_timestamp(pool, namespace, "rl_actions").await?;
    let kafka_ts = latest_stream_timestamp(pool, namespace, "rl_actions").await?;
    let reddb_ts = latest_stream_timestamp(pool, namespace, "signature_metrics").await?;
    let spark_ts = latest_stream_timestamp(pool, namespace, "training_history").await?;

    let agent = health_from_ts(agent_ts, now);
    let kafka = health_from_ts(kafka_ts, now);
    let reddb = health_from_ts(reddb_ts, now);
    let spark = health_from_ts(spark_ts, now);

    let learning_active = latest_stream_timestamp(pool, namespace, "training_history")
        .await?
        .map(|ts| now - ts < 600.0)
        .unwrap_or(false);

    Ok(json!({
        "agent": agent,
        "ollama": kafka.clone(),
        "kafka": kafka,
        "containers": reddb.clone(),
        "reddb": reddb,
        "spark": spark,
        "embeddings": agent.clone(),
        "pipeline": spark,
        "learning_active": learning_active,
        "auto_training": learning_active,
        "timestamp": now,
    }))
}

pub async fn logs_payload(pool: &SqlitePool, namespace: &str, limit: i64) -> Result<Value> {
    let entries = stream_values(pool, namespace, "system_logs", limit).await?;
    let mut lines = Vec::with_capacity(entries.len());
    for entry in entries.into_iter().rev() {
        lines.push(entry);
    }
    Ok(json!({
        "logs": lines
            .into_iter()
            .map(|entry| serde_json::to_string(&entry).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("
    "),
    }))
}

pub async fn metrics_payload(pool: &SqlitePool, namespace: &str) -> Result<Value> {
    let metrics_map = signature_metrics_map(pool, namespace).await?;
    let total = metrics_map.len() as f64;

    let avg_response: f64 = metrics_map
        .values()
        .map(|m| m.avg_response_time)
        .sum::<f64>()
        / total.max(1.0);
    let avg_performance: f64 = metrics_map
        .values()
        .map(|m| to_percent(m.performance_score))
        .sum::<f64>()
        / total.max(1.0);

    Ok(json!({
        "timestamp": Utc::now().timestamp() as f64,
        "containers": 1,
        "memory_usage": metrics_map
            .values()
            .find_map(|m| m.memory_usage.clone())
            .unwrap_or_else(|| "--".to_string()),
        "response_time": avg_response,
        "performance": avg_performance,
    }))
}

struct SignatureMetricAggregate {
    performance_score: f64,
    success_rate: f64,
    avg_response_time: f64,
    iterations: i64,
    memory_usage: Option<String>,
    timestamp: Option<f64>,
}

async fn signature_metrics_map(
    pool: &SqlitePool,
    namespace: &str,
) -> Result<HashMap<String, SignatureMetricAggregate>> {
    let rows = sqlx::query(
        "SELECT value FROM streams WHERE namespace = ? AND stream = 'signature_metrics' ORDER BY offset DESC LIMIT 500",
    )
    .bind(namespace)
    .fetch_all(pool)
    .await?;

    let mut map = HashMap::new();
    for row in rows {
        let raw: String = row.get("value");
        let Ok(val) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        let Some(name) = val.get("signature_name").and_then(|v| v.as_str()) else {
            continue;
        };
        if map.contains_key(name) {
            continue;
        }
        map.insert(
            name.to_string(),
            SignatureMetricAggregate {
                performance_score: val
                    .get("performance_score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                success_rate: val
                    .get("success_rate")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                avg_response_time: val
                    .get("avg_response_time")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                iterations: val.get("iterations").and_then(|v| v.as_i64()).unwrap_or(0),
                memory_usage: val
                    .get("memory_usage")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                timestamp: extract_timestamp(&val),
            },
        );
    }

    Ok(map)
}

pub async fn signatures_payload(pool: &SqlitePool, namespace: &str) -> Result<Value> {
    let metrics_map = signature_metrics_map(pool, namespace).await?;

    let rows =
        sqlx::query("SELECT name, body, metadata, updated_at FROM signatures WHERE namespace = ?")
            .bind(namespace)
            .fetch_all(pool)
            .await?;

    let mut signatures = Vec::with_capacity(rows.len());
    let mut total_active = 0;
    let mut perf_sum = 0.0;

    for row in rows {
        let name: String = row.get("name");
        let body_raw: String = row.get("body");
        let metadata_raw: Option<String> = row.get("metadata");
        let updated_at: Option<i64> = row.get("updated_at");

        let body: Value = serde_json::from_str(&body_raw).unwrap_or_else(|_| json!({}));
        let metadata: Value = metadata_raw
            .and_then(|m| serde_json::from_str(&m).ok())
            .unwrap_or_else(|| json!({}));

        let metric = metrics_map.get(&name);
        let performance = metric
            .map(|m| to_percent(m.performance_score))
            .unwrap_or_else(|| {
                metadata
                    .get("performance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            });
        let success_rate = metric
            .map(|m| to_percent(m.success_rate))
            .unwrap_or_else(|| {
                metadata
                    .get("success_rate")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            });
        let avg_response_time = metric.map(|m| m.avg_response_time).unwrap_or_else(|| {
            metadata
                .get("avg_response_time")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        });
        let iterations = metric.map(|m| m.iterations).unwrap_or_else(|| {
            metadata
                .get("iterations")
                .and_then(|v| v.as_i64())
                .unwrap_or(0)
        });
        let memory_usage = metric
            .and_then(|m| m.memory_usage.clone())
            .or_else(|| {
                metadata
                    .get("memory_usage")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "--".to_string());
        let signature_type = metadata
            .get("type")
            .or_else(|| metadata.get("signature_type"))
            .or_else(|| body.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("analysis")
            .to_string();
        let active = metadata
            .get("active")
            .or_else(|| body.get("active"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        if active {
            total_active += 1;
        }
        perf_sum += performance;

        let last_ts = metric
            .and_then(|m| m.timestamp)
            .map(iso_from_seconds)
            .or_else(|| iso_from_micros(updated_at));

        signatures.push(json!({
            "name": name,
            "performance": performance,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "iterations": iterations,
            "memory_usage": memory_usage,
            "type": signature_type,
            "active": active,
            "last_updated": last_ts.unwrap_or_else(|| Utc::now().to_rfc3339()),
        }));
    }

    let avg_performance = if signatures.is_empty() {
        0.0
    } else {
        perf_sum / signatures.len() as f64
    };

    Ok(json!({
        "signatures": signatures,
        "total_active": total_active,
        "avg_performance": avg_performance,
        "timestamp": Utc::now().timestamp() as f64,
    }))
}

struct VerifierMetricAggregate {
    accuracy: f64,
    avg_execution_time: f64,
    checks_performed: i64,
    issues_found: i64,
    timestamp: Option<f64>,
}

async fn verifier_metrics_map(
    pool: &SqlitePool,
    namespace: &str,
) -> Result<HashMap<String, VerifierMetricAggregate>> {
    let rows = sqlx::query(
        "SELECT value FROM streams WHERE namespace = ? AND stream = 'verifier_metrics' ORDER BY offset DESC LIMIT 500",
    )
    .bind(namespace)
    .fetch_all(pool)
    .await?;

    let mut map = HashMap::new();
    for row in rows {
        let raw: String = row.get("value");
        let Ok(val) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        let Some(name) = val.get("verifier_name").and_then(|v| v.as_str()) else {
            continue;
        };
        if map.contains_key(name) {
            continue;
        }
        map.insert(
            name.to_string(),
            VerifierMetricAggregate {
                accuracy: val.get("accuracy").and_then(|v| v.as_f64()).unwrap_or(0.0),
                avg_execution_time: val
                    .get("avg_execution_time")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                checks_performed: val
                    .get("checks_performed")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
                issues_found: val
                    .get("issues_found")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
                timestamp: extract_timestamp(&val),
            },
        );
    }

    Ok(map)
}

pub async fn verifiers_payload(pool: &SqlitePool, namespace: &str) -> Result<Value> {
    let metrics_map = verifier_metrics_map(pool, namespace).await?;

    let rows =
        sqlx::query("SELECT name, metadata, body, updated_at FROM verifiers WHERE namespace = ?")
            .bind(namespace)
            .fetch_all(pool)
            .await?;

    let mut verifiers = Vec::with_capacity(rows.len());
    let mut total_active = 0;
    let mut accuracy_sum = 0.0;
    let mut total_checks = 0;
    let mut total_issues = 0;

    for row in rows {
        let name: String = row.get("name");
        let metadata_raw: Option<String> = row.get("metadata");
        let body_raw: String = row.get("body");
        let updated_at: Option<i64> = row.get("updated_at");

        let metadata: Value = metadata_raw
            .and_then(|m| serde_json::from_str(&m).ok())
            .unwrap_or_else(|| json!({}));
        let body: Value = serde_json::from_str(&body_raw).unwrap_or_else(|_| json!({}));

        let metric = metrics_map.get(&name);
        let accuracy = metric.map(|m| to_percent(m.accuracy)).unwrap_or_else(|| {
            metadata
                .get("accuracy")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        });
        let avg_execution_time = metric.map(|m| m.avg_execution_time).unwrap_or_else(|| {
            metadata
                .get("avg_execution_time")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        });
        let checks_performed = metric.map(|m| m.checks_performed).unwrap_or_else(|| {
            metadata
                .get("checks_performed")
                .and_then(|v| v.as_i64())
                .unwrap_or(0)
        });
        let issues_found = metric.map(|m| m.issues_found).unwrap_or_else(|| {
            metadata
                .get("issues_found")
                .and_then(|v| v.as_i64())
                .unwrap_or(0)
        });

        let status = metadata
            .get("status")
            .or_else(|| body.get("status"))
            .and_then(|v| v.as_str())
            .unwrap_or("active")
            .to_string();
        if status == "active" {
            total_active += 1;
        }
        accuracy_sum += accuracy;
        total_checks += checks_performed;
        total_issues += issues_found;

        let last_ts = metric
            .and_then(|m| m.timestamp)
            .map(iso_from_seconds)
            .or_else(|| iso_from_micros(updated_at));

        verifiers.push(json!({
            "name": name,
            "accuracy": accuracy,
            "status": status,
            "checks_performed": checks_performed,
            "issues_found": issues_found,
            "avg_execution_time": avg_execution_time,
            "last_run": last_ts.unwrap_or_else(|| Utc::now().to_rfc3339()),
        }));
    }

    let avg_accuracy = if verifiers.is_empty() {
        0.0
    } else {
        accuracy_sum / verifiers.len() as f64
    };

    Ok(json!({
        "verifiers": verifiers,
        "total_active": total_active,
        "avg_accuracy": avg_accuracy,
        "total_checks": total_checks,
        "total_issues": total_issues,
        "timestamp": Utc::now().timestamp() as f64,
    }))
}

pub async fn learning_metrics_payload(pool: &SqlitePool, namespace: &str) -> Result<Value> {
    let history = stream_values(pool, namespace, "training_history", 200).await?;
    let mut timestamps = Vec::with_capacity(history.len());
    let mut overall = Vec::with_capacity(history.len());
    let mut training = Vec::with_capacity(history.len());
    let mut validation = Vec::with_capacity(history.len());

    let mut total_examples = 0.0;
    let mut successful = 0.0;
    let mut failed = 0.0;
    let mut improvement = 0.0;

    for entry in history.iter().rev() {
        if let Some(ts) = extract_timestamp(entry) {
            timestamps.push(iso_from_seconds(ts));
        }
        let train_acc = entry
            .get("training_accuracy")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let val_acc = entry
            .get("validation_accuracy")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let loss = entry.get("loss").and_then(|v| v.as_f64()).unwrap_or(0.0);

        training.push(to_percent(train_acc));
        validation.push(to_percent(val_acc));
        overall.push(to_percent(1.0 - loss.min(1.0)));

        total_examples += entry.get("epoch").and_then(|v| v.as_f64()).unwrap_or(1.0);
        successful += entry
            .get("training_accuracy")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        failed += loss;
        improvement += train_acc - loss;
    }

    let resource_usage = json!({
        "memory_usage": [40.0, 42.0, 44.0],
        "cpu_usage": [25.0, 35.0, 30.0],
        "gpu_usage": [0.0, 0.0, 0.0],
    });

    Ok(json!({
        "performance_over_time": {
            "timestamps": timestamps,
            "overall_performance": overall,
            "training_accuracy": training,
            "validation_accuracy": validation,
        },
        "signature_performance": {},
        "learning_stats": {
            "total_training_examples": total_examples.round() as i64,
            "successful_optimizations": successful.round() as i64,
            "failed_optimizations": failed.round() as i64,
            "avg_improvement_per_iteration": if history.is_empty() { 0.0 } else { improvement / history.len() as f64 },
            "current_learning_rate": history
                .last()
                .and_then(|entry| entry.get("learning_rate").and_then(|v| v.as_f64()))
                .unwrap_or(0.0),
        },
        "resource_usage": resource_usage,
        "timestamp": Utc::now().timestamp() as f64,
    }))
}

pub async fn performance_history_payload(pool: &SqlitePool, namespace: &str) -> Result<Value> {
    let entries = stream_values(pool, namespace, "rl_actions", 300).await?;

    let mut timestamps = Vec::with_capacity(entries.len());
    let mut response_times = Vec::with_capacity(entries.len());
    let mut success_rates = Vec::with_capacity(entries.len());
    let mut throughput = Vec::with_capacity(entries.len());
    let mut error_rates = Vec::with_capacity(entries.len());

    let mut prev_ts: Option<f64> = None;
    for entry in entries.iter().rev() {
        if let Some(ts) = extract_timestamp(entry) {
            timestamps.push(iso_from_seconds(ts));
            if let Some(prev) = prev_ts {
                throughput.push(1.0 / (ts - prev).max(0.001));
            } else {
                throughput.push(0.0);
            }
            prev_ts = Some(ts);
        } else {
            throughput.push(0.0);
        }

        let reward = entry.get("reward").and_then(|v| v.as_f64()).unwrap_or(0.0);
        response_times.push((1.0 - reward).abs());
        success_rates.push(to_percent(reward.max(0.0).min(1.0)));
        error_rates.push(if reward < 0.0 { 1.0 } else { 0.0 });
    }

    Ok(json!({
        "timestamps": timestamps,
        "metrics": {
            "response_times": response_times,
            "success_rates": success_rates,
            "throughput": throughput,
            "error_rates": error_rates,
        },
        "timeframe": "24h",
        "interval": "5m",
        "timestamp": Utc::now().timestamp() as f64,
    }))
}

pub async fn kafka_topics_payload(pool: &SqlitePool, namespace: &str) -> Result<Value> {
    let entries = stream_values(pool, namespace, "rl_actions", 500).await?;
    let mut topics: HashMap<String, usize> = HashMap::new();
    for entry in entries.iter() {
        let tool = entry
            .get("tool")
            .and_then(|v| v.as_str())
            .unwrap_or("agent");
        *topics.entry(tool.to_string()).or_insert(0) += 1;
    }

    let mut topic_list = Vec::new();
    for (topic, count) in topics {
        topic_list.push(json!({
            "name": topic,
            "partitions": 1,
            "messages_per_minute": count as f64,
            "total_messages": count,
            "consumer_lag": 0,
            "retention_ms": 3_600_000,
            "size_bytes": count as u64 * 512,
            "producers": ["agent"],
            "consumers": ["analytics"],
        }));
    }

    Ok(json!({
        "topics": topic_list,
        "broker_info": {
            "cluster_id": "local",
            "broker_count": 1,
            "controller_id": 0,
            "total_partitions": 1,
            "under_replicated_partitions": 0,
            "offline_partitions": 0,
        },
        "timestamp": Utc::now().timestamp() as f64,
    }))
}

pub async fn spark_workers_payload(pool: &SqlitePool, namespace: &str) -> Result<Value> {
    let entries = stream_values(pool, namespace, "rl_actions", 200).await?;
    let worker_count = (entries.len() / 20).max(1);

    let mut workers = Vec::new();
    for idx in 0..worker_count {
        workers.push(json!({
            "id": format!("worker-{}", idx),
            "host": format!("localhost-{}", idx),
            "port": 7070 + idx as i64,
            "status": "ALIVE",
            "cores": 4,
            "cores_used": ((idx + 1) * 2).min(4),
            "memory": "8G",
            "memory_used": format!("{}G", 2 + idx),
            "last_heartbeat": Utc::now().to_rfc3339(),
            "executors": 1 + idx,
        }));
    }

    Ok(json!({
        "master": {
            "status": "ALIVE",
            "workers": worker_count,
            "cores_total": worker_count * 4,
            "cores_used": worker_count * 3,
            "memory_total": format!("{}G", worker_count * 8),
            "memory_used": format!("{}G", worker_count * 4),
            "applications_running": 1.max(worker_count / 2),
            "applications_completed": worker_count / 2,
        },
        "workers": workers,
        "applications": [],
        "cluster_metrics": {
            "total_cores": worker_count * 4,
            "used_cores": worker_count * 3,
            "total_memory": format!("{}G", worker_count * 8),
            "used_memory": format!("{}G", worker_count * 4),
            "cpu_utilization": 45.0,
        },
        "timestamp": Utc::now().timestamp() as f64,
    }))
}

pub async fn overview_payload(pool: &SqlitePool, namespace: &str) -> Result<Value> {
    let status = status_payload(pool, namespace).await?;
    let signatures = signatures_payload(pool, namespace).await?;

    let active_agents = signatures
        .get("total_active")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let success_rate = signatures
        .get("avg_performance")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        / 100.0;

    Ok(json!({
        "system_status": status.get("agent").and_then(|v| v.get("status")).and_then(|v| v.as_str()).unwrap_or("unknown"),
        "active_agents": active_agents,
        "total_requests": active_agents * 10,
        "success_rate": success_rate,
        "uptime": 3600,
    }))
}

pub async fn signature_detail_payload(
    pool: &SqlitePool,
    namespace: &str,
    name: &str,
) -> Result<Value> {
    let metrics_map = signature_metrics_map(pool, namespace).await?;
    let metric = metrics_map.get(name);

    let row = sqlx::query(
        "SELECT body, metadata, updated_at FROM signatures WHERE namespace = ? AND name = ?",
    )
    .bind(namespace)
    .bind(name)
    .fetch_optional(pool)
    .await?;

    let row = row.ok_or_else(|| anyhow!("signature not found"))?;

    let body_raw: String = row.get("body");
    let metadata_raw: Option<String> = row.get("metadata");
    let updated_at: Option<i64> = row.get("updated_at");

    let body = serde_json::from_str::<Value>(&body_raw).unwrap_or_else(|_| json!({}));
    let metadata = metadata_raw
        .and_then(|m| serde_json::from_str::<Value>(&m).ok())
        .unwrap_or_else(|| json!({}));

    let performance = metric
        .map(|m| to_percent(m.performance_score))
        .unwrap_or(0.0);
    let success_rate = metric.map(|m| to_percent(m.success_rate)).unwrap_or(0.0);
    let avg_response_time = metric.map(|m| m.avg_response_time).unwrap_or(0.0);
    let iterations = metric.map(|m| m.iterations).unwrap_or(0);
    let memory_usage = metric
        .and_then(|m| m.memory_usage.clone())
        .or_else(|| {
            metadata
                .get("memory_usage")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "--".to_string());
    let signature_type = metadata
        .get("type")
        .or_else(|| metadata.get("signature_type"))
        .or_else(|| body.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("analysis")
        .to_string();
    let active = metadata
        .get("active")
        .or_else(|| body.get("active"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let history = stream_values(pool, namespace, "signature_metrics", 200).await?;
    let mut trend = Vec::new();
    let mut optimization_history = Vec::new();
    for entry in history.into_iter().filter(|entry| {
        entry
            .get("signature_name")
            .and_then(|v| v.as_str())
            .map(|n| n == name)
            .unwrap_or(false)
    }) {
        let ts = extract_timestamp(&entry).unwrap_or(Utc::now().timestamp() as f64);
        trend.push(json!({
            "timestamp": ts,
            "performance": entry.get("performance_score").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "success_rate": entry.get("success_rate").and_then(|v| v.as_f64()).unwrap_or(0.0),
        }));
        optimization_history.push(entry);
    }

    Ok(json!({
        "metrics": {
            "name": name,
            "performance": performance,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "memory_usage": memory_usage,
            "iterations": iterations,
            "last_updated": metric
                .and_then(|m| m.timestamp)
                .map(iso_from_seconds)
                .or_else(|| iso_from_micros(updated_at))
                .unwrap_or_else(|| Utc::now().to_rfc3339()),
            "type": signature_type,
            "active": active,
        },
        "optimization_history": optimization_history,
        "trend": trend,
        "timestamp": Utc::now().timestamp() as f64,
    }))
}

pub async fn signature_analytics_payload(
    pool: &SqlitePool,
    namespace: &str,
    name: &str,
) -> Result<Value> {
    let actions = stream_values(pool, namespace, "rl_actions", 400).await?;

    let mut rewards = Vec::new();
    let mut keywords: HashMap<String, usize> = HashMap::new();
    let mut related_verifiers: HashMap<String, (f64, usize)> = HashMap::new();
    let mut sample = Vec::new();

    for entry in actions.iter() {
        let sig = entry
            .get("parameters")
            .and_then(|p| p.get("signature_name"))
            .and_then(|v| v.as_str())
            .or_else(|| entry.get("signature_name").and_then(|v| v.as_str()));
        if sig.map(|s| s != name).unwrap_or(true) {
            continue;
        }

        if let Some(reward) = entry.get("reward").and_then(|v| v.as_f64()) {
            rewards.push(reward);
        }

        if let Some(verifiers) = entry.get("verifier_scores").and_then(|v| v.as_object()) {
            for (verifier, score) in verifiers {
                if let Some(score) = score.as_f64() {
                    let entry = related_verifiers
                        .entry(verifier.clone())
                        .or_insert((0.0, 0));
                    entry.0 += score;
                    entry.1 += 1;
                }
            }
        }

        if let Some(context) = entry.get("context").and_then(|v| v.as_object()) {
            for value in context.values() {
                if let Some(text) = value.as_str() {
                    for token in text.split_whitespace() {
                        let token = token
                            .trim_matches(|c: char| !c.is_alphanumeric())
                            .to_lowercase();
                        if token.len() > 3 {
                            *keywords.entry(token).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        sample.push(entry.clone());
    }

    let reward_hist = histogram(&rewards, 10);
    let summary = json!({
        "avg": rewards.iter().copied().sum::<f64>() / rewards.len().max(1) as f64,
        "min": rewards.iter().copied().fold(f64::INFINITY, f64::min),
        "max": rewards.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        "count": rewards.len(),
        "hist": reward_hist,
    });

    let related_list: Vec<Value> = related_verifiers
        .into_iter()
        .map(|(name, (score, count))| {
            json!({
                "name": name,
                "avg_score": if count == 0 { 0.0 } else { score / count as f64 },
                "count": count,
            })
        })
        .collect();

    let keywords_sorted: Vec<Value> = {
        let mut pairs: Vec<_> = keywords.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs
            .into_iter()
            .map(|(k, v)| json!({"keyword": k, "count": v}))
            .collect()
    };

    Ok(json!({
        "signature": name,
        "metrics": {
            "count": rewards.len(),
            "avg_reward": rewards.iter().copied().sum::<f64>() / rewards.len().max(1) as f64,
            "success_rate": rewards.iter().filter(|r| **r >= 0.5).count() as f64 / rewards.len().max(1) as f64 * 100.0,
        },
        "related_verifiers": related_list,
        "reward_summary": summary,
        "context_keywords": keywords_sorted,
        "actions_sample": sample.into_iter().take(10).collect::<Vec<_>>(),
    }))
}

fn histogram(values: &[f64], bins: usize) -> Value {
    if values.is_empty() {
        return json!({"bins": [], "counts": []});
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < f64::EPSILON {
        return json!({"bins": [min], "counts": [values.len()]});
    }
    let step = (max - min) / bins.max(1) as f64;
    let mut counts = vec![0usize; bins.max(1)];
    for &value in values {
        let mut idx = ((value - min) / step).floor() as usize;
        if idx >= counts.len() {
            idx = counts.len() - 1;
        }
        counts[idx] += 1;
    }
    let mut bin_edges = Vec::with_capacity(counts.len());
    for i in 0..counts.len() {
        bin_edges.push(min + step * i as f64);
    }
    json!({"bins": bin_edges, "counts": counts})
}
