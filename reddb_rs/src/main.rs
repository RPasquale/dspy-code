use std::env;
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use axum_extra::headers::{authorization::Bearer, Authorization};
use axum_extra::TypedHeader;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::{Row, SqlitePool};
use thiserror::Error;
use tower_http::trace::TraceLayer;
use tracing::{error, info};

mod dashboard;

#[derive(Clone)]
struct AppState {
    pool: SqlitePool,
    admin_token: Option<String>,
    data_dir: Arc<PathBuf>,
    default_namespace: String,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("unauthorized")]
    Unauthorized,
    #[error("not found")]
    NotFound,
    #[error("{0}")]
    BadRequest(String),
    #[error("invalid json: {0}")]
    InvalidJson(String),
    #[error("database error: {0}")]
    Database(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::Unauthorized => (StatusCode::UNAUTHORIZED, self.to_string()),
            AppError::NotFound => (StatusCode::NOT_FOUND, self.to_string()),
            AppError::BadRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            AppError::InvalidJson(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            AppError::Database(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };
        let body = Json(json!({
            "ok": false,
            "error": message,
        }));
        (status, body).into_response()
    }
}

type AppResult<T> = std::result::Result<T, AppError>;

fn check_authorization(state: &AppState, token: Option<&str>) -> AppResult<()> {
    if let Some(expected) = state.admin_token.as_deref() {
        if token != Some(expected) {
            return Err(AppError::Unauthorized);
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct KvResponse {
    ok: bool,
    namespace: String,
    key: String,
}

#[derive(Debug, Deserialize)]
struct NamespaceQuery {
    namespace: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LogsQuery {
    namespace: Option<String>,
    #[serde(default)]
    limit: Option<i64>,
}

fn resolve_namespace(state: &AppState, namespace: Option<String>) -> String {
    namespace.unwrap_or_else(|| state.default_namespace.clone())
}

#[derive(Serialize)]
struct StreamAppendResponse {
    ok: bool,
    namespace: String,
    stream: String,
    offset: i64,
}

#[derive(Debug, Deserialize)]
struct StreamQuery {
    #[serde(default)]
    start: i64,
    #[serde(default = "default_count")]
    count: i64,
}

fn default_count() -> i64 {
    100
}

#[derive(Serialize)]
struct StreamItem {
    offset: i64,
    value: Value,
}

fn empty_object() -> Value {
    Value::Object(Default::default())
}

#[derive(Debug, Deserialize)]
struct SignaturePayload {
    name: String,
    #[serde(default)]
    namespace: Option<String>,
    body: Value,
    #[serde(default = "empty_object")]
    metadata: Value,
    #[serde(default)]
    verifiers: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct VerifierPayload {
    name: String,
    #[serde(default)]
    namespace: Option<String>,
    body: Value,
    #[serde(default = "empty_object")]
    metadata: Value,
}

#[derive(Serialize)]
struct GraphNode {
    id: String,
    kind: String,
    label: String,
    namespace: String,
    metadata: Value,
}

#[derive(Serialize)]
struct GraphEdge {
    from: String,
    to: String,
    kind: String,
}

#[derive(Serialize)]
struct GraphResponse {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
}

async fn health(State(state): State<AppState>) -> AppResult<Json<Value>> {
    Ok(Json(json!({
        "ok": true,
        "service": "reddb",
        "version": "1.0.0",
        "timestamp": Utc::now().timestamp_millis() as f64 / 1000.0,
        "data_dir": state.data_dir.display().to_string(),
        "auth_enabled": state.admin_token.is_some(),
    })))
}

async fn put_kv(
    State(state): State<AppState>,
    AxumPath((namespace, key)): AxumPath<(String, String)>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Json(body): Json<Value>,
) -> AppResult<Json<KvResponse>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let payload =
        serde_json::to_string(&body).map_err(|err| AppError::InvalidJson(err.to_string()))?;
    let now = Utc::now().timestamp_micros();

    sqlx::query(
        "INSERT INTO kv_store (namespace, key, value, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(namespace, key)
         DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
    )
    .bind(&namespace)
    .bind(&key)
    .bind(payload)
    .bind(now)
    .bind(now)
    .execute(&state.pool)
    .await
    .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(KvResponse {
        ok: true,
        namespace,
        key,
    }))
}

async fn get_kv(
    State(state): State<AppState>,
    AxumPath((namespace, key)): AxumPath<(String, String)>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let row: Option<String> =
        sqlx::query_scalar("SELECT value FROM kv_store WHERE namespace = ? AND key = ?")
            .bind(&namespace)
            .bind(&key)
            .fetch_optional(&state.pool)
            .await
            .map_err(|err| AppError::Database(err.to_string()))?;

    match row {
        Some(payload) => {
            let value = serde_json::from_str(&payload)
                .map_err(|err| AppError::InvalidJson(err.to_string()))?;
            Ok(Json(value))
        }
        None => Err(AppError::NotFound),
    }
}

async fn delete_kv(
    State(state): State<AppState>,
    AxumPath((namespace, key)): AxumPath<(String, String)>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
) -> AppResult<Json<KvResponse>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    sqlx::query("DELETE FROM kv_store WHERE namespace = ? AND key = ?")
        .bind(&namespace)
        .bind(&key)
        .execute(&state.pool)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(KvResponse {
        ok: true,
        namespace,
        key,
    }))
}

async fn append_stream(
    State(state): State<AppState>,
    AxumPath((namespace, stream)): AxumPath<(String, String)>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Json(body): Json<Value>,
) -> AppResult<Json<StreamAppendResponse>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let payload =
        serde_json::to_string(&body).map_err(|err| AppError::InvalidJson(err.to_string()))?;
    let now = Utc::now().timestamp_micros();

    let mut tx = state
        .pool
        .begin()
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    let next_offset: i64 = sqlx::query_scalar::<_, i64>(
        "SELECT COALESCE(MAX(offset), -1) FROM streams WHERE namespace = ? AND stream = ?",
    )
    .bind(&namespace)
    .bind(&stream)
    .fetch_one(&mut *tx)
    .await
    .map_err(|err| AppError::Database(err.to_string()))?
        + 1;

    sqlx::query(
        "INSERT INTO streams (namespace, stream, offset, value, created_at)
             VALUES (?, ?, ?, ?, ?)",
    )
    .bind(&namespace)
    .bind(&stream)
    .bind(next_offset)
    .bind(payload)
    .bind(now)
    .execute(&mut *tx)
    .await
    .map_err(|err| AppError::Database(err.to_string()))?;

    tx.commit()
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(StreamAppendResponse {
        ok: true,
        namespace,
        stream,
        offset: next_offset,
    }))
}

async fn api_profile(
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    if token.is_some() {
        // No-op authorization check placeholder for future use
    }
    Ok(Json(json!({
        "profile": "local",
        "updated_at": Utc::now().timestamp(),
    })))
}

async fn api_config(
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    if token.is_some() {
        // Placeholder for config auth
    }
    Ok(Json(json!({
        "kafka_enabled": true,
        "reddb_enabled": true,
        "spark_enabled": true,
        "infermesh_enabled": true,
        "version": "0.2.0",
    })))
}

async fn record_action_result(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Json(body): Json<Value>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, None);
    let payload =
        serde_json::to_string(&body).map_err(|err| AppError::InvalidJson(err.to_string()))?;
    let now = Utc::now().timestamp_micros();

    let mut tx = state
        .pool
        .begin()
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    let next_offset: i64 = sqlx::query_scalar::<_, i64>(
        "SELECT COALESCE(MAX(offset), -1) FROM streams WHERE namespace = ? AND stream = ?",
    )
    .bind(&namespace)
    .bind("action_results")
    .fetch_one(&mut *tx)
    .await
    .map_err(|err| AppError::Database(err.to_string()))?
        + 1;

    sqlx::query(
        "INSERT INTO streams (namespace, stream, offset, value, created_at)
             VALUES (?, ?, ?, ?, ?)",
    )
    .bind(&namespace)
    .bind("action_results")
    .bind(next_offset)
    .bind(payload)
    .bind(now)
    .execute(&mut *tx)
    .await
    .map_err(|err| AppError::Database(err.to_string()))?;

    tx.commit()
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(json!({
        "success": true,
        "timestamp": now as f64 / 1_000_000.0,
    })))
}

async fn read_stream(
    State(state): State<AppState>,
    AxumPath((namespace, stream)): AxumPath<(String, String)>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(params): Query<StreamQuery>,
) -> AppResult<Json<Vec<StreamItem>>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let count = params.count.clamp(1, 1000);
    let rows = sqlx::query(
        "SELECT offset, value FROM streams
         WHERE namespace = ? AND stream = ? AND offset >= ?
         ORDER BY offset
         LIMIT ?",
    )
    .bind(&namespace)
    .bind(&stream)
    .bind(params.start)
    .bind(count)
    .fetch_all(&state.pool)
    .await
    .map_err(|err| AppError::Database(err.to_string()))?;

    let mut items = Vec::with_capacity(rows.len());
    for row in rows {
        let offset: i64 = row.get("offset");
        let raw: String = row.get("value");
        let value =
            serde_json::from_str(&raw).map_err(|err| AppError::InvalidJson(err.to_string()))?;
        items.push(StreamItem { offset, value });
    }

    Ok(Json(items))
}

async fn list_signatures(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);

    let payload = dashboard::signatures_payload(&state.pool, &namespace)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(payload))
}

async fn get_signature(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);

    let payload = dashboard::signature_detail_payload(&state.pool, &namespace, &name)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(payload))
}

async fn get_signature_schema(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);

    let row = sqlx::query("SELECT body, metadata FROM signatures WHERE namespace = ? AND name = ?")
        .bind(&namespace)
        .bind(&name)
        .fetch_optional(&state.pool)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    let Some(row) = row else {
        return Err(AppError::NotFound);
    };

    let body_raw: String = row.get("body");
    let metadata_raw: Option<String> = row.get("metadata");

    let body = serde_json::from_str::<Value>(&body_raw).unwrap_or_else(|_| json!({}));
    let metadata = metadata_raw
        .and_then(|m| serde_json::from_str::<Value>(&m).ok())
        .unwrap_or_else(|| json!({}));

    let schema_value = metadata
        .get("schema")
        .cloned()
        .or_else(|| body.get("schema").cloned())
        .unwrap_or_else(|| {
            json!({
                "inputs": [{"name": "context", "desc": "Input context"}],
                "outputs": [{"name": "result", "desc": "Result"}],
            })
        });

    let inputs = schema_value
        .get("inputs")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_else(|| vec![json!({"name": "context", "desc": "Input context"})]);
    let outputs = schema_value
        .get("outputs")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_else(|| vec![json!({"name": "result", "desc": "Result"})]);

    Ok(Json(json!({
        "name": name,
        "inputs": inputs,
        "outputs": outputs,
    })))
}

async fn create_signature(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Json(payload): Json<SignaturePayload>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    if payload.name.trim().is_empty() {
        return Err(AppError::BadRequest("name must not be empty".into()));
    }

    let namespace = resolve_namespace(&state, payload.namespace.clone());

    let existing: Option<i64> =
        sqlx::query_scalar("SELECT 1 FROM signatures WHERE namespace = ? AND name = ?")
            .bind(&namespace)
            .bind(&payload.name)
            .fetch_optional(&state.pool)
            .await
            .map_err(|err| AppError::Database(err.to_string()))?;

    if existing.is_some() {
        return Err(AppError::BadRequest("signature already exists".into()));
    }

    upsert_signature(&state, &namespace, &payload.name, &payload).await?;

    let detail = dashboard::signature_detail_payload(&state.pool, &namespace, &payload.name)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    let summary = detail
        .get("metrics")
        .cloned()
        .unwrap_or_else(|| json!({"name": payload.name}));

    Ok(Json(json!({
        "success": true,
        "updated": summary,
    })))
}

async fn update_signature(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
    Json(mut payload): Json<SignaturePayload>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);

    payload.name = name.clone();

    let existing: Option<i64> =
        sqlx::query_scalar("SELECT 1 FROM signatures WHERE namespace = ? AND name = ?")
            .bind(&namespace)
            .bind(&name)
            .fetch_optional(&state.pool)
            .await
            .map_err(|err| AppError::Database(err.to_string()))?;

    if existing.is_none() {
        return Err(AppError::NotFound);
    }

    upsert_signature(&state, &namespace, &name, &payload).await?;

    let detail = dashboard::signature_detail_payload(&state.pool, &namespace, &name)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    let summary = detail
        .get("metrics")
        .cloned()
        .unwrap_or_else(|| json!({"name": name}));

    Ok(Json(json!({
        "success": true,
        "updated": summary,
    })))
}

async fn delete_signature(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);

    let affected = sqlx::query("DELETE FROM signatures WHERE namespace = ? AND name = ?")
        .bind(&namespace)
        .bind(&name)
        .execute(&state.pool)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    if affected.rows_affected() == 0 {
        return Err(AppError::NotFound);
    }

    sqlx::query("DELETE FROM signature_verifier_links WHERE namespace = ? AND signature_name = ?")
        .bind(&namespace)
        .bind(&name)
        .execute(&state.pool)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(json!({
        "success": true,
        "deleted": name,
        "namespace": namespace,
    })))
}

async fn list_verifiers(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);

    let payload = dashboard::verifiers_payload(&state.pool, &namespace)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(payload))
}

async fn get_verifier(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);

    let payload = dashboard::verifiers_payload(&state.pool, &namespace)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    let summary = payload
        .get("verifiers")
        .and_then(|list| list.as_array())
        .and_then(|list| {
            list.iter()
                .find(|item| item.get("name").and_then(|v| v.as_str()) == Some(name.as_str()))
        })
        .cloned()
        .ok_or(AppError::NotFound)?;

    Ok(Json(summary))
}

async fn create_verifier(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Json(payload): Json<VerifierPayload>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    if payload.name.trim().is_empty() {
        return Err(AppError::BadRequest("name must not be empty".into()));
    }

    let namespace = resolve_namespace(&state, payload.namespace.clone());

    let existing: Option<i64> =
        sqlx::query_scalar("SELECT 1 FROM verifiers WHERE namespace = ? AND name = ?")
            .bind(&namespace)
            .bind(&payload.name)
            .fetch_optional(&state.pool)
            .await
            .map_err(|err| AppError::Database(err.to_string()))?;

    if existing.is_some() {
        return Err(AppError::BadRequest("verifier already exists".into()));
    }

    upsert_verifier(&state, &namespace, &payload.name, &payload).await?;

    let payload_resp = dashboard::verifiers_payload(&state.pool, &namespace)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    let summary = payload_resp
        .get("verifiers")
        .and_then(|list| list.as_array())
        .and_then(|list| {
            list.iter().find(|item| {
                item.get("name").and_then(|v| v.as_str()) == Some(payload.name.as_str())
            })
        })
        .cloned()
        .unwrap_or_else(|| json!({"name": payload.name}));

    Ok(Json(json!({
        "success": true,
        "updated": summary,
    })))
}

async fn update_verifier(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
    Json(mut payload): Json<VerifierPayload>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);
    payload.name = name.clone();

    let existing: Option<i64> =
        sqlx::query_scalar("SELECT 1 FROM verifiers WHERE namespace = ? AND name = ?")
            .bind(&namespace)
            .bind(&name)
            .fetch_optional(&state.pool)
            .await
            .map_err(|err| AppError::Database(err.to_string()))?;

    if existing.is_none() {
        return Err(AppError::NotFound);
    }

    upsert_verifier(&state, &namespace, &name, &payload).await?;

    let payload_resp = dashboard::verifiers_payload(&state.pool, &namespace)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    let summary = payload_resp
        .get("verifiers")
        .and_then(|list| list.as_array())
        .and_then(|list| {
            list.iter()
                .find(|item| item.get("name").and_then(|v| v.as_str()) == Some(name.as_str()))
        })
        .cloned()
        .unwrap_or_else(|| json!({"name": name.clone()}));

    Ok(Json(json!({
        "success": true,
        "updated": summary,
    })))
}

async fn delete_verifier(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);

    let result = sqlx::query("DELETE FROM verifiers WHERE namespace = ? AND name = ?")
        .bind(&namespace)
        .bind(&name)
        .execute(&state.pool)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    if result.rows_affected() == 0 {
        return Err(AppError::NotFound);
    }

    sqlx::query("DELETE FROM signature_verifier_links WHERE namespace = ? AND verifier_name = ?")
        .bind(&namespace)
        .bind(&name)
        .execute(&state.pool)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(json!({
        "success": true,
        "deleted": name,
        "namespace": namespace,
    })))
}

async fn system_graph(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
) -> AppResult<Json<GraphResponse>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let signature_rows = sqlx::query("SELECT namespace, name, metadata FROM signatures")
        .fetch_all(&state.pool)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    let verifier_rows = sqlx::query("SELECT namespace, name, metadata FROM verifiers")
        .fetch_all(&state.pool)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    let link_rows = sqlx::query(
        "SELECT namespace, signature_name, verifier_name FROM signature_verifier_links",
    )
    .fetch_all(&state.pool)
    .await
    .map_err(|err| AppError::Database(err.to_string()))?;

    let mut nodes = Vec::new();

    for row in signature_rows {
        let namespace: String = row.get("namespace");
        let name: String = row.get("name");
        let id = format!("signature:{}:{}", namespace, name);
        let metadata_raw: Option<String> = row.get("metadata");
        let metadata = metadata_raw
            .map(|m| serde_json::from_str(&m).map_err(|err| AppError::InvalidJson(err.to_string())))
            .transpose()?
            .unwrap_or_else(|| empty_object());

        nodes.push(GraphNode {
            id: id.clone(),
            kind: "signature".into(),
            label: name,
            namespace,
            metadata,
        });
    }

    for row in verifier_rows {
        let namespace: String = row.get("namespace");
        let name: String = row.get("name");
        let id = format!("verifier:{}:{}", namespace, name);
        let metadata_raw: Option<String> = row.get("metadata");
        let metadata = metadata_raw
            .map(|m| serde_json::from_str(&m).map_err(|err| AppError::InvalidJson(err.to_string())))
            .transpose()?
            .unwrap_or_else(|| empty_object());

        nodes.push(GraphNode {
            id: id.clone(),
            kind: "verifier".into(),
            label: name,
            namespace,
            metadata,
        });
    }

    let mut edges = Vec::new();
    for row in link_rows {
        let namespace: String = row.get("namespace");
        let signature_name: String = row.get("signature_name");
        let verifier_name: String = row.get("verifier_name");

        let from = format!("signature:{}:{}", namespace, signature_name);
        let to = format!("verifier:{}:{}", namespace, verifier_name);

        edges.push(GraphEdge {
            from,
            to,
            kind: "verifies".into(),
        });
    }

    Ok(Json(GraphResponse { nodes, edges }))
}

async fn api_status(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);

    let payload = dashboard::status_payload(&state.pool, &namespace)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(payload))
}

async fn api_logs(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<LogsQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);
    let limit = query.limit.unwrap_or(200).clamp(1, 1000);

    let payload = dashboard::logs_payload(&state.pool, &namespace, limit)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(payload))
}

async fn api_metrics(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);
    let payload = dashboard::metrics_payload(&state.pool, &namespace)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(payload))
}

async fn api_learning_metrics(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);
    let payload = dashboard::learning_metrics_payload(&state.pool, &namespace)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(payload))
}

async fn api_performance_history(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);
    let payload = dashboard::performance_history_payload(&state.pool, &namespace)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(payload))
}

async fn api_kafka_topics(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);
    let payload = dashboard::kafka_topics_payload(&state.pool, &namespace)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(payload))
}

async fn api_spark_workers(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);
    let payload = dashboard::spark_workers_payload(&state.pool, &namespace)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(payload))
}

async fn api_overview(
    State(state): State<AppState>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);
    let payload = dashboard::overview_payload(&state.pool, &namespace)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(payload))
}

async fn signature_analytics(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    maybe_auth: Option<TypedHeader<Authorization<Bearer>>>,
    Query(query): Query<NamespaceQuery>,
) -> AppResult<Json<Value>> {
    let token = maybe_auth.as_ref().map(|auth| auth.token());
    check_authorization(&state, token)?;

    let namespace = resolve_namespace(&state, query.namespace);
    let payload = dashboard::signature_analytics_payload(&state.pool, &namespace, &name)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(Json(payload))
}

async fn upsert_signature(
    state: &AppState,
    namespace: &str,
    name: &str,
    payload: &SignaturePayload,
) -> AppResult<()> {
    let body = serde_json::to_string(&payload.body)
        .map_err(|err| AppError::InvalidJson(err.to_string()))?;
    let metadata = serde_json::to_string(&payload.metadata)
        .map_err(|err| AppError::InvalidJson(err.to_string()))?;
    let now = Utc::now().timestamp_micros();

    let mut tx = state
        .pool
        .begin()
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    sqlx::query(
        "INSERT INTO signatures (namespace, name, body, metadata, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?)
         ON CONFLICT(namespace, name)
         DO UPDATE SET body = excluded.body, metadata = excluded.metadata, updated_at = excluded.updated_at",
    )
    .bind(namespace)
    .bind(name)
    .bind(&body)
    .bind(&metadata)
    .bind(now)
    .bind(now)
    .execute(&mut *tx)
    .await
    .map_err(|err| AppError::Database(err.to_string()))?;

    sqlx::query("DELETE FROM signature_verifier_links WHERE namespace = ? AND signature_name = ?")
        .bind(namespace)
        .bind(name)
        .execute(&mut *tx)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    for verifier in &payload.verifiers {
        if verifier.trim().is_empty() {
            continue;
        }
        sqlx::query(
            "INSERT INTO signature_verifier_links (namespace, signature_name, verifier_name, created_at)
             VALUES (?, ?, ?, ?)",
        )
        .bind(namespace)
        .bind(name)
        .bind(verifier.trim())
        .bind(now)
        .execute(&mut *tx)
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;
    }

    tx.commit()
        .await
        .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(())
}

async fn upsert_verifier(
    state: &AppState,
    namespace: &str,
    name: &str,
    payload: &VerifierPayload,
) -> AppResult<()> {
    let body = serde_json::to_string(&payload.body)
        .map_err(|err| AppError::InvalidJson(err.to_string()))?;
    let metadata = serde_json::to_string(&payload.metadata)
        .map_err(|err| AppError::InvalidJson(err.to_string()))?;
    let now = Utc::now().timestamp_micros();

    sqlx::query(
        "INSERT INTO verifiers (namespace, name, body, metadata, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?)
         ON CONFLICT(namespace, name)
         DO UPDATE SET body = excluded.body, metadata = excluded.metadata, updated_at = excluded.updated_at",
    )
    .bind(namespace)
    .bind(name)
    .bind(&body)
    .bind(&metadata)
    .bind(now)
    .bind(now)
    .execute(&state.pool)
    .await
    .map_err(|err| AppError::Database(err.to_string()))?;

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let data_dir = env::var("REDDB_DATA_DIR").unwrap_or_else(|_| "/data".to_string());
    let data_path = Path::new(&data_dir);
    fs::create_dir_all(data_path)?;
    let db_path = data_path.join("reddb.sqlite");

    let mut connect_opts =
        SqliteConnectOptions::from_str(&format!("sqlite://{}", db_path.display()))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(std::time::Duration::from_secs(5));
    connect_opts = connect_opts.foreign_keys(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(16)
        .connect_with(connect_opts)
        .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS kv_store (
             namespace TEXT,
             key TEXT,
             value TEXT,
             created_at INTEGER,
             updated_at INTEGER,
             PRIMARY KEY (namespace, key)
         )",
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS streams (
             namespace TEXT,
             stream TEXT,
             offset INTEGER,
             value TEXT,
             created_at INTEGER,
             PRIMARY KEY (namespace, stream, offset)
         )",
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS signatures (
             namespace TEXT,
             name TEXT,
             body TEXT,
             metadata TEXT,
             created_at INTEGER,
             updated_at INTEGER,
             PRIMARY KEY (namespace, name)
         )",
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS verifiers (
             namespace TEXT,
             name TEXT,
             body TEXT,
             metadata TEXT,
             created_at INTEGER,
             updated_at INTEGER,
             PRIMARY KEY (namespace, name)
         )",
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS signature_verifier_links (
             namespace TEXT,
             signature_name TEXT,
             verifier_name TEXT,
             created_at INTEGER,
             PRIMARY KEY (namespace, signature_name, verifier_name)
         )",
    )
    .execute(&pool)
    .await?;

    let admin_token = env::var("REDDB_ADMIN_TOKEN").ok();
    if admin_token.is_some() {
        info!("authentication enabled for RedDB server");
    } else {
        info!("authentication disabled for RedDB server");
    }

    let default_namespace = env::var("REDDB_NAMESPACE").unwrap_or_else(|_| "dspy".to_string());

    let state = AppState {
        pool,
        admin_token,
        data_dir: Arc::new(data_path.to_path_buf()),
        default_namespace,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/health", get(health))
        .route("/api/profile", get(api_profile))
        .route("/api/config", get(api_config))
        .route("/api/status", get(api_status))
        .route("/api/logs", get(api_logs))
        .route("/api/metrics", get(api_metrics))
        .route("/api/learning-metrics", get(api_learning_metrics))
        .route("/api/performance-history", get(api_performance_history))
        .route("/api/kafka-topics", get(api_kafka_topics))
        .route("/api/spark-workers", get(api_spark_workers))
        .route("/api/overview", get(api_overview))
        .route(
            "/api/kv/:namespace/:key",
            get(get_kv).put(put_kv).delete(delete_kv),
        )
        .route(
            "/api/streams/:namespace/:stream/append",
            post(append_stream),
        )
        .route("/api/streams/:namespace/:stream/read", get(read_stream))
        .route(
            "/api/signatures",
            get(list_signatures).post(create_signature),
        )
        .route(
            "/api/signatures/:name",
            get(get_signature)
                .put(update_signature)
                .delete(delete_signature),
        )
        .route("/api/signatures/:name/schema", get(get_signature_schema))
        .route("/api/signatures/:name/analytics", get(signature_analytics))
        .route("/api/verifiers", get(list_verifiers).post(create_verifier))
        .route(
            "/api/verifiers/:name",
            get(get_verifier)
                .put(update_verifier)
                .delete(delete_verifier),
        )
        .route("/api/action/record-result", post(record_action_result))
        .route("/api/system/graph", get(system_graph))
        .with_state(state)
        .layer(TraceLayer::new_for_http());

    let host = env::var("REDDB_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = env::var("REDDB_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let ip: IpAddr = host.parse()?;
    let addr = SocketAddr::from((ip, port));

    info!("Starting RedDB server on {}", addr);

    axum::serve(tokio::net::TcpListener::bind(addr).await?, app)
        .await
        .map_err(|err| {
            error!("server error: {err}");
            err
        })?;

    Ok(())
}
