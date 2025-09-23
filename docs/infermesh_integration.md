InferMesh Integration Guide
===========================

Overview
--------
- InferMesh provides high‑throughput, microservice‑style text embeddings.
- This repo integrates InferMesh via a Kafka‑first pipeline:
  - Spark parses and publishes `{text, topic, kafka_ts, doc_id}` to Kafka `embedding_input`.
  - `embed-worker` batches texts, calls InferMesh `/embed`, and writes enriched records to Kafka `embeddings`, Parquet, and optionally RedDB.

Endpoints Used
--------------
- `GET /health`: liveness probe. Used by monitoring and container healthchecks.
- `POST /embed`: main embedding endpoint.
  - Request JSON: `{ "model": string, "inputs": string[] }`
  - Response JSON: `{ "vectors": number[][] }` or `{ "embeddings": number[][] }`

Authentication
--------------
- Optional bearer token via `Authorization: Bearer <INFERMESH_API_KEY>` header.
- Configure with `INFERMESH_API_KEY` in the embed‑worker container.

Batching and Timeouts
---------------------
- `EMBED_BATCH_SIZE` (default 64) and `EMBED_MAX_WAIT_SEC` (default 0.5) control the batcher.
- `INFERMESH_TIMEOUT_SEC` (default 30), `INFERMESH_RETRIES` (default 2), and `INFERMESH_BACKOFF_SEC` (default 0.5) control HTTP behavior and backoff.

Record Schema (emitted)
-----------------------
Each embedded record conforms to the following JSON schema and is sent to Kafka and Parquet:
```
{
  "topic": string | null,
  "text": string,
  "doc_id": string,            // propagated or sha256(text)
  "vector": number[],          // embedding
  "kafka_ts": number | null,   // source Kafka timestamp
  "embedded_ts": number,       // epoch seconds
  "model": string              // InferMesh model id
}
```

RedDB Integration (optional)
----------------------------
- Controlled by `REDDB_URL`, `REDDB_NAMESPACE`, `REDDB_TOKEN` (optional), `REDDB_STREAM`, and `REDDB_MODE` (`stream|kv|both`).
- Stream append: `POST /api/streams/{ns}/{REDDB_STREAM}/append` with record JSON.
- KV upsert: `PUT /api/kv/{ns}/embedding:{doc_id}` → record JSON. Falls back to `embedding:{sha256(text)}` if no `doc_id`.

Health and Metrics
------------------
- Embed‑worker exposes `GET /health` and `GET /metrics` on `EMBED_METRICS_PORT` (default 9100) inside the container. Compose maps it to host `127.0.0.1:9101`.
- InferMesh liveness and RTT are monitored by the backend SSE (`/api/infermesh/stream`). Throughput is approximated from Parquet sink.

DLQ and Normalization
---------------------
- Optional normalization: `EMBED_NORMALIZE=1` outputs unit‑length vectors (and records `normalized=true`, `dim=<len>`).
- DLQ: Set `EMBED_DLQ_TOPIC` to publish failed/empty vectors to Kafka, and/or configure `REDDB_DLQ_STREAM` (default `embeddings_dlq`) to append DLQ entries into RedDB.
- Backend endpoint `GET /api/embed-worker/dlq?limit=N` fetches DLQ snapshots; override the DLQ HTTP path via `REDDB_DLQ_HTTP_PATH` (placeholders: `{ns}`, `{stream}`, `{limit}`).

Operational Notes
-----------------
- Idempotency: Downstream KV keys use `doc_id` to avoid duplicates. If your upstream generates stable IDs, set them in the input payload (`doc_id`/`id`/`document_id`/`key`).
- Backpressure: Tune `EMBED_BATCH_SIZE`, `EMBED_MAX_WAIT_SEC`, and Kafka consumer group settings. Increase Parquet directory write throughput by mounting a fast volume.
- Failure modes: The worker retries InferMesh calls with exponential backoff. Persistent failures produce empty vectors for those inputs to avoid consumer stalls; monitor `/metrics` for `infermesh_failures`.

Security
--------
- Treat InferMesh and RedDB tokens as secrets. Pass via env variables or container secrets, not as CLI args.
- If running InferMesh outside Docker, restrict access with network ACLs and TLS where applicable.

Examples
--------
- Compose (excerpt): see `docker/lightweight/docker-compose.yml` services `infermesh`, `spark-vectorizer`, and `embed-worker`.
- Configure image:
  - Set a valid InferMesh image in `docker/lightweight/.env`:
    - `INFERMESH_IMAGE=ghcr.io/<org>/infermesh:<tag>` (CPU/GPU tag per your environment)
  - Optional: change host port via `INFERMESH_HOST_PORT` (default 19000).
- Quick start:
  - `docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env up -d infermesh`
  - Validate: `curl -fsS http://127.0.0.1:$INFERMESH_HOST_PORT/health`
  - Start pipeline: `docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env up -d spark-vectorizer embed-worker`
  - Monitor: UI → Monitoring → InferMesh panel and `/api/infermesh/stream`.
