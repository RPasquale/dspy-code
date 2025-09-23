Production Spark Embeddings
===========================

Architecture
------------
- Parse (Spark):
  - Reads Kafka topic(s) (default `agent.results`).
  - Extracts text and derives `doc_id` (propagated from upstream if present; otherwise `sha256(text)`).
  - Publishes `{text, topic, kafka_ts, doc_id}` to Kafka `embedding_input`.
- Embed (InferMesh client):
  - `embed-worker` batches from `embedding_input`, calls InferMesh `/embed` and emits enriched records to:
    - Kafka `embeddings`
    - Parquet `/workspace/vectorized/embeddings_imesh`
    - Optional RedDB stream + KV (`embedding:{doc_id}`)
 - Index (RedDB kNN layout):
   - `emb-indexer` consumes `embeddings` and persists records to RedDB stream `emb_index` and KV `embvec:{doc_id}` for retrieval/index building.

Deployment
----------
- Start the lightweight stack:
  - `cd docker/lightweight`
  - `docker compose up -d zookeeper kafka infermesh spark-vectorizer embed-worker`
- Healthchecks:
  - `spark-vectorizer`: process probe via `pgrep -f 'spark_vectorize.py'`.
  - `embed-worker`: HTTP `GET /health` on port 9100 (mapped to `127.0.0.1:9101`).
- Ports:
  - InferMesh: `127.0.0.1:9000`
  - Embed‑worker metrics: `127.0.0.1:9101`
  - Spark UI (driver): `127.0.0.1:4041` (REST: `/api/v1/...`)

Configuration
-------------
- Spark vectorizer (parse‑only by default):
  - `KAFKA_BOOTSTRAP=kafka:9092`, `SPARK_KAFKA_TOPICS=agent.results`
  - `SINK_INPUT_TOPIC=embedding_input` to publish parsed payloads
  - Optional sinks: Parquet (`VEC_OUTPUT_DIR`), Kafka vectors (`SINK_TO_KAFKA=1`)
- Embed‑worker (InferMesh client):
  - Kafka: `KAFKA_BOOTSTRAP_SERVERS`, `EMBED_INPUT_TOPIC`, `EMBED_OUTPUT_TOPIC`, `EMBED_GROUP`
  - InferMesh: `INFERMESH_URL`, `EMBED_MODEL`, `INFERMESH_API_KEY` (optional), timeouts/retries/backoff
  - Parquet: `EMBED_WRITE_PARQUET=1`, `EMBED_PARQUET_DIR`
  - RedDB (optional): `REDDB_URL`, `REDDB_NAMESPACE`, `REDDB_TOKEN`, `REDDB_STREAM`, `REDDB_MODE=stream|kv|both`
  - Metrics: `EMBED_METRICS_PORT` (default 9100)
  - Normalization: `EMBED_NORMALIZE=1` (unit vectors), records `normalized=true` and `dim`
  - Cache warmup: `EMBED_CACHE_WARMUP=1000`
  - DLQ: `EMBED_DLQ_TOPIC` (Kafka) and `REDDB_DLQ_STREAM` (default `embeddings_dlq`)

- Indexer (RedDB kNN):
  - `EMB_INDEX_SHARDS` (default 32) and `EMB_SHARD_MAX_IDS` (default 5000)
  - `REDDB_INDEX_STREAM` (default `emb_index`)
  - Backend DLQ tail override path: `REDDB_DLQ_HTTP_PATH` with placeholders `{ns}`, `{stream}`, `{limit}`

Monitoring
----------
- Backend SSE endpoints:
  - `/api/infermesh/stream`: status + RTT + throughput estimate from Parquet.
  - `/api/vectorizer/stream`: Parquet throughput for vectorizer output.
  - `/api/embed-worker/stream`: embed‑worker internal metrics from its `/metrics` endpoint.
  - `/api/spark/stream`: cluster + streaming batch metrics via Spark REST UI.
- Embed‑worker internal: `GET /metrics` for counters (`batches`, `records_in/out`, `last_infermesh_rtt_ms`, `infermesh_failures`).
 - UI panels: Spark streaming metrics and rolling rates; Vectorizer and Embed throughput charts; Overview includes a kNN Quick Search; Monitoring has full kNN query, shard stats, and DLQ download.

Reliability & Backpressure
--------------------------
- Batching: tune `EMBED_BATCH_SIZE` and `EMBED_MAX_WAIT_SEC` for latency vs throughput.
- Retries: `INFERMESH_RETRIES` with exponential backoff (`INFERMESH_BACKOFF_SEC`) to smooth transient issues.
- Idempotency: stable `doc_id` avoids KV duplication; ensure upstream producer supplies it for exactness.
- Storage: Parquet sink provides durable checkpoints for RL training and analytics.

Schema
------
Emitted record (Kafka/Parquet):
```
{
  "topic": string | null,
  "text": string,
  "doc_id": string,
  "vector": number[],
  "kafka_ts": number | null,
  "embedded_ts": number,
  "model": string
}
```

Coding Agent Integration
------------------------
- Context Features: Use `attach_vector_context` to stream vectors from Parquet into the RL environment observation space.
  - Example: `make_env2 = attach_vector_context(make_env, "/workspace/vectorized/embeddings_imesh", batch_size=128)`.
- Retrieval‑Augmented Signals: Compute similarity between current task context and recent embeddings to guide action selection.
- Skill Indexing: Build an index over `embeddings` to recommend tools/prompts based on nearest neighbor to past successful actions.
- Curriculum: Filter vectors by difficulty/novelty to schedule training episodes (e.g., sample hard/problematic contexts more often).

Reinforcement Learning Enhancements
----------------------------------
- Reward Shaping:
  - Add verifiers that score solution proximity using embedding similarity to known good trajectories.
  - Penalize repetitive states by measuring vector drift; encourage exploration with novelty bonuses.
- State Augmentation:
  - Concatenate vector context with verifier scores; the RL policy observes semantic context rather than raw text.
- Off‑Policy Training:
  - Use Parquet vectors with labels (success/failure, latency, tokens) via `VectorLabelSampler` for supervised pretraining.
- Telemetry‑Driven Policies:
  - Track `/metrics` (RTT, throughput) and adapt batch sizes or action timeouts in the agent when the embedding backend is under load.

Performance Tuning
------------------
- Increase `EMBED_BATCH_SIZE` on high‑throughput workloads; adjust `EMBED_MAX_WAIT_SEC` to cap tail latency.
- Co‑locate embed‑worker and InferMesh on the same host/network to minimize RTT.
- Enable CPU/GPU acceleration in InferMesh as applicable and match model choice (`EMBED_MODEL`) to throughput goals.

Next Steps
----------
- Exact‑once delivery: adopt Kafka transactional writes or Spark structured streaming with idempotent sinks.
- Spark metrics: wire `/api/spark/stream` to real Spark REST metrics instead of simulated data.
- Indexing: add a consumer to persist `embeddings` into RedDB kNN structures for low‑latency retrieval.
- Frontend: surface embed‑worker `/metrics` and rolling charts for throughput and error rates.
 - KNN: Query endpoint `/api/knn/query` (POST JSON: `{ vector?: number[], doc_id?: string, k?: number, shards?: number[] }`). Returns `{ neighbors: [{doc_id, score}], count_scored }`. Intended for small shard sets.
