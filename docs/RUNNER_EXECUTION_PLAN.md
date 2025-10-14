# Env Runner Execution Plan

This document captures the requirements and proposed design changes needed to
transform the current mock-style runner into a production-ready execution
engine that powers the orchestrator end-to-end.

## 1. Scope & Goals

* Remove all placeholder logic and execute real workloads on behalf of the
  orchestrator.
* Support the core DSPy task classes (`cpu_short`, `cpu_long`, `gpu`, `gpu_slurm`)
  with explicit resource constraints.
* Provide reliable status tracking, logging, and metrics that the orchestrator
  can surface to end users.
* Maintain isolation and security boundaries (Docker containers, namespaces,
  GPU selection).

## 2. Workload Requirements

| Task Class   | Typical Use Case                           | Target SLA | Execution Mode                        |
|--------------|--------------------------------------------|------------|---------------------------------------|
| `cpu_short`  | GEPA inference, small data transforms      | < 60s      | Local process or Docker               |
| `cpu_long`   | Batch analytics, embeddings, ETL           | < 15m      | Docker container w/ limits            |
| `gpu`        | LLM inference, CUDA workloads              | < 10m      | Docker + dedicated single GPU per task|
| `gpu_slurm`  | Remote HPC jobs (defer to Slurm bridge)    | external   | Submit via Slurm (unchanged)          |

Each task payload supplies a JSON document that MAY contain:

* `workflow_id`, `tenant`, and `context` – used for auditing and payload
  enrichment.
* `execution` block with desired docker image, entrypoint, command, env vars,
  mounts, resource hints.
* `artifacts` block describing input/output locations (S3, RedB, local path).

## 3. Execution Engine

### 3.1 Runner Responsibilities

1. Accept assignments from the orchestrator (via the existing supervisor gRPC).
2. Translate the assignment payload into an execution plan:
   * Determine the runtime mode (native binary, docker, python entry point).
   * Resolve resource hints (CPU, memory, GPU) → structured limit configuration.
   * Prepare working directory, fetch artifacts, inject secrets.
3. Launch the workload with isolation:
   * Default: Docker container using `docker run` (through Bollard or CLI).
   * Optional: Native exec for simple scripts (with process sandboxing).
   * GPU: dedicate a full GPU per task by issuing `--gpus device=<id>` (no fractional sharing) and exporting device-specific environment variables.
4. Stream stdout/stderr back to orchestrator or append to RedB topic.
5. Handle completion:
   * Collect exit status, execution duration, produced artifacts metadata.
   * Publish result back to orchestrator (HTTP response) and RedB stream.
   * Update Prometheus counters/ histograms.
6. Retry policy: automatic retry for transient errors (network, container
   pull) with exponential backoff (`RetryConfig`).

### 3.2 Components to Implement / Update

* `runner/executor.rs` (new): orchestrates runtime selection, container launch,
  artifact management.
* `metrics.rs`: add histograms for task duration, counters per class/outcome,
  gauge for active containers.
* `http_server.rs`: expose `/healthz` and `/metrics` endpoints (Prometheus text).
* `config.rs`: describe docker socket, allowed images, GPU policy, retry limits,
  working dir base path.
* `logging`: integrate structured tracing (task-id, tenant), persist to local
  log files under `/var/log/env-runner` and stream back to orchestrator.

## 4. Orchestrator Integration

### 4.1 TaskDispatcher Changes

* Replace placeholder response with JSON parsing of runner output:
  ```json
  {
    "status": "completed",
    "result": {
      "latency_ms": 1234,
      "outputs": [{ "path": "s3://...", "type": "parquet" }],
      "metrics": { "tokens": 512 }
    },
    "logs_url": "https://.../log.txt"
  }
  ```
* Update `tasks` map so that each state transitions through `pending` →
  `running` → `completed|failed` with timestamps.
* Emit detailed events (`task_started`, `task_completed`, `task_failed`) with
  enriched payload for Kafka/RedB.
* Surface runner-provided errors (stderr excerpts, exit codes) in the
  `GetTaskStatusResponse` `error` field.

### 4.2 StreamTaskResults (optional enhancement)

* Push incremental updates (percentage, step name) if the runner emits them.
* For LLM-like streaming, consider WebSocket relay from runner to orchestrator.

## 5. Rust env-manager Updates

* Review docker service definitions: ensure all production services (Redis,
  RedB, InferMesh, RL trainers) have accurate images, volumes, env, health
  checks.
* Support per-service override via `ENV_MANAGER_CONFIG` to allow custom images
  or ports per environment.
* Prometheus metrics: confirm `service_start_count`, `service_stop_count`,
  `active_services`, `docker_api_duration_seconds` capture expected values.
* Add service-specific labels (tenant, environment) if multiple stacks run on
  the same host.

## 6. Logging & Observability

* Runner should log to stdout/stderr with TRACE-level metadata (task_id,
  workflow, class). Use `tracing` crates with JSON formatter for easier
  ingestion.
* Structured event logging: orchestrator’s event bus already writes to
  `logs/agent_action.jsonl`—extend schema to include runner-specific fields.
* Prometheus scrape targets:
  * `env-manager`: `http://127.0.0.1:50101/metrics`
  * `env-runner`: `http://127.0.0.1:<runner-port>/metrics`
  * Orchestrator (existing `:9097`).
* Alerting rules to add later: high failure rate, queue depth > threshold,
  runner offline.

## 7. Deliverables & Sequencing

1. **Design finalisation** (this doc + feedback → lock requirements) with explicit per-tenant isolation and RedB artifact storage plans.
2. **Runner execution engine** implementation + tests (single-GPU-per-task scheduling enforced).
3. **Prom/health endpoints** for runner.
4. **Orchestrator dispatcher** integration tests (mock runner → real runner).
5. **Rust env-manager validation** and config tweaks.
6. **Python pipeline updates** (GEPA/RL using real gRPC flows).
7. **End-to-end validation**: submit real task, observe results, metrics, ensure tenant isolation.
8. **Documentation**: update READMEs, runbooks, systemd samples.

## 8. Open Questions

* Tenant isolation implementation details (namespaces vs. dedicated workers).
* Artifact storage contract using RedB (schema, retention, access controls).
* Security controls: allowed container images, secret injection mechanism.
* Progress event format for long-running tasks.

---

Next action: review this plan with stakeholders, resolve open questions, then
begin implementing the runner execution engine (Step 2).
