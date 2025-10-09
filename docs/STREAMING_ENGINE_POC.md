# Streaming Engine Proof-of-Concept

This document captures the end-to-end flow for the Rust data-plane worker and Go control-plane supervisor introduced in the streaming prototype.

## Components

- **Protobuf contracts** live in `proto/runner.proto`. Generation is driven via [`buf`](https://buf.build) so the Go and Rust stubs stay aligned.
- **Go supervisor** (`orchestrator/cmd/stream_supervisor`) consumes Kafka partitions, tracks worker credit, and assigns tasks over a bidirectional gRPC stream.
- **Rust worker** (`env_runner_rs/src/main.rs`) connects to the supervisor, reports credit, pushes acknowledgements, and writes transformed payloads to the output Kafka topic.
- **Docker compose scaffold** under `docker/streaming/` wires up Kafka, the supervisor, and a worker for local iteration.

## Workflow

1. **Generate gRPC bindings** (requires `docker` to pull the `buf` toolchain once):
   ```sh
   make streaming-proto
   ```

2. **Build containers** (produces supervisor + worker images):
   ```sh
   make streaming-build
   ```

3. **Launch the stack**:
   ```sh
   make streaming-up
   ```

   This starts Kafka (`broker`), the Go supervisor (`supervisor`), and a Rust worker (`worker`).

4. **Inspect logs**:
   ```sh
   make streaming-logs
   ```

5. **Tear down** when finished:
   ```sh
   make streaming-down
   ```

## Configuration

Defaults are templated into `docker/streaming/.env` the first time `make streaming-env` or any streaming target runs. Override by editing that file or exporting environment variables before invoking `make`.

Key settings:

| Variable            | Description                         | Default              |
|---------------------|-------------------------------------|----------------------|
| `KAFKA_BROKERS`     | Broker bootstrap servers            | `broker:9092`        |
| `INPUT_TOPIC`       | Topic the supervisor consumes       | `raw.events.demo`    |
| `OUTPUT_TOPIC`      | Topic workers publish feature data  | `features.events.demo` |
| `SUPERVISOR_LISTEN` | gRPC listen address in the container| `:7000`              |
| `SUPERVISOR_GRPC_ADDR` | Worker-side gRPC dial address    | `supervisor:7000`    |
| `MAX_INFLIGHT`      | Worker credit / concurrency budget | `4`                  |
| `MESH_ENDPOINT`     | Optional MeshData target for worker | unset                |
| `MESH_NODE_ID`      | Worker node id when using mesh      | derived from worker  |
| `MESH_DOMAIN`       | Mesh domain scope for routing       | `default`            |
| `MESH_WORKER_ENDPOINT` | Explicit endpoint for the worker node | `http://mesh-worker:50052` |
| `MESH_SERVICES_FILE`| Path to mesh topology manifest (JSON) | supervisor + workers |

## Notes & Follow-ups

- The repo does not commit generated `*.pb.go` or Rust `runner.v1.rs`; run `make streaming-proto` after installing Docker to refresh them.
- `go mod tidy`/`cargo fetch` require network access to pull new dependencies. Run them once outside the sandbox if necessary.
- Supervisor currently round-robins based on the first worker with available credit. Future iterations should add tenant-aware queues and smarter scheduling.
- Worker simply forwards payloads to `OUTPUT_TOPIC`. Slot-in GPU or Arrow processing in `forward_to_output` to extend the hot path.
- Offsets are committed only when the worker acknowledges success. Failed tasks are re-queued by the supervisor.
- Early Mesh support: exporting `MESH_ENDPOINT` (and optional `MESH_NODE_ID`/`MESH_DOMAIN`) switches the worker into MeshData subscription mode. The current supervisor still sources work from Kafka; upcoming work will publish assignments over mesh so both sides can operate without the Kafka hop.
- Go-side mesh helpers live behind the `mesh_integration` build tag (`go build -tags mesh_integration`). Without that tag, the package provides a stub so the supervisor keeps compiling until mesh support is fully wired in.
- Workers in mesh mode keep the supervisor gRPC stream alive for credit/ack control, so the control-plane semantics (credit updates, task acks, offset commits) remain identical across transports.
- Supervisor mesh publishing is optional: set `MESH_PUBLISH_ENDPOINT`, `MESH_SOURCE_NODE`, and `MESH_PUBLISH_DOMAIN` (or use the equivalent CLI flags) and rebuild with `-tags mesh_integration` to dispatch assignments through MeshData instead of the gRPC stream.
- Multi-node topology is the default in the compose files: a hub node (`9001`), a worker node (`9002`), and a trainer node (`9003`). `MESH_SERVICES_FILE` now points at a shared manifest so the supervisor and workers resolve endpoints automatically (additional nodes just extend the JSON document).
- Mesh dispatch is now asynchronous: the supervisor queues job publishes and retries failed sends without blocking the main dispatcher, while credits/acks continue to flow over the gRPC control stream.
- Credit reconciliation: each Mesh publish reserves a worker credit until an ack returns. Retry attempts run inside the async pool with exponential backoff (100 ms → 3 s). When the retry budget is exhausted the supervisor restores the credit, requeues the Kafka message, and emits `supervisor_mesh_dispatch_total{status="failure"}` so offsets are only committed after a confirmed worker ack.
- Override routing per tenant with `MESH_TENANT_DOMAIN_MAP` (comma-separated `tenant=domain` pairs) or the `--tenant-domain-map` CLI flag so latency-sensitive tenants can pin to dedicated mesh domains.
- Enable RL buffering with `RL_RESULTS_TOPIC` and `RL_BUFFER_DIR`; invoke
  `POST /training/rl/start` to launch the new Slurm-backed trainer and monitor
  the run at `/training/rl/status/<task>`. Skill-aware metrics are available
  via `supervisor_training_*` and `supervisor_skill_events_total`.
