# Mesh Observability Primer

The mesh control plane now exports first-class Prometheus metrics so routing and backoff behaviour can be inspected without tailing logs.

## Supervisor endpoints

* **Metrics**: `http://stream-supervisor:9098/metrics` (configurable via `SUPERVISOR_METRICS_LISTEN`)
* **Key counters**
  * `supervisor_mesh_dispatch_total{status="success|retry|failure|missing"}` – async publish lifecycle
  * `supervisor_worker_latency_seconds_total` / `supervisor_worker_latency_samples_total` – per-worker latency rolling averages
  * `supervisor_worker_task_failures_total` – acks with `success=false`
  * `supervisor_pending_queue_depth` / `supervisor_mesh_queue_depth` – dispatcher and async pool backlogs

Add the following scrape job to Prometheus:

```yaml
- job_name: mesh-supervisor
  static_configs:
    - targets: ['stream-supervisor:9098']
```

A starter Grafana panel (Stat → PromQL) for dispatch success ratio:

```promql
sum(rate(supervisor_mesh_dispatch_total{status="success"}[5m]))
/
sum(rate(supervisor_mesh_dispatch_total[5m]))
```

## Rust mesh worker

* **Metrics**: `http://rust-env-runner:8083/prometheus`
* The worker logs per-assignment latency and mirror counters for processed / failed jobs. When `MESH_SERVICES_FILE` is configured, the manifest is echoed at startup for audit.

## Tenant routing overrides

The new `MESH_TENANT_DOMAIN_MAP` (or CLI flag `--tenant-domain-map`) lets you pin tenants to latency-optimised domains without rebuilds, and the resulting overrides flow into both the supervisor metrics and the mesh manifest factored at `docker/lightweight/mesh-services.json`.

## Dashboard seed

A minimal dashboard JSON lives at `docs/dashboards/mesh_supervisor_dashboard.json` with three starter panels:

1. Dispatch success ratio (stat)
2. Mesh publish retries (time-series on `status="retry"`)
3. Worker latency heatmap (table using `supervisor_worker_latency_seconds_total`)

Import the JSON into Grafana and update the Prometheus datasource to start iterating.

## Workflow execution telemetry

The Go orchestrator now publishes workflow run records under `http://go-orchestrator:9097/workflows/{id}/runs`.
Prometheus-compatible gauges were added to the supervisor metrics namespace:

* `supervisor_workflow_events_total{workflow, outcome}` – success/failure counts per workflow graph.
* `supervisor_workflow_latency_seconds_total{workflow}` and `supervisor_workflow_latency_samples_total{workflow}` – latency aggregates for downstream SLO panels.

## Runner hardware snapshot

The Rust environment runner exports `/hardware`, returning the latest auto-detected CPU, memory, and GPU
inventory. The `env_runner_rs` metrics server also inlines the snapshot in `/metrics`, and the orchestrator
relays the most recent snapshot to `/workflow` queue submissions so downstream consumers can audit which
hardware was used for each run.
