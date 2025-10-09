# Mesh Integration Optimization Roadmap

This document captures the concrete follow-on work to evolve the current mesh-enabled DSPy stack. It covers near-term transport optimizations, how Spark fits into the picture, and a staged plan for rolling the changes out.

## Decision Snapshot (September 2025)

- **QoS policies**: baseline guardrails ship with tenant/domain weight presets checked into `docker/lightweight/mesh-services.json`.  Supervisors must load overrides from `MESH_TENANT_DOMAIN_MAP`, and new Go tests cover the routing helpers so regressions are caught during CI.
- **Async publish metrics**: the supervisor now emits gauges/counters (`supervisor_mesh_dispatch_total`, `mesh_queue_depth`) that are exercised by unit tests.  Dashboards consume these metrics via the React frontend smoke tests.
- **Spark/mesh alignment**: streaming workers prefer mesh-first delivery in staging; Kafka remains as durability fallback.  The roadmap keeps a phased rollout but marks mesh-only jobs as the next activation gate once automated coverage is green.

## 1. Transport Optimizations

1. **Multi-node mesh topology**
   - Run at least three mesh nodes: hub (`9001`), inference worker (`9002`), and trainer (`9003`). We now ship a default gateway (`9010`) config to encourage further expansion.
   - Add additional gateways (edge nodes, on-prem bridges) by appending entries to `MESH_SERVICES_JSON`; the helper scripts auto-propagate this map to agent, supervisor, and worker containers.
   - Maintain `mesh_nodes.yaml` or extend `MESH_SERVICES_JSON` with tag metadata (e.g., `{"tags": ["gateway", "edge"]}`) to help the supervisor route by role.

2. **Mesh QoS & batching**
   - Use `MeshControl.SetPolicy` to install per-domain limits (priority weights, max pending, circuit breakers). Map Kafka topic / tenant → mesh domain so noisy tenants can be throttled at the transport level.
   - Tune `send` batching in mesh-core (linger, chunk size) to match the worker’s `max_inflight` and Kafka commit cadence.

3. **Direct mesh dispatch**
   - Add a “mesh-first” dispatcher that publishes assignments directly to MeshData (with WAL enabled) and persists only checkpoints/offsets to Kafka.
   - Keep the Kafka path as a compatibility fallback until we confirm mesh durability/metrics meet SLOs.

4. **Asynchronous publishing**
   - Wrap the supervisor’s mesh `Send` call in a goroutine pool with retry/back-off. Each in-flight Kafka message should remain buffered until the mesh delivery is acknowledged, then commit offsets.
   - Add a bounded queue + telemetry (`mesh_publish_latency_ms`, `mesh_publish_retries_total`).

5. **Structured telemetry**
   - Scrape mesh-core’s metrics endpoints (session RTT, WAL writes, dedup hits) and export them to Prometheus under `mesh_*` namespaces.
   - Add Prometheus service discovery/job entries pointing at `mesh-hub:9100`, `mesh-worker:9100`, etc., or deploy a Prometheus sidecar that scrapes `/metrics` inside each mesh container.
   - Ship Grafana dashboards correlating supervisor metrics (`mesh_dispatch_latency_ms`, `mesh_requeue_total`, async queue depth) with mesh-core RTT/WAL counters to identify bottlenecks.

6. **Adaptive worker concurrency**
   - Expose per-domain `max_inflight` overrides via environment or mesh policy so hot domains can raise the cap without flooding Kafka commits.
   - Feed mesh credits through the existing supervisor stream so control semantics remain unchanged.

## 2. Spark & Mesh Alignment

Spark remains the heavyweight data plane for embeddings and analytics, but there are opportunities to integrate:

- **Current state**: Spark vectorizers read from Kafka/RedDB, call InferMesh, and sink Parquet. Mesh is presently orthogonal—it accelerates orchestrator ↔ worker control traffic only.
- **Chunked mesh delivery**: For high-volume jobs (code snapshots, telemetry batches), Spark could publish work items to dedicated mesh domains. Mesh-aware workers would consume those items, deliver results back over Kafka or direct mesh replies.
- **Unified transport vision**: If Spark executors register as mesh nodes, we can reduce duplicate Kafka topics—Spark publishes inference tasks via mesh, workers respond mesh-first, and Kafka is relegated to durability checkpoints.

## 3. Phased Implementation Plan

1. **Mesh topology shakeout**
   - Launch multi-node mesh cluster locally (compose) and in staging.
   - Verify link-state propagation, ECMP selection, and message RTT under concurrent load.

2. **Control plane enhancements**
   - Implement supervisor mesh publisher workers with async retry.
   - Pipe mesh metrics into Prometheus / Grafana dashboards.
   - Add per-domain credit overrides in the worker and validate backpressure remains stable.

3. **Policy rollout**
   - Define tenant → mesh domain mappings.
   - Apply `MeshControl.SetPolicy` rules in staging; monitor for throttled tenants and adjust weights.

4. **Direct mesh dispatch pilot**
   - Enable WAL + delivery confirmations in mesh-core.
   - Run dual-write (Kafka + mesh) in canary mode, comparing offsets/throughput.
   - Graduate workloads to mesh-only once durability SLOs are met.

5. **Spark alignment**
   - Prototype a Spark job that publishes to mesh domains instead of Kafka.
   - Measure latency/throughput vs. Kafka-only baseline; decide on hybrid vs. full mesh adoption for analytics lanes.

## 4. Immediate Action Items

- Update compose/infra manifests to include at least three mesh nodes with explicit IDs.
- Extend the supervisor to cache tenant → mesh domain mappings and call `SetPolicy` when new tenants appear.
- Instrument mesh publish/ack metrics and export them via the existing supervisor metrics HTTP endpoint.
- Draft a playbook for operators covering mesh node scaling, policy changes, and telemetry dashboards.
- Supervisor CLI exposes `--mesh-services-file` and `--tenant-domain-map`, so updating topology manifests or tenant routing no longer requires rebuilds; docker compose defaults mount `docker/lightweight/mesh-services.json` into each mesh-aware service.

Once these steps land, we’ll have a transport that scales beyond a single node, honors tenant QoS, and gives us the telemetry needed to tune latency before we graduate to mesh-first dispatch.
