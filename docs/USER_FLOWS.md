# DSPy Agent User Flows

This guide highlights the quickest way to experience the DSPy agent from the command line and through the React dashboard. Each flow keeps the setup lightweight while surfacing the most engaging views.

## CLI Quickstart

1. **Start core services**
   ```bash
   make stack-build
   make stack-up
   ```
   These targets build the multi-stage Docker images (Go orchestrator, Rust env-runner, Python services) and launch the lightweight stack.

2. **Submit a task**
   ```bash
   curl -X POST http://localhost:9097/queue/submit \
     -H "Content-Type: application/json" \
     -d '{"id":"demo-001","class":"cpu_short","payload":{"echo":"hello"}}'
   ```
   - `gpu_slurm` submissions are forwarded to Slurm and reconciled automatically.
   - The Go orchestrator updates metrics at `http://localhost:9097/metrics`.

3. **Inspect queue state**
   ```bash
   curl http://localhost:9097/queue/status | jq
   curl http://localhost:8080/metrics | jq       # Rust env-runner metrics
   ```

4. **Shut down gracefully**
   ```bash
   make stack-down
   ```
   Closing the queue triggers orchestrator cancellation, so waiting tasks now exit cleanly.

## React Dashboard

The React dashboard lives in `frontend/react-dashboard` and is no longer ignored by default. It ships with Vite + Tailwind for fast reloads and can share the same backend services started above.

```bash
cd frontend/react-dashboard
npm ci
npm run dev
# open http://localhost:5173
```

### Notable views

- **Live Queue** – Mirrors `/queue/status` with animated throughput indicators.
- **Slurm Jobs** – Streams completion/failure events emitted by the Slurm bridge.
- **Latency & Errors** – Plots p95 latency and error buckets sourced from the Rust metrics endpoint.

### Production build

```bash
npm run build
npm run preview
```

The build step outputs to `frontend/react-dashboard/dist`, which is excluded from commits but can be picked up by Nginx or the packaging bundle if desired.

## Bundle Workflow

To generate a distributable bundle with prebuilt binaries:

```bash
./scripts/dspy_stack_packager.sh
ls dist/dspy_stack_bundle_*/start_bundle.sh
```

Inside the extracted bundle, `./start_bundle.sh` validates Docker dependencies, generates an `.env`, and starts the complete stack (Go orchestrator, Rust env-runner, Python APIs, Kafka/RedDB). The script now tolerates missing host binaries and compiles them inside the Docker build when necessary.

## Troubleshooting

- **Metrics not updating** – Ensure `log/env_queue` is mounted (`make stack-up` recreates directories) and confirm the env-runner metrics endpoint at `:8080/metrics`.
- **React proxy errors** – Restart `npm run dev` after bringing the stack up so proxy targets pick up healthy services.
- **Slurm submission** – The bridge honors `ENV_QUEUE_DIR`; set it before launching services if you relocate the queue.
- **RedDB server** – The new Rust RedDB service listens on `:8080`; ensure `REDDB_ADMIN_TOKEN` is set if you expect authentication.
- **Signatures/Verifiers** – CRUD operations now fan out to RedDB (`/api/signatures`, `/api/verifiers`); the React dashboard writes directly to the live store, and the `/api/system/graph` endpoint renders the updated dependency graph.

Enjoy the new streamlined flows and feel free to extend them—each component is now production-ready and designed for predictable local testing.
