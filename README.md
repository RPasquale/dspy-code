# DSPy Code (DSPy Agent)

A minimal, CLI-first coding assistant.

## Install
```bash
pipx install dspy-code      # or: pip install dspy-code
# optional (local LLM): ollama pull qwen3:1.7b
```

## Run
```bash
# One-command local setup (Docker if available, else local threads)
dspy-code quickstart

# Or start an interactive session
dspy-code --workspace /path/to/project
```

## Try a few commands
```bash
plan "fix failing tests"
grep "def .*test"
index && esearch "vector search"
edit "update API route" --apply
```

## Help
```bash
dspy-code --help
```

## Embeddings Pipeline (Kafka + Spark + InferMesh)
- Docs: see `docs/production_spark_embeddings.md` and `docs/infermesh_integration.md` for architecture, configuration, healthchecks, and RL integration.

## License
Apache-2.0 (see LICENSE)

## Streaming Bus API (LocalBus)
- Subscribe (fan-out): `q = bus.subscribe("agent.results", maxsize=100)`
- Subscribe in a consumer group (exactly-once per group): `q = bus.subscribe_group("agent.results", "worker-A", maxsize=100)`
- Backpressure events: subscribe to `agent.backpressure` to observe queue pressure.
- Dead letters: failures and backpressure are appended to `.dspy_reports/dlq.jsonl` and published on `agent.deadletter`.
- Metrics: the status server `/metrics` includes `dlq` and `bus` snapshots when launched via the CLI (`dspy-code up`).

## RL Trainer – Stability Options
The neural trainer adds a few stability knobs:
- `entropy_coef` (default 0.01): encourage exploration.
- `replay_capacity` (default 2048) and `replay_batch` (default 128): on-policy auxiliary replay.
- `grad_clip` (default 1.0): gradient clipping.
- `checkpoint_dir` + `checkpoint_interval`: periodic JSON/PT checkpoints.
- `early_stop_patience`: stop if average reward stagnates.

Example:
```python
from dspy_agent.rl.rlkit import train_puffer_policy
stats = train_puffer_policy(make_env=my_env_factory,
                            steps=5000,
                            n_envs=4,
                            entropy_coef=0.02,
                            replay_capacity=4096,
                            replay_batch=256,
                            grad_clip=1.0,
                            checkpoint_dir=".dspy_checkpoints/rl",
                            checkpoint_interval=100,
                            early_stop_patience=200)
```
## Testing & Debugging

- Run unit/integration tests (excluding docker‐dependent tests):

```
make test
```

- Run docker‐dependent tests (enable with env):

```
make test-docker        # runs pytest -m docker with DOCKER_TESTS=1
```

- Live server debug trace (SSE):
  - Open `/api/debug/trace/stream` in a browser or use the Debug Trace panel on the Overview page.
  - Clear trace via `POST /api/debug/trace` with `{ "clear": true }`.
  - Trace file: `.dspy_reports/server_trace.log`.

## Dev Cycle (One‑Button)

- CLI: `make dev-cycle` runs the end‑to‑end script `scripts/dev_cycle.sh`.
- Frontend/API:
  - Start: `POST /api/dev-cycle/start` (admin only; set `X-Admin-Key` if configured)
  - Stop: `POST /api/dev-cycle/stop` (admin only)
  - Status: `GET /api/dev-cycle/status` (JSON), Stream: `GET /api/dev-cycle/stream` (SSE)
  - Download Logs: `GET /api/dev-cycle/logs` (text/plain attachment)
- Compose compatibility: the script validates `docker/lightweight/docker-compose.yml` and, if your Compose requires env mappings, rewrites `environment:` list items (`- KEY=VAL` / `- KEY`) into mapping style (`KEY: "VAL"` / `KEY: ${KEY:-}`) and uses the rewritten file automatically.

## Experiments (CPU‑only or with external InferMesh)

- Start experiment (admin):
  - `POST /api/experiments/run` with JSON:
    - `model` (optional, defaults to env `INFERMESH_MODEL`)
    - `dataset_path` (JSONL with `{ "text": str }` per line, or TXT one text per line)
    - `max_count` (optional int)
    - `batch_size` (optional, defaults to env `EMBED_BATCH_SIZE`)
    - `normalize` (optional bool)
  - Returns `{ ok: true, id }` and streams logs via `GET /api/experiments/stream?id=<id>`.
- Status/history
  - `GET /api/experiments/status?id=<id>` → progress, rate, logs
  - `GET /api/experiments/history` → recent experiment entries
  - `GET /api/datasets/preview?path=...` → first few lines
- Storage
  - Logs under `.dspy_reports/experiments/<id>.log`
  - History under `.dspy_reports/experiments/history.jsonl`

Local CPU InferMesh
- Build + up: `docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env up -d infermesh`
- Health: `curl http://127.0.0.1:19000/health`
- Configure cache: HuggingFace cache persisted via volume `hf-cache`.

- Workspace & Guards
  - Set workspace path via the Guard Settings panel (UI) or `POST /api/system/workspace`.
  - Set guard thresholds (min_free_gb, min_ram_gb, min_vram_mb) via the Guard Settings panel or `POST /api/system/guard`.

## Observability & Metrics

The agent emits structured events to RedDB (and optionally Kafka) to power dashboards and training:

- tool.start / tool.end: lifecycle for each command with args, session, success, score, duration_sec
- rl.* events: rl.train.start/env_ready, rl.async.started/summary/finished, rl.ppo.start/finished, rl.neural.start/summary/finished
- patch.apply: emitted on patch success/failure with diff stats
- session.summary: rolling chat session summary (also stored in KV agent.session.summary)

Tail streams:

- RedDB: `python scripts/tail_agent_streams.py --stream agent.metrics`
- Kafka: `python scripts/tail_agent_streams.py --kafka --stream agent.metrics --bootstrap localhost:9092`

Lite dashboard:

- `dspy-code-dashboard-lite` (or `python scripts/agent_dashboard_panel.py`)
  - Shows per‑tool stats (count, success, avg_score, avg_dur), recent tool.end, and latest session summary.
  - Flags: `--start`, `--interval`, `--workspace` to filter a specific workspace.
