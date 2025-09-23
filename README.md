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

- Workspace & Guards
  - Set workspace path via the Guard Settings panel (UI) or `POST /api/system/workspace`.
  - Set guard thresholds (min_free_gb, min_ram_gb, min_vram_mb) via the Guard Settings panel or `POST /api/system/guard`.
