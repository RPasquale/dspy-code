## Overview

DSPy Agent is a local coding assistant that:
- Analyzes your repo and logs
- Suggests safe, useful actions (grep, index, vector search, plan, etc.)
- Learns online as you interact (bandit policy)
- Trains offline with a toolchain (tests/lint/build) using RL
- Can run as a full streaming stack (Kafka + Spark + Ollama + agent)

Default LLM: `qwen3:1.7b` via Ollama.

## Quick Start (local)

```bash
pip install uv
uv sync

# Ensure Ollama is installed; pull a small model
ollama pull qwen3:1.7b

# Launch the agent (interactive)
uv run dspy-agent
```

You’ll see “RL suggested tool: …” before actions, and a session summary with suggested next steps.

## Usage (highlights)

Interactive commands:
- `ctx` — key log events (with de-noised, collapsed stack traces)
- `plan <task>` — propose a plan + commands
- `grep <regex>` — code search
- `index` — build code index for semantic search
- `emb-index` — build embeddings index (auto-build also happens on first “intel/vretr” miss)
- `esearch --q '<query>'` — semantic search over code index
- `vretr --query '<text>'` — vector retrieval over embeddings
- `intel --query '<text>'` — knowledge + vector evidence

The agent learns as you go and persists to `.dspy_rl_state.json`. Per-step rewards also append to `.dspy_rl_events.jsonl` and can be streamed via Kafka `agent.learning`.

## Reinforcement Learning

Two cooperating paths:

1) Online RL (in-session)
- Bandit suggests a safe tool from: `context, codectx, grep, esearch, plan, tree, ls, index, emb-index, intel, vretr`.
- It auto-fills reasonable args from your query.
- Cooldowns for heavy actions (e.g., `emb-index`: 600s) keep sessions snappy.
- Outcomes are scored and persisted so the policy improves across runs.

2) Offline RL CLI (toolchain)
- Trains a policy over actions: `run_tests, lint, build` using verifiers.
- Observations include Kafka-derived context if available (`logs.ctx.*`), otherwise local logs.

```bash
# Optional extras for neural RL
pip install '.[rl]'

# Config stub
uv run dspy-agent rl config init --out .dspy_rl.json

# Bandit (epsilon-greedy)
uv run dspy-agent rl train --workspace . --steps 300

# Neural REINFORCE (uses PufferLib vectorization if available)
uv run dspy-agent rl train --workspace . --steps 1000 --neural --n-envs 4

# PuffeRL PPO shell (example)
uv run dspy-agent rl ppo --workspace . --n-envs 8 --total-steps 200000
```

## Full Stack: Kafka + Spark + Ollama + Agent (Docker)

This spins up Zookeeper/Kafka, Spark to transform `logs.raw.*` → `logs.ctx.*`, Ollama, and agent services.

```bash
uv run dspy-agent lightweight_init --workspace . --logs ./logs --db auto
export DOCKER_BUILDKIT=1
docker compose -f docker/lightweight/docker-compose.yml pull
docker compose -f docker/lightweight/docker-compose.yml build --pull
docker compose -f docker/lightweight/docker-compose.yml up -d --remove-orphans
docker compose -f docker/lightweight/docker-compose.yml ps

# use the stack from your local CLI
export KAFKA_BOOTSTRAP=localhost:9092
uv run dspy-agent
```

Space‑friendly rebuilds:
```bash
export DOCKER_BUILDKIT=1
docker compose -f docker/lightweight/docker-compose.yml pull
docker compose -f docker/lightweight/docker-compose.yml build --pull
docker compose -f docker/lightweight/docker-compose.yml up -d --no-deps dspy-agent dspy-worker dspy-worker-backend dspy-worker-frontend dspy-router

# Safe cleanup (keeps named volumes like Ollama models)
docker image prune -f
docker builder prune -f
```

## LLM Configuration

Ollama (recommended for local):
```bash
export USE_OLLAMA=true
export OLLAMA_MODEL=qwen3:1.7b
```

OpenAI-compatible API:
```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
```

Notes:
- The agent uses native Ollama endpoints by default (`http://localhost:11434`).
- LLM timeouts are capped (~30s) with low retries to avoid stalls; legacy 600s timeout lines are filtered from `ctx`.

## Streaming & Kafka

- Spark job in the stack reads `logs.raw.*` and publishes JSON `{"ctx": [...]}` to `logs.ctx.*` topics.
- The offline RL env consumes these to add context features.
- The agent publishes per-step learning events to `agent.learning` (and `.dspy_rl_events.jsonl` fallback).
- The background trainer consumes `agent.learning` and updates `.dspy_rl_state.json` continuously.

Environment:
```bash
export KAFKA_BOOTSTRAP=localhost:9092
export RL_BACKGROUND_STEPS=50   # background bandit updates per batch (optional)
```

## Architecture

- **embedding/**: code/embeddings indexing, vector search
- **code_tools/**: code analysis & manipulation
- **agents/**: orchestration, router worker
- **streaming/**: local/Kafka runtime, Spark integration, background RL trainer
- **training/**: GEPA modules, deploy helpers
- **rl/**: env, bandits, neural trainer, PufferLib shells

## Troubleshooting

- Kafka not running: the agent’s RL context falls back to local logs; set `KAFKA_BOOTSTRAP=localhost:9092` when Kafka is up.
- No vector matches: `intel`/`vretr` auto-build the embeddings index. You can also run `emb-index` explicitly.
- Docker rebuilds: prefer `docker image prune -f` and `docker builder prune -f`. Avoid `--volumes` if you want to keep Ollama models.

## License

See LICENSE for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

Open an issue with logs and exact commands; include `docker compose ps`, agent logs, and your environment (Python/uv/Ollama versions).
