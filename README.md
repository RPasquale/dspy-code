dspy-code — Trainable Local Coding Agent

Overview

- Reads your local logs and builds a concise context.
- Uses DSPy with a pluggable LLM backend to propose a plan and suggested commands for your task.
- CLI via `dspy-code` (alias: `dspy-agent`) using uv-managed environment.
- Built with DSPy and integrates with RedDB.

Quick Start

Super Simple Setup (Lightweight)

- Prereqs:
  - Install `Docker` and ensure `docker compose` works.
  - Install `Python 3.10+` and [`uv`](https://github.com/astral-sh/uv) (`pip install uv` if needed).
- Initialize the lightweight stack (generates Dockerfile + compose):
  - `dspy-code lightweight_init --workspace $(pwd) --logs ./logs --db auto`
  - Uses the current directory as workspace; creates `docker/lightweight/`.
- Build and start:
  - `docker compose -f docker/lightweight/docker-compose.yml build --no-cache`
  - `docker compose -f docker/lightweight/docker-compose.yml up -d`
- Verify it’s running:
  - Status page: `http://127.0.0.1:8765`
  - Health: `http://127.0.0.1:8765/health`
  - Deployment info: `http://127.0.0.1:8765/deploy`
  - Latest container outputs: `http://127.0.0.1:8765/containers`
- Open the interactive CLI inside the agent container:
  - `docker compose -f docker/lightweight/docker-compose.yml exec -it dspy-agent dspy-agent start --workspace /workspace`
- Rebuild after code changes (fast loop):
  - `dspy-code lightweight_build`
  - or `./scripts/lightweight_build.sh`

Optional, recommended

- Enable Kafka (local):
  - Kafka + Zookeeper + Spark are included in lightweight by default.
  - Host bootstrap: `localhost:29092` (inside containers use `kafka:9092`).
  - Enable publishing (host tools): `export KAFKA_BOOTSTRAP_SERVERS=localhost:29092`
  - Install client: `uv pip install confluent-kafka`
  - Create topics:
    - Print creation commands: `dspy-agent stream-topics`
    - Create inside container: `docker compose -f docker/lightweight/docker-compose.yml exec dspy-agent dspy-agent stream-topics-create --bootstrap kafka:9092`
    - Create from host: `dspy-agent stream-topics-create --bootstrap localhost:29092`
- Enable RedDB persistence:
  - `export REDDB_URL=http://localhost:8080` (and optionally `REDDB_NAMESPACE`, `REDDB_TOKEN`)
- Inspect latest agent outputs:
  - `dspy-code last --container app --what all`

Learn the Codebase

- Build a code graph and (optionally) embeddings so the agent deeply understands your repo:
  - `dspy-code learn --workspace $(pwd) --embeddings`
  - Writes `./.dspy_index/knowledge.json`, persists to storage (if configured), and prints a concise summary.
  - Status endpoint `GET /code` returns the stored graph + summary.

Streaming Engine (Kafka + Spark)

- Dataflow (lightweight):
  - Local tailer publishes `logs.raw.<container>` to Kafka (via LocalBus → Kafka).
  - Spark Structured Streaming job consumes `logs.raw.app`, aggregates error windows, and publishes `logs.ctx.app`.
  - You can run a Kafka worker to process `logs.ctx.<container>` and emit results to `agent.results.<container>`:
    - Inside container: `docker compose -f docker/lightweight/docker-compose.yml exec dspy-agent dspy-agent worker --topic app --bootstrap kafka:9092`
    - From host (not in Docker): `dspy-agent worker --topic app --bootstrap localhost:29092`
- Compose services:
  - `kafka`, `zookeeper`, and `spark` services are included and start with the stack.
  - Kafka is reachable at `kafka:9092` from containers and `localhost:29092` from the host.
  - Spark runs the job `scripts/streaming/spark_logs.py` with a checkpoint under `/workspace/.dspy_checkpoints/spark_logs`.
  - Smoke test:
    - Produce: `docker exec -i lightweight-kafka-1 /opt/bitnami/kafka/bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic logs.raw.backend <<< "hello from backend"`
    - Watch: `docker compose -f docker/lightweight/docker-compose.yml logs -f spark`
- Generate or customize Spark job:
  - `dspy-code spark-script --out scripts/streaming/spark_logs.py`
  - The default job filters for error keywords and emits context windows to Kafka.

Topic-Aware Orchestrator Training

- Goal: train the router to select the right tool/args per topic (e.g., backend vs. frontend).
- Dataset (JSONL), one per line:
  - `{ "query": "investigate API timeout", "workspace": "/abs/repo", "logs": "/abs/repo/logs/backend.log", "targets": ["timeout", "retry"], "topic": "backend" }`
  - `{ "query": "UI error on login", "workspace": "/abs/repo", "logs": "/abs/repo/logs/frontend.log", "targets": ["TypeError"], "topic": "frontend" }`
- Train:
  - `dspy-code gepa-orchestrator --train-jsonl data/orch_train.jsonl --auto light --ollama --model deepseek-coder:1.3b`
- Notes:
  - The training loader injects `topic=<name>` into the state field so the Orchestrator can condition decisions by topic.
  - Workers subscribe to `logs.ctx.<topic>` and publish to `agent.results.<topic>` so the router can aggregate results and persist per-topic summaries.

Notes & Tips

- If you pass unwritable paths to `lightweight_init`, the CLI automatically falls back to safe defaults (your current directory) and prints a yellow adjustments panel. You can always edit `docker/lightweight/docker-compose.yml` to mount a different host path.
- Kafka and RedDB are optional; the agent runs fine without them. When enabled, logs and results stream to Kafka topics and persist to RedDB KV/streams.

Links

- DSPy: https://github.com/stanfordnlp/dspy.git
- RedDB (open): https://github.com/redbco/redb-open.git


1) Ensure Python 3.10+ and uv are installed.
2) Configure an LLM backend (choose one):
   - Ollama (local, recommended): install Ollama and pull a model. For DeepSeek Coder 1.3B: `ollama pull deepseek-coder:1.3b`. Run with `--ollama` and `--model deepseek-coder:1.3b`. You may also set `USE_OLLAMA=true`, `OLLAMA_MODEL=deepseek-coder:1.3b`.
   - OpenAI-compatible server: set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` (LM Studio, etc.). Also set `MODEL_NAME`.
   - Offline preview: set `LOCAL_MODE=true` to run without an LLM (heuristics only).
3) Install deps:
   - `uv sync`

Usage

- Show context from logs:
  - `uv run dspy-code context --workspace /path/to/repo --logs ./logs --ollama --model deepseek-coder:1.3b`

- Propose plan and commands for a task using logs as context:
  - `uv run dspy-code run "fix failing tests" --workspace /path/to/repo --logs ./logs --ollama --model deepseek-coder:1.3b`

- Generate a code patch (unified diff) with rationale:
  - `dspy-code edit "refactor X to Y" --workspace /path/to/repo --files app/service.py --apply`
  - Builds context from logs if omitted and uses learned code summary.

Code Context

- Summarize a file or directory:
  - `dspy-code codectx --path src/ --workspace /path/to/repo --ollama --model deepseek-coder:1.3b`

Semantic Index

- Build index:
  - `dspy-code index --workspace /path/to/repo --smart`
- Search:
  - `dspy-code esearch "http client retry" --workspace /path/to/repo --k 5 --context 4`

Embeddings (optional)

- Option A: DSPy Embeddings provider (e.g., `openai/text-embedding-3-small`).
  - Build index: `dspy-code emb-index --workspace /path/to/repo --model openai/text-embedding-3-small --api-key $OPENAI_API_KEY`
  - Search: `dspy-code emb-search "retry logic" --workspace /path/to/repo --model openai/text-embedding-3-small --api-key $OPENAI_API_KEY`

- Option B: Local HuggingFace (recommended for Qwen 0.6B)
  - Install: `uv sync` (adds sentence-transformers>=2.7.0, transformers>=4.51.0)
  - Use Qwen/Qwen3-Embedding-0.6B locally:
    - Build index:
      - `dspy-code emb-index --workspace /path/to/repo --hf --model Qwen/Qwen3-Embedding-0.6B --device auto --flash`
    - Search:
      - `dspy-code emb-search "retry logic" --workspace /path/to/repo --hf --model Qwen/Qwen3-Embedding-0.6B --device auto --flash`
  - Notes:
    - `--device auto` will try GPU if available; use `--device cpu` to force CPU.
    - `--flash` enables flash_attention_2 when supported (GPU recommended).

Persist Embeddings + Chunks (RedDB)

- Build and persist:
  - `dspy-code emb_index --workspace $(pwd) --hf --model all-MiniLM-L6-v2 --persist`
- Inspect:
  - `dspy-code embeddings_inspect --start 0 --count 10`
  - Shows embedding metadata and the aligned code chunk text (via KV cache per hash).
- Compact chunks (deduplicate into KV):
  - `dspy-code chunks_compact --start 0 --count 1000`
  - Fills `code:chunk:<hash>` KV entries for fast lookups.

Interactive Session

- Start a REPL to pick workspace/logs and run tasks:
  - `uv run dspy-code start --workspace /path/to/repo --ollama --model deepseek-coder:1.3b`
- Inside the REPL:
  - You can type natural instructions; the agent will choose the best tools and arguments.
  - `cd <PATH>`: change workspace
  - `logs [PATH]`: show or set logs path
  - `ctx`: show context and enhanced context
  - `plan <TASK>`: propose plan and commands
  - `ls [PATH]`: list files under current workspace
  - `tree [PATH] [-d N] [--hidden]`: show directory tree (default depth 2)
  - `grep <PATTERN> [-f] [-c N] [-g GLOB]* [-x GLOB]* [-F FILE]`
  - `extract --file F [--symbol NAME | --re REGEX --before N --after N --nth K]`
  - `codectx [PATH]`: summarize code snapshot
  - `index`, `esearch <QUERY>`: build and search code index
  - `emb-index`, `emb-search <QUERY>`: embedding-based index and search (if configured)
  - `open <PATH>`: open a file in your editor / OS default
    - Supports `open path/to/file.py:42:7` for common editors (`code`, `subl`, `nvim`, `vim`, `emacs`, `idea`)
  - `patch <PATCHFILE>`: apply unified diff patch in workspace
  - `diff <FILE>`: show unified diff between last extract buffer and a file
  - `gstatus`, `gadd <PATHS...>`, `gcommit -m "message"`: basic git helpers
  - `write <PATH>`: write last extracted segment to file
  - `sg [-p PATTERN] [-l LANG] [-r RULE.yaml] [--json]`
  - `watch [-n SECS] [-t LINES]`: tail and refresh key events
  - `ollama on|off`, `model <NAME>`: set backend and model
  - `exit`: quit

Code Search

- Grep-style search over a folder:
  - `dspy-coder grep "def .*connect" --workspace /path/to/repo --glob "**/*.py" --context 2`
- Extract from a file:
  - Python symbol: `dspy-coder extract --file app/service.py --symbol connect`
  - Regex context: `dspy-coder extract --file app/service.py --regex "timeout after" --before 2 --after 4`
  - Save directly: add `--out snippets/connect.py`
- ast-grep integration (if installed):
  - `dspy-coder sg --pattern "function $A($B) { ... }" --lang js --root /path/to/repo`
  - Install ast-grep: `brew install ast-grep` or
    `curl -fsSL https://raw.githubusercontent.com/ast-grep/ast-grep/main/install.sh | bash`

Watching Logs

- One-off command:
  - `dspy-coder watch --workspace /path/to/repo --interval 2 --tail 20`
- From REPL:
  - `watch -n 2 -t 20` (Ctrl-C to stop)

Environment Variables

- `MODEL_NAME` (default: `gpt-4o-mini`): The model name for OpenAI-compatible backends (also used if you do not set Ollama variables).
- `OPENAI_API_KEY`: API key for OpenAI or compatible server.
- `OPENAI_BASE_URL`: Base URL for OpenAI-compatible servers (OpenAI/LM Studio, etc.). Not required for Ollama.
- `LOCAL_MODE` (true/false): If true, skip LLM calls and use heuristics.
- `USE_OLLAMA` (true/false): Force Ollama mode without passing `--ollama`.
- `OLLAMA_MODEL`: Default model for Ollama mode (e.g. `deepseek-coder:1.3b`).
- `OLLAMA_API_KEY`: Optional; any string is accepted. Defaults to `ollama` when needed.
- `DSPY_FORCE_JSON_OBJECT` (true/false): Force simple JSON output mode and skip structured-outputs (suppresses "Failed to use structured output format" warnings). Defaults to true for Ollama.
  - Toggle per-run with CLI `--force-json` (or prefer structured with `--structured`).

RedDB Integration (Local)

- Set `REDDB_URL` to enable RedDB-backed persistence for local streaming, or pass `--db reddb` to `up` to force it.
  - Example: `export REDDB_URL=http://localhost:8080` (adjust per RedDB deployment)
  - Optional: `REDDB_NAMESPACE` to prefix keys/streams (default `dspy`).
- When present (or `--db reddb`), `dspy-agent up` wires a RedDB storage adapter and persists:
  - Append-only event logs for LocalBus topics (e.g., `logs.raw.<container>`, `logs.ctx.<container>`, `agent.results.<container>`).
  - Key-value slots via the storage adapter API (future use).
- Notes:
  - The adapter is a thin stub awaiting the official client; it falls back to in-memory when the client is unavailable, so local dev is unaffected.
  - Dev/Prod stacks can reuse the same adapter once the RedDB client is finalized.
  - You can disable persistence with `--db none`.

Lightweight Containers (Local Dev)

- Generate a ready-to-run Docker setup that mounts your workspace and (optionally) logs:
  - Initialize:
    - `dspy-agent lightweight_init --workspace /absolute/path/to/your/repo --logs /absolute/path/to/your/logs --db auto`
  - This writes:
    - `docker/lightweight/Dockerfile`
    - `docker/lightweight/docker-compose.yml`
  - Next steps:
    - `docker compose -f docker/lightweight/docker-compose.yml build --no-cache`
    - `docker compose -f docker/lightweight/docker-compose.yml up -d`
    - `docker compose -f docker/lightweight/docker-compose.yml logs -f dspy-agent`
    - Interactive CLI: `docker compose -f docker/lightweight/docker-compose.yml exec -it dspy-agent dspy-agent start --workspace /workspace`
- Helpful commands:
  - `dspy-agent lightweight_up` (builds and starts; prints manual commands if Docker CLI is unavailable)
  - `dspy-agent lightweight_status` (equivalent of `docker compose ps`)
  - `dspy-agent lightweight_down` (stops the stack)
  - `dspy-agent lightweight_build` (rebuild image and optionally restart, with comprehensive logs)
  - Shell helper: `scripts/lightweight_build.sh [docker/lightweight/docker-compose.yml]`
- Container behavior:
  - Runs `dspy-agent up --workspace /workspace --db <auto|none|reddb>`.
  - Mounts your host repo to `/workspace` (rw). If `--logs` was provided, mounts it read-only at `/workspace/logs`.
  - Honors env vars (set them in your shell before `docker compose up`, or edit the compose env section):
    - `DB_BACKEND`, `REDDB_URL`, `REDDB_NAMESPACE`, `REDDB_TOKEN`
    - `LOCAL_MODE`, `MODEL_NAME`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OLLAMA_MODEL`
    - `KAFKA_BOOTSTRAP_SERVERS` (inside containers defaults to `kafka:9092`; from host use `localhost:29092`)
  - Status server: `http://localhost:8765` (routes: `/health`, `/deploy`, `/containers`)
  - Stop & clean up:
    - `docker compose -f docker/lightweight/docker-compose.yml down -v`
- Validation & robustness:
  - CLI validates paths and provides clear next steps; it never crashes the process if Docker is missing — it prints the exact commands to run manually.
  - LocalBus persistence is best-effort; failures won’t stop the agent.
  - Deployment logs are written under `logs/deployments/` and also appended to RedDB stream `deploy.logs.lightweight` when configured.
  - Quick inspect latest agent outputs: `dspy-agent last --container <name> --what all`

Deployment Data Model

- Streams (append-only):
  - `deploy.logs.lightweight`: all build/up/down/status output lines with `{ts, phase, level, message}`.
  - `deploy.events.lightweight`: reserved for higher-level structured events.
- KV (quick lookups):
  - `deploy:last:lightweight:status`: `pending|building|up|down|error|done`
  - `deploy:last:lightweight:image`: last built image tag/ID (reserved)
  - `deploy:last:lightweight:compose_hash`: SHA256 of the compose file used
  - `deploy:last:lightweight:ts`: last update timestamp

Kafka Topics (optional)

- If `KAFKA_BOOTSTRAP_SERVERS` is set and `confluent-kafka` is installed, the agent publishes JSON messages to Kafka:
  - `deploy.logs.lightweight` — mirrors the RedDB deploy log stream
  - All LocalBus topics (e.g., `logs.raw.*`, `logs.ctx.*`, `agent.results.*`)
- To create topics, use your Kafka tooling or leverage `dspy-agent stream_topics` to print creation commands for standard topics.
  - For deployment-only topics: `dspy-agent deploy_topics [--bootstrap localhost:29092]`
  - The lightweight Dockerfile tries to install `confluent-kafka`; if it fails, the agent still runs and simply disables Kafka publishing.
- Install Kafka client support:
  - `uv pip install confluent-kafka`
  - Or system install via your package manager; then `pip install confluent-kafka`.

Kafka Connectivity (host vs. container)

- Inside containers: use `kafka:9092`.
- From the host: use `localhost:29092`.

Status HTTP API

- `GET /health` → `{ "ok": true }`
- `GET /deploy` → `{ status, ts, compose_hash, image }`
- `GET /containers` → `{ names: ["..."], containers: { name: { summary, key_points, plan, ts } } }`

Project Layout

- `dspy_agent/cli.py` — Typer CLI.
- `dspy_agent/log_reader.py` — Read and extract key events from logs.
- `dspy_agent/skills/context_builder.py` — Build enhanced context (heuristic or DSPy-backed).
- `dspy_agent/skills/task_agent.py` — DSPy module to propose a plan and commands.
- `dspy_agent/config.py`, `dspy_agent/llm.py` — Configuration and LLM wiring.

Notes

- For fully local inference, point `OPENAI_BASE_URL` to a local OpenAI-compatible server (e.g., LM Studio) and provide a model via `MODEL_NAME`.
- Ollama exposes an OpenAI-compatible API on `http://localhost:11434/v1`. Use `--ollama` or set `USE_OLLAMA=true` and `OPENAI_BASE_URL` accordingly.
 - You can also launch via `dspy-coder` or `dspy_coder` commands (aliases of `dspy-agent`).

DeepSeek Coder 1.3B via Ollama

- Pull the model:
  - `ollama pull deepseek-coder:1.3b`
- Quick test:
  - `ollama run deepseek-coder:1.3b "Write a Python function to sum a list."`
- Run the agent with DeepSeek Coder:
  - `uv run dspy-agent context --logs ./logs --ollama --model deepseek-coder:1.3b`
  - `uv run dspy-agent chat "investigate test flakiness" --logs ./logs --ollama --model deepseek-coder:1.3b --force-json`

Alternate quick connect (DSPy snippet):

```python
import dspy
lm = dspy.LM("ollama_chat/deepseek-coder:1.3b", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)
```
Training With GEPA

- Prepare a JSONL dataset for the module you want to optimize.
- Supported modules and example schema (one JSON object per line):
  - context:
    - {"task": str, "logs_preview": str, "context_keywords": [str], "key_points_keywords": [str]}
  - task:
    - {"task": str, "context": str, "plan_keywords": [str], "commands_keywords": [str]}
  - code:
    - {"snapshot": str, "ask": str, "keywords": [str]}

- Run GEPA optimization (uses your LLM for reflection):
  - `dspy-code gepa-train --module task --train-jsonl data/task_train.jsonl --auto medium --ollama --model deepseek-coder:1.3b --log-dir .gepa_logs --track-stats`
  - Save best candidate program:
    - `dspy-code gepa-train --module context --train-jsonl data/context_train.jsonl --auto light --save-best prompts/context_best.json`

- Orchestrate tool-selection with GEPA (agent decides which command to run):
  - JSONL schema per line: `{ "query": str, "workspace": str, "logs": str|null, "targets": [str] }`
  - `dspy-coder gepa-orchestrator --train-jsonl data/orch_train.jsonl --auto light --ollama --model deepseek-coder:1.3b --log-dir .gepa_orch`
  - The metric executes safe evaluations of chosen actions (grep/extract/context/codectx/index/esearch/plan) and scores success; GEPA evolves the orchestration prompts using this feedback.

- Notes:
  - GEPA benefits from a strong reflection model; local small models work but may improve more slowly.
  - The metrics score keyword coverage and gently penalize risky shell commands for task plans.
  - Set `--auto light|medium|heavy` or use `--max-full-evals`/`--max-metric-calls` for budget control.
- Initialize on a repo (auto-dataset + optional light training):
  - `dspy-code init --workspace /path/to/repo --ollama --model deepseek-coder:1.3b --train --budget light`
  - Datasets written to `<repo>/.dspy_data`: `orch_train.jsonl`, `context_train.jsonl`, `code_train.jsonl`, `task_train.jsonl`.
  - Then you can run `gepa-orchestrator` or `gepa-train` explicitly for deeper training later.

- Auto-bootstrap:
  - `dspy-code init --workspace /path/to/repo --train --budget light`
  - Creates datasets and runs quick GEPA passes for orchestrator/context/code/task.

Codegen GEPA (Trainable Coding Agent)

- Goal: learn prompts that generate the best code patches for your repo.
- Dataset (JSONL), one per line: `{ "task": str, "context": str, "file_hints": str? }`
- Run training:
  - `dspy-code gepa-codegen --train-jsonl data/codegen_train.jsonl --workspace $(pwd) --test-cmd "pytest -q" --type-cmd "python -m compileall -q ."`
- Metric (composite):
  - Tests (0.6), Types/Syntax (0.25), Lint (0.15 if provided). Runs in a temporary copy of your repo.
  - Partial credit when some checks pass. Feedback includes test/type/lint outputs (tail) for quick iteration.
- Outputs:
  - Optimized program (prompt) for `CodeEdit` module (rationale + unified diff), callable by the agent.
