# Blampert - AI Coding Assistant

## Quick Start

### Option 1: Install and Use Anywhere
```bash
# Install the agent
pipx install blampert             # or: pip install blampert
# or: uv tool install blampert    # installs with Astral's uv

# Install a local LLM (recommended)
ollama pull qwen3:1.7b            # pick any Ollama model you like

# Launch from any directory (creates ./logs if needed)
blampert

# Target another project without cd-ing
blampert --workspace /path/to/project

# Need the dashboard too?
blampert --open-dashboard

# (Optional) Bootstrap the Docker stack (Kafka/Spark/RedDB)
blampert stack init
```

### Option 2: From Source
```bash
# Clone and setup
git clone <your-repo>
cd dspy_stuff
pip install uv
uv sync

# Quick start (recommended)
./scripts/start-agent.sh

# Or use the full setup
./scripts/quick-start.sh

# Or start manually
uv run blampert --workspace $(pwd)
```

### Option 3: One-Command Bootstrap (Windows/macOS/Linux)
```bash
python scripts/run_agent.py
```
- Handles virtualenv creation, `uv sync`, and sensible defaults.
- Pass `--workspace /path/to/your/project` to run the agent against another repo.
- Need a dashboard too? `python scripts/run_agent.py --open-dashboard` launches CLI + UI together.
- Want a dry run? `python scripts/run_agent.py --check-only` verifies prerequisites and exits.
- Prefer a packaged binary? `dspy-run` is an alias for `dspy-code` (both installed via pip/pipx).

## Spin Up the Full Stack
```bash
# After installing the package
dspy-code stack init
```
- Generates a ready-to-use Docker Compose bundle under `~/.dspy/stack` by default.
- Automatically builds and starts Kafka, Spark, Ollama, and the agent (use `--no-start` to skip).
- Manage the services later with `dspy-code stack up`, `stack down`, `stack status`, and `stack logs`.

### React Monitoring Dashboard
```bash
cd frontend/react-dashboard
npm install               # install dependencies
npm run dev               # local development server (http://localhost:5173)
```

Build the static assets and serve them through the Python dashboard server:
```bash
npm run build                         # outputs to frontend/react-dashboard/dist
python3 enhanced_dashboard_server.py  # serves the React app + APIs on :8080
```

The React UI consolidates the legacy HTML dashboards (status, monitoring, advanced analytics, system map) into a single application. All API routes continue to be served from `enhanced_dashboard_server.py`.

### Need the dashboard too?
```bash
blampert code --open-dashboard   # launches web UI + CLI together
```

## Use It In A Session
The agent prints a command menu on launch. Common moves:
- `plan "fix the failing tests"` → get a task plan
- `grep <regex>` or `esearch "vector search"` → find code
- `edit "update the API route" --apply` → propose & apply a patch
- `ollama off` / `model gpt-4o-mini` → switch LLM providers mid-run

Everything is logged under `.dspy_rl_state.json` and `.dspy_rl_events.jsonl` so the bandit keeps learning.

## Pointing At Your LLM
- **Ollama (default)**: export `USE_OLLAMA=true` and set `OLLAMA_MODEL`. The CLI falls back to `http://localhost:11434`.
- **OpenAI-compatible**: export `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `MODEL_NAME`, then run the same commands as above.

## Optional: Full Docker Stack
Want Kafka + Spark streaming plus the agent?
```bash
blampert lightweight_init --workspace $(pwd) --logs ./logs --out-dir docker/lightweight
cd docker/lightweight
DOCKER_BUILDKIT=1 docker compose up -d
```
That spins up Kafka, Spark, Ollama, and the agent in one go.

## Real-Time Vectorized Streaming
- The local stack now emits RL-ready feature vectors on the `agent.rl.vectorized` topic. Every log context and agent result is transformed into a fixed-length vector so the trainer can consume it without extra parsing.
- `LocalBus.vector_metrics()` returns current throughput, while `LocalBus.kafka_health()` and `LocalBus.spark_health()` surface Kafka metadata and Spark checkpoint freshness. These methods power dashboards and can be polled from custom monitors.
- Set `DSPY_VECTOR_TOPICS=topic1,topic2` to add extra Kafka topics to the vectorization pipeline. Long-term defaults can be captured in `.dspy_stream.json` via `kafka.vector_topic` (output) and `kafka.vector_topics` (additional sources). Live feature statistics are maintained in-memory through `LocalBus.feature_snapshot()` for fast RL state queries.
- Trainers automatically aggregate these vectors during every batch; you can inspect recent averages in the stored training metrics (`vector_features` payload) or by tailing the `agent.rl.vectorized` stream.

## CLI-Aware Agent
- The RL toolchain exposes native shell actions (`shell_ls`, `shell_pwd`, `shell_cat`, `shell_cd`, `shell_run`) so the agent can inspect the workspace, move around directories, and execute ad-hoc commands safely.
- Default shell commands and timeouts are configurable via `shell_actions` in `.dspy_stream_rl.json` or environment variables such as `RL_SHELL_LS`, `RL_SHELL_RUN`, and `RL_SHELL_TIMEOUT`. The trainer automatically merges these into its action set, and the toolchain executor keeps commands sandboxed to the workspace.
- The executor constrains `cd` targets to the active workspace, while outputs are streamed back through Kafka for learning and dashboards.

## Online Optimisation Loop
- The streaming trainer consumes both contextual log windows and the live feature store snapshot, fusing mean/std/min/max statistics into the RL observation space.
- Training metrics now capture `vector_variances` alongside batch counts, providing deeper visibility into data drift.
- Gepa prompt signatures inherit streaming metrics so successful shell/patch runs propagate into future planning automatically.


## Development & Testing

### Quick Development Setup
```bash
./scripts/deploy-dev.sh            # Full development environment
```

### Run All Tests
```bash
./scripts/run_all_tests.py         # Comprehensive test suite (includes RL)
./scripts/test_agent_simple.py     # Simple functionality test
./scripts/test_rl.py               # RL components test
uv run pytest                      # Unit tests only
uv run pytest tests/test_rl_tooling.py  # RL unit tests
```

### Deployment Options
```bash
./scripts/deploy-test.sh           # Test environment validation
./scripts/deploy-prod.sh           # Production deployment
```

### Publish to PyPI
```bash
./scripts/publish_package.sh --dry-run      # build artifacts only
UV_PUBLISH_TOKEN=your-token ./scripts/publish_package.sh
```
Use `--skip-tests` or `--allow-dirty` if needed when iterating locally.

### Manual Development
```bash
uv venv                           # create virtual environment
uv sync                           # install dependencies
uv pip install -e .              # install editable copy
uv run blampert --workspace $(pwd)  # start agent
```

Entry points live in `dspy_agent/cli.py`. Most tools are in `dspy_agent/skills` and `dspy_agent/code_tools`.

## Documentation

- **Usage Guide**: See `USAGE_GUIDE.md` for comprehensive usage instructions
- **Learning Guide**: See `AGENT_LEARNING_GUIDE.md` for how to make the agent learn and improve
- **Testing Guide**: See `docs/TESTING.md` for testing and development guidelines
- **API Documentation**: See `docs/API.md` for detailed API reference
- **Deployment Guide**: See `docs/DEPLOYMENT.md` for deployment instructions

## License
Apache 2.0 — see `LICENSE` for details.
