# DSPy Code Agent

## Quick Start

### Option 1: Install and Use Anywhere
```bash
# Install the agent
pipx install dspy-code            # or: pip install dspy-code

# Install a local LLM (recommended)
ollama pull qwen3:1.7b            # pick any Ollama model you like

# Start the agent in your project
dspy-agent --workspace $(pwd)
```

### Option 2: From Source
```bash
# Clone and setup
git clone <your-repo>
cd dspy_stuff
pip install uv
uv sync

# Quick start (recommended)
./scripts/quick-start.sh

# Or start manually
uv run dspy-agent --workspace $(pwd)
```

### Need the dashboard too?
```bash
dspy-agent code --open-dashboard   # launches web UI + CLI together
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
dspy-agent lightweight_init --workspace $(pwd) --logs ./logs --out-dir docker/lightweight
cd docker/lightweight
DOCKER_BUILDKIT=1 docker compose up -d
```
That spins up Kafka, Spark, Ollama, and the agent in one go.

## Development & Testing

### Quick Development Setup
```bash
./scripts/deploy-dev.sh            # Full development environment
```

### Run All Tests
```bash
./scripts/run_all_tests.py         # Comprehensive test suite
uv run pytest                      # Unit tests only
```

### Deployment Options
```bash
./scripts/deploy-test.sh           # Test environment validation
./scripts/deploy-prod.sh           # Production deployment
```

### Manual Development
```bash
uv venv                           # create virtual environment
uv sync                           # install dependencies
uv pip install -e .              # install editable copy
uv run dspy-agent --workspace $(pwd)  # start agent
```

Entry points live in `dspy_agent/cli.py`. Most tools are in `dspy_agent/skills` and `dspy_agent/code_tools`.

## Documentation

- **Usage Guide**: See `USAGE_GUIDE.md` for comprehensive usage instructions
- **API Documentation**: See `docs/API.md` for detailed API reference
- **Deployment Guide**: See `docs/DEPLOYMENT.md` for deployment instructions

## License
Apache 2.0 — see `LICENSE` for details.
