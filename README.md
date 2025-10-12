# DSPy Agent

## Cross-Platform Quick Start

The stack now ships with a small Python helper that wraps all of the `docker compose`
invocations. It works on macOS, Linux, and Windows (PowerShell / CMD) as long as
Docker Desktop/Engine and Python 3.9+ are installed.

```bash
# 1. Create the docker/.env file and workspace defaults
python scripts/manage_stack.py init

# 2. Build the images (add service names to build a subset)
python scripts/manage_stack.py build

# 3. Start everything in the background
python scripts/manage_stack.py up

# Helpful extras
python scripts/manage_stack.py ps         # show container status
python scripts/manage_stack.py logs -f    # tail logs (add a service name to filter)
python scripts/manage_stack.py down       # stop the stack (add --volumes to wipe data)
```

The script mirrors the existing Makefile targets internally, so `make stack-env` and the
other recipes continue to work unchanged. Use whichever interface fits your shell environment.

### One-command agent session

Once the project is installed with `uv`, simply run:

```bash
uv run dspy-code            # starts all services, preloads Ollama models, drops into the agent CLI
uv run dspy-code -- --gpu   # same as above but enables NVIDIA GPU scheduling (needs container toolkit)
```

`dspy-code` accepts the following subcommands:

```bash
dspy-code start [--gpu] [--no-attach] [agent args...]   # default; boots the stack and opens an agent session
dspy-code attach [agent args...]                # attach to an already-running stack
dspy-code status                                # docker compose ps
dspy-code logs [-f] [service ...]               # stream logs
dspy-code stop [--volumes]                      # stop the stack
```

All dependencies (Kafka, Redis, RL services, streaming workers, dashboards, and Ollama with
`deepseek-coder:1.3b` plus `qwen3:1.7b`) are brought up automatically. The models are pulled the
first time you run the command and cached for subsequent launches.

After the stack is up, verify Kafka connectivity from inside the embeddings indexer container:

```bash
docker compose -f docker/lightweight/docker-compose.yml exec emb-indexer bash -lc 'source /entrypoints/wait_for_kafka.sh'
```

> Need the legacy “direct” CLI without Docker? Use `uv run dspy-cli --help`.

## Build

```bash
# Build all components
make build

# Or build individual components
make build-go      # Build Go orchestrator
make build-rust     # Build Rust components
make build-python   # Build Python components
```

## Run

```bash
# Start all services
make up

# Or start specific services
make up-core       # Start core services only
make up-full       # Start all services including monitoring
```

## Development

```bash
# Start development environment
make dev

# Run tests
make test

# Check service health
make health

# View service logs
make logs
```
