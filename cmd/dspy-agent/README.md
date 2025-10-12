# dspy-agent CLI

Unified command-line interface for managing DSPy agent infrastructure.

## Overview

The `dspy-agent` CLI provides a simple, single-command interface to manage all infrastructure services required for the DSPy agent, including:

- **Environment Manager** (Rust): Container lifecycle management
- **Orchestrator** (Go): Task scheduling and workflow execution
- **RedDB**: Lightweight database
- **Redis**: Cache and pub/sub
- **InferMesh**: Inference services
- **Ollama**: Optional local LLM

## Installation

### Build from Source

```bash
cd cmd/dspy-agent
go build -o dspy-agent
```

### Install Globally

```bash
cd cmd/dspy-agent
go install
```

## Usage

### Start All Services

```bash
# Start with default configuration
dspy-agent start

# Start with GPU support
dspy-agent start --gpu

# Start in daemon mode
dspy-agent start --daemon

# Specify workspace
dspy-agent start --workspace /path/to/workspace
```

### Stop All Services

```bash
# Graceful shutdown
dspy-agent stop

# Force shutdown
dspy-agent stop --force

# Custom timeout
dspy-agent stop --timeout 60
```

### Check Status

```bash
# Show status of all services
dspy-agent status
```

### View Logs

```bash
# Show logs from all services
dspy-agent logs

# Follow logs
dspy-agent logs --follow

# Show specific service
dspy-agent logs --service reddb

# Limit output
dspy-agent logs --tail 50
```

### Configuration

```bash
# Initialize configuration file
dspy-agent config init

# Show current configuration
dspy-agent config show
```

Configuration is stored at `~/.dspy-agent/config.toml`.

## Architecture

```
dspy-agent CLI
├── Starts env_manager (Rust)
│   ├── Manages Docker containers
│   ├── Performs health checks
│   └── Resolves dependencies
├── Starts orchestrator (Go)
│   ├── Schedules tasks
│   ├── Executes workflows
│   └── Provides gRPC API
└── Provides status/control interface
```

## Commands

- `start`: Start all infrastructure services
- `stop`: Stop all services
- `status`: Show service status
- `logs`: View service logs
- `config`: Manage configuration
  - `init`: Create default config
  - `show`: Display current config
- `version`: Show version info

## Environment Variables

- `DOCKER_HOST`: Docker socket/endpoint
- `DSPY_WORKSPACE`: Default workspace directory
- `ORCHESTRATOR_ADDR`: Orchestrator gRPC address
- `ENV_MANAGER_ADDR`: Environment manager address

## Examples

### Quick Start

```bash
# Initialize configuration
dspy-agent config init

# Start everything
dspy-agent start

# Check that services are running
dspy-agent status

# Stop when done
dspy-agent stop
```

### Development Workflow

```bash
# Start in foreground (see logs)
dspy-agent start

# In another terminal, check status
dspy-agent status

# View logs
dspy-agent logs --follow

# Stop with Ctrl+C or:
dspy-agent stop
```

### Production Deployment

```bash
# Start as daemon
dspy-agent start --daemon

# Monitor status
watch dspy-agent status

# View logs if needed
dspy-agent logs --service orchestrator --tail 100
```

## Troubleshooting

### Services Won't Start

1. Check Docker is running:
   ```bash
   docker ps
   ```

2. Check for port conflicts:
   ```bash
   dspy-agent config show
   ```

3. Try stopping and restarting:
   ```bash
   dspy-agent stop
   dspy-agent start
   ```

### Can't Connect to Services

1. Verify services are running:
   ```bash
   dspy-agent status
   ```

2. Check logs for errors:
   ```bash
   dspy-agent logs
   ```

### Configuration Issues

1. Reset to defaults:
   ```bash
   rm ~/.dspy-agent/config.toml
   dspy-agent config init
   ```

## Integration with Python Agent

The `dspy-agent` CLI is designed to work seamlessly with the Python agent:

```python
# Python code automatically connects to services started by dspy-agent
from dspy_agent.infra import AgentInfra

async with AgentInfra.start() as infra:
    # All services already running via dspy-agent
    result = await infra.submit_task("task-1", {"data": "value"})
```

## Development

To modify the CLI:

1. Edit source files in `cmd/dspy-agent/`
2. Rebuild: `go build`
3. Test: `./dspy-agent start`

Dependencies are managed via `go.mod`.

