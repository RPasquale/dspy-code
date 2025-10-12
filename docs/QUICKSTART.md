# Quick Start: New Infrastructure

Get up and running with DSPy agent in under 5 minutes.

## Prerequisites

- **Docker**: Running and accessible
- **Go**: 1.20+ (for building CLI)
- **Rust**: 1.70+ (for building env_manager)
- **Python**: 3.11+ (for agent)

Check prerequisites:
```bash
docker ps              # Should list containers
go version            # Should show 1.20+
rustc --version       # Should show 1.70+
python --version      # Should show 3.11+
```

## Installation

### Option 1: Quick Build (Recommended)

```bash
# 1. Build environment manager (Rust)
cd env_manager_rs
cargo build --release
cd ..

# 2. Build unified CLI (Go)
cd cmd/dspy-agent
go build -o dspy-agent
sudo mv dspy-agent /usr/local/bin/  # Optional: install globally
cd ../..

# 3. Initialize configuration
dspy-agent config init
```

### Option 2: Development Build

```bash
# Use Makefile (coming soon)
make build-infra

# Or build everything
make all
```

## First Run

### 1. Start Infrastructure

```bash
# Start all services
dspy-agent start

# Expected output:
# → Starting environment manager...
# ✓ Environment manager started
# → Starting infrastructure services...
#   ✓ redis: running
#   ✓ reddb: running
#   ✓ infermesh-router: running
# → Starting orchestrator...
# ✓ Orchestrator started
# ✅ DSPy agent infrastructure is ready!
```

This takes approximately **15-20 seconds**.

### 2. Verify Services

```bash
# Check status
dspy-agent status

# Expected output:
# Service              Status      Container ID
# ------------------   ---------   ----------------------------------
# redis                🟢 running   a1b2c3d4e5f6
# reddb                🟢 running   f6e5d4c3b2a1
# infermesh-router     🟢 running   1234567890ab
```

### 3. Use from Python

Create `test_agent.py`:
```python
import asyncio
from dspy_agent.infra import AgentInfra

async def main():
    # Start infrastructure (connects to running services)
    async with AgentInfra.start() as infra:
        print("✓ Infrastructure connected")
        
        # Submit a task
        result = await infra.submit_task(
            task_id="hello-world",
            payload={"message": "Hello from DSPy!"},
            task_class="cpu_short"
        )
        
        print(f"✓ Task submitted: {result}")
        
        # Check system health
        health = await infra.health_check()
        print(f"✓ System healthy: {health}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python test_agent.py

# Expected output:
# ✓ Infrastructure connected
# ✓ Task submitted: {'success': True, 'task_id': 'hello-world'}
# ✓ System healthy: {'healthy': True, 'version': '0.1.0', ...}
```

> **Tip:** point `AgentInfra` at a custom binary or address with:
> ```bash
> export DSPY_AGENT_CLI_PATH=/path/to/dspy-agent
> export ORCHESTRATOR_GRPC_ADDR=127.0.0.1:50052
> export DSPY_AGENT_SKIP_START=1      # Use if services are already running
> ```
> These variables are optional—by default the Python API will launch `dspy-agent start` for you.

### 4. Stop Services

```bash
# Graceful shutdown
dspy-agent stop

# Expected output:
# Stopping DSPy agent infrastructure...
# ✓ All services stopped
```

## Common Commands

```bash
# Start services
dspy-agent start

# Start in background (daemon mode)
dspy-agent start --daemon

# Check status
dspy-agent status

# View logs
dspy-agent logs
dspy-agent logs --follow
dspy-agent logs --service redis

# Stop services
dspy-agent stop

# Configuration
dspy-agent config show
dspy-agent config init

# Help
dspy-agent --help
dspy-agent start --help
```

## Next Steps

### Enable GPU Support

```bash
# Start with GPU services
dspy-agent start --gpu
```

### Customize Configuration

```bash
# Edit configuration file
vim ~/.dspy-agent/config.toml

# Restart services
dspy-agent stop
dspy-agent start
```

### Explore Examples

```bash
# Run example agents
cd examples
python basic_agent.py
python rl_training.py
python graph_search.py
```

### Development Workflow

```bash
# Start services once
dspy-agent start

# Develop your agent
# ... edit code ...
python my_agent.py

# Services keep running
# ... edit more ...
python my_agent.py

# Stop when done
dspy-agent stop
```

## Troubleshooting

### Services Won't Start

```bash
# Check Docker
docker ps

# Check ports
lsof -i :8080    # RedDB
lsof -i :6379    # Redis
lsof -i :9098    # Orchestrator

# View detailed logs
dspy-agent logs --follow
```

### Build Errors

```bash
# Rust build fails - update toolchain
rustup update

# Go build fails - update dependencies
cd cmd/dspy-agent
go mod tidy
go build
```

### Connection Errors

```bash
# Verify services are running
dspy-agent status

# Check network
docker network ls | grep dspy

# Restart services
dspy-agent stop
dspy-agent start
```

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  dspy-agent CLI                 │
│              (Go - Single Binary)               │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────────┐    ┌──────────────────┐
│   env_manager     │    │   orchestrator   │
│   (Rust)          │◄──►│   (Go)           │
│   - Containers    │    │   - Tasks        │
│   - Health        │    │   - Workflows    │
└───────┬───────────┘    └─────────┬────────┘
        │                          │
        │ manages                  │ schedules
        ▼                          ▼
┌────────────────────────────────────────────────┐
│           Docker Containers                    │
│  ┌──────┐  ┌──────┐  ┌────────────┐          │
│  │Redis │  │RedDB │  │ InferMesh  │  ...     │
│  └──────┘  └──────┘  └────────────┘          │
└────────────────────────────────────────────────┘
        ▲
        │ uses via gRPC
        │
┌───────────────────┐
│  Python Agent     │
│  (Your Code)      │
│  - AgentInfra     │
│  - DSPy Programs  │
└───────────────────┘
```

## Performance

Typical startup times:

- **First run** (with image pulls): 45-60s
- **Subsequent runs** (images cached): 15-20s
- **With services already running**: < 2s

Compared to old approach:

- **Old**: 60-90s, 7+ manual steps
- **New**: 15-20s, 1 command
- **Improvement**: 3-4x faster, 7x simpler

## Getting Help

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory
- **CLI Help**: `dspy-agent --help`
- **Logs**: `dspy-agent logs`
- **Status**: `dspy-agent status`
- **GitHub Issues**: Report problems
- **Migration Guide**: `docs/MIGRATION.md`

## What's Next?

1. ✅ Infrastructure running
2. 📝 **Read**: Full documentation in `README.md`
3. 🔧 **Customize**: Edit `~/.dspy-agent/config.toml`
4. 🎯 **Build**: Create your DSPy agent
5. 🚀 **Deploy**: Use `dspy-agent start --daemon`

Welcome to streamlined DSPy development!

