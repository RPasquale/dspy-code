# Infrastructure Architecture

Detailed architecture documentation for the DSPy agent infrastructure.

## Overview

The infrastructure consists of three main layers:

1. **Management Layer** (Go + Rust)
   - CLI (`dspy-agent`)
   - Orchestrator (task scheduling)
   - Environment Manager (container lifecycle)

2. **Service Layer** (Containers)
   - RedDB (database)
   - Redis (cache)
   - InferMesh (inference)
   - Ollama (LLM)

3. **Agent Layer** (Python)
   - DSPy programs
   - Agent logic
   - Infrastructure client

## Components

### 1. dspy-agent CLI (Go)

**Purpose:** Unified command-line interface

**Location:** `cmd/dspy-agent/`

**Responsibilities:**
- Start/stop all services with one command
- Configuration management
- Status monitoring
- Log aggregation
- Daemon mode support

**Key Files:**
- `main.go` - CLI commands
- `daemon.go` - Service lifecycle
- `config.go` - Configuration management

**Commands:**
- `start` - Start all services
- `stop` - Stop all services
- `status` - Show service status
- `logs` - View logs
- `config` - Manage configuration

### 2. Environment Manager (Rust)

**Purpose:** Container lifecycle management

**Location:** `env_manager_rs/`

**Responsibilities:**
- Docker API integration (Bollard crate)
- Service registry and definitions
- Health checking with exponential backoff
- Dependency resolution
- gRPC API for orchestrator

**Key Files:**
- `src/container.rs` - Docker operations
- `src/service_registry.rs` - Service definitions
- `src/health.rs` - Health checking
- `src/manager.rs` - Orchestration logic
- `src/grpc_server.rs` - gRPC API

**Services Managed:**
- `redis` - Cache (port 6379)
- `reddb` - Database (port 8080)
- `infermesh-node-a` - Inference (port 19001)
- `infermesh-node-b` - Inference (port 19002)
- `infermesh-router` - Load balancer (port 19000)
- `ollama` - Local LLM (port 11434)

**gRPC API:**
- `StartServices` - Start containers
- `StopServices` - Stop containers
- `GetServicesStatus` - Query status
- `StreamHealth` - Real-time health
- `PullImages` - Pull Docker images

### 3. Orchestrator (Go)

**Purpose:** Task and workflow orchestration

**Location:** `orchestrator/`

**Responsibilities:**
- HTTP API (existing)
- gRPC API (new)
- Task scheduling
- Workflow execution
- Resource management
- Event streaming

**Key Files:**
- `cmd/orchestrator/main.go` - Main entry point
- `internal/grpc/server.go` - gRPC server
- `internal/envmanager/client.go` - env_manager client
- `internal/envmanager/lifecycle.go` - Process management
- `internal/workflow/orchestrator.go` - Task orchestration

**APIs:**

**HTTP (port 9097):**
- `/queue/submit` - Submit tasks
- `/workflows` - Workflow management
- `/metrics` - Prometheus metrics

**gRPC (port 9098):**
- `SubmitTask` - Task submission
- `StreamTaskResults` - Task results stream
- `CreateWorkflow` - Workflow creation
- `GetMetrics` - System metrics
- `StreamEvents` - Event stream
- `Health` - Health check

### 4. Python Infrastructure Client

**Purpose:** Simple Python interface

**Location:** `dspy_agent/infra/`

**Responsibilities:**
- gRPC client wrapper
- AgentInfra context manager
- Automatic service detection
- Type-safe API

**Key Files:**
- `grpc_client.py` - gRPC communication
- `agent_infra.py` - High-level interface
- `__init__.py` - Public API

**Usage:**
```python
from dspy_agent.infra import AgentInfra

async with AgentInfra.start() as infra:
    await infra.submit_task("task-1", {"data": "value"})
```

## Communication Protocols

### gRPC (Primary)

**Why gRPC:**
- Bidirectional streaming
- Type-safe (Protocol Buffers)
- Efficient binary protocol
- Built-in load balancing
- Cross-language support

**Protocol Buffers:**
- `proto/orchestrator.v1.proto` - Python ↔ Go
- `proto/env_manager.v1.proto` - Go ↔ Rust

**Code Generation:**
```bash
# Go
protoc --go_out=. --go-grpc_out=. proto/*.proto

# Python
python -m grpc_tools.protoc -Iproto --python_out=. --grpc_python_out=. proto/*.proto

# Rust (via build.rs)
cargo build
```

### HTTP (Legacy)

Still supported for backward compatibility:
- Orchestrator HTTP API (port 9097)
- Direct container HTTP APIs

## Data Flow

### Task Submission

```
Python Agent
    │
    │ gRPC: SubmitTask
    ▼
Orchestrator (Go)
    │
    │ gRPC: GetResourceAvailability
    ▼
Environment Manager (Rust)
    │
    │ Docker API
    ▼
Container Execution
    │
    │ gRPC: TaskResult
    ▼
Orchestrator (Go)
    │
    │ gRPC Stream: TaskResult
    ▼
Python Agent
```

### Service Startup

```
dspy-agent CLI
    │
    │ Spawn Process
    ▼
Environment Manager (Rust)
    │
    │ 1. Resolve Dependencies
    │ 2. Start Containers (parallel)
    │ 3. Health Check
    ▼
Services Running
    │
    │ gRPC Status
    ▼
Orchestrator (Go)
    │
    │ gRPC Connect
    ▼
Python Agent
```

## Dependency Graph

```
┌──────────────┐
│  dspy-agent  │  (CLI - entry point)
└──────┬───────┘
       │
       ├─────────────┐
       │             │
       ▼             ▼
┌────────────┐  ┌─────────────┐
│env_manager │  │orchestrator │
│  (Rust)    │  │    (Go)     │
└──────┬─────┘  └──────┬──────┘
       │                │
       │  ┌─────────────┘
       │  │
       ▼  ▼
┌────────────────┐
│   Services     │
│ (Containers)   │
└────────────────┘
       ▲
       │
       │ gRPC
       │
┌──────────────┐
│ Python Agent │
└──────────────┘
```

## Service Dependencies

```
redis (no dependencies)
  ↓
reddb (no dependencies)
  ↓
infermesh-node-a → redis
  ↓
infermesh-node-b → redis
  ↓
infermesh-router → infermesh-node-a, infermesh-node-b
```

Startup order is automatically determined by env_manager.

## Health Checking

### Strategy

1. **Container Status**: Docker API
2. **HTTP Endpoints**: `/health` routes
3. **TCP Connectivity**: For services without HTTP
4. **Exponential Backoff**: 500ms → 10s max

### Health Check URLs

- RedDB: `http://localhost:8080/health`
- InferMesh Router: `http://localhost:19000/health`
- InferMesh Nodes: `http://localhost:1900[1-2]/health`
- Ollama: `http://localhost:11434/api/tags`
- Redis: TCP PING command

### Implementation

```rust
// env_manager_rs/src/health.rs
pub async fn wait_for_health(
    &self,
    service_name: &str,
    check_url: Option<&str>,
    max_attempts: u32,
) -> Result<()>
```

## Resource Management

### CPU/Memory

Managed by Docker:
- Container limits set per service
- Resource queries via Docker API
- Automatic scaling (future)

### GPU

- Detection via `nvidia-smi`
- Allocation tracked by env_manager
- Passed to orchestrator for scheduling

### Ports

**Automatic Resolution:**
1. Check configured port
2. If in use, try next port
3. Update configuration
4. Retry

**Default Ports:**
- RedDB: 8080
- Redis: 6379
- InferMesh Router: 19000
- InferMesh Node A: 19001
- InferMesh Node B: 19002
- Ollama: 11434
- Orchestrator HTTP: 9097
- Orchestrator gRPC: 9098
- env_manager: 50100

## Configuration

### Hierarchy

1. **Environment Variables** (highest priority)
2. **Config File** (`~/.dspy-agent/config.toml`)
3. **Defaults** (lowest priority)

### Configuration File

```toml
workspace = "."
orchestrator_addr = "localhost:9098"
env_manager_addr = "localhost:50100"
gpu = false

[services]
reddb = true
redis = true
infermesh-router = true
ollama = false

[ports]
reddb = 8080
redis = 6379
infermesh-router = 19000
ollama = 11434
orchestrator = 9098
env_manager = 50100
```

### Environment Variables

```bash
ENV_MANAGER_GRPC_ADDR=0.0.0.0:50100
ORCHESTRATOR_GRPC_ADDR=:9098
DOCKER_HOST=unix:///var/run/docker.sock
RUST_LOG=info
```

## Logging

### Structured Logging

All components use structured logging:

- **Rust** (env_manager): `tracing` crate
- **Go** (orchestrator): `log` package
- **Python** (agent): `logging` module

### Log Aggregation

```bash
# View all logs
dspy-agent logs

# Follow logs
dspy-agent logs --follow

# Specific service
dspy-agent logs --service redis

# Limit output
dspy-agent logs --tail 100
```

### Log Levels

- `ERROR`: Failures requiring attention
- `WARN`: Issues that don't prevent operation
- `INFO`: Normal operational messages
- `DEBUG`: Detailed diagnostic information

## Metrics

### Prometheus Metrics

Orchestrator exposes metrics at `http://localhost:9097/metrics`:

- `orchestrator_concurrency_limit`
- `orchestrator_inflight_tasks`
- `orchestrator_task_errors_total`
- `env_queue_depth`
- `gpu_wait_seconds`
- `env_error_rate`

### Custom Metrics

Via gRPC `GetMetrics`:
```python
metrics = await infra.get_metrics()
print(metrics)
# {'tasks_pending': 5, 'tasks_running': 3, ...}
```

## Security

### Current Status

**Development Mode:**
- No authentication
- Local-only connections
- Docker socket access required

### Future Enhancements

- TLS for gRPC
- Authentication tokens
- RBAC for API access
- Audit logging

## Performance

### Benchmarks

**Startup Time:**
- Cold start (image pull): 45-60s
- Warm start (images cached): 15-20s
- Service restart: 5-10s

**Task Throughput:**
- gRPC submission: 500+ tasks/sec
- Task execution: depends on workload

**Latency:**
- gRPC round-trip: < 1ms
- Task submission: < 5ms
- Status query: < 2ms

### Optimization

- Parallel container startup
- Connection pooling
- Streaming APIs
- Binary protocols (gRPC)

## Troubleshooting

### Debug Mode

```bash
# Enable debug logging
export RUST_LOG=debug
dspy-agent start

# Or
RUST_LOG=debug dspy-agent start
```

### Common Issues

1. **Port conflicts**: Check `dspy-agent config show`
2. **Docker access**: Check `docker ps`
3. **Service health**: Check `dspy-agent status`
4. **Logs**: Check `dspy-agent logs`

### Diagnostics

```bash
# System status
dspy-agent status

# Service logs
dspy-agent logs --service redis

# Configuration
dspy-agent config show

# Docker status
docker ps
docker stats
```

## Development

### Building

```bash
# Rust components
cd env_manager_rs
cargo build --release

# Go components
cd cmd/dspy-agent
go build

cd orchestrator
go build ./cmd/orchestrator
```

### Testing

```bash
# Integration tests
pytest tests/integration/ -v

# Specific tests
pytest tests/integration/test_grpc_infra.py::TestGRPCInfrastructure::test_health_check
```

### Contributing

See `CONTRIBUTING.md` for guidelines.

