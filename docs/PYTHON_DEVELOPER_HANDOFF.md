# Python Developer Handoff: Infrastructure Modernization

**Date:** October 12, 2025  
**Purpose:** Complete infrastructure overhaul to streamline DSPy agent startup and operations

---

## 🎯 Executive Summary

We are replacing the complex multi-step Docker Compose setup with a **unified single-command infrastructure** powered by:
- **Rust** for high-performance environment/container management
- **Go** for orchestration and workflow management  
- **gRPC** for efficient inter-service communication
- **Python** remains the core agent logic (DSPy, LLMs, reasoning)

### Before vs After

| Aspect | Old System | New System |
|--------|-----------|------------|
| **Startup** | 7+ manual steps, 60-90s | `dspy-agent start`, 15-20s |
| **Container Mgmt** | docker-compose | Rust `env-manager` (direct Docker API) |
| **Orchestration** | Python subprocess calls | Go orchestrator with gRPC |
| **Communication** | HTTP polling | gRPC streaming |
| **Services** | Manual coordination | Automatic dependency resolution |

---

## 📁 New File Structure

```
dspy-code/
├── proto/                          # NEW: Protocol definitions
│   ├── orchestrator.v1.proto       # Go ↔ Python communication
│   ├── env_manager.v1.proto        # Go ↔ Rust communication
│   ├── buf.yaml                    # Protobuf config
│   └── buf.gen.yaml                # Code generation config
│
├── env_manager_rs/                 # NEW: Rust environment manager
│   ├── src/
│   │   ├── main.rs                 # Entry point
│   │   ├── container.rs            # Docker API wrapper
│   │   ├── manager.rs              # Service lifecycle management
│   │   ├── service_registry.rs    # Service definitions (RedDB, InferMesh, etc.)
│   │   ├── health.rs               # Health checking
│   │   └── grpc_server.rs          # gRPC service implementation
│   ├── Cargo.toml                  # Rust dependencies
│   ├── build.rs                    # Proto compilation
│   └── README.md
│
├── orchestrator/                   # UPDATED: Go orchestrator
│   ├── internal/
│   │   ├── grpc/
│   │   │   └── server.go           # NEW: gRPC server for Python clients
│   │   ├── envmanager/
│   │   │   ├── client.go           # NEW: gRPC client for Rust env-manager
│   │   │   └── lifecycle.go        # NEW: High-level env management
│   │   ├── pb/
│   │   │   ├── orchestrator/       # Generated from orchestrator.v1.proto
│   │   │   └── envmanager/         # Generated from env_manager.v1.proto
│   │   └── workflow/               # Existing orchestrator logic
│   └── cmd/orchestrator/
│       ├── main.go                 # UPDATED: Now includes gRPC server
│       └── grpc_integration.go     # NEW: gRPC startup logic
│
├── cmd/dspy-agent/                 # NEW: Unified CLI
│   ├── main.go                     # Entry point, cobra CLI
│   ├── daemon.go                   # Manages Rust env-manager daemon
│   ├── config.go                   # Configuration management
│   ├── go.mod
│   └── README.md
│
├── dspy_agent/
│   ├── infra/                      # NEW: Infrastructure module
│   │   ├── __init__.py
│   │   ├── agent_infra.py          # Context manager for all services
│   │   ├── grpc_client.py          # gRPC client for Go orchestrator
│   │   └── pb/                     # Generated Python stubs
│   │       ├── __init__.py
│   │       └── orchestrator/       # Generated from orchestrator.v1.proto
│   │
│   └── [existing modules remain unchanged]
│
├── tests/integration/              # NEW: Integration tests
│   ├── test_grpc_infra.py          # Test gRPC communication
│   ├── test_startup.py             # Test unified startup
│   └── pytest.ini
│
├── docs/
│   ├── QUICKSTART.md               # NEW: 5-minute getting started
│   ├── MIGRATION.md                # NEW: Migration from old setup
│   ├── INFRASTRUCTURE.md           # NEW: Detailed architecture
│   ├── IMPLEMENTATION_SUMMARY.md   # NEW: What was built
│   └── PYTHON_DEVELOPER_HANDOFF.md # THIS FILE
│
├── BUILD_INSTRUCTIONS.md           # NEW: Complete build guide
├── .cursor/settings.json           # NEW: WSL terminal config
└── .gitignore                      # UPDATED: Added .cursor/

```

---

## 🔧 Key Technical Changes

### 1. **Protocol Definitions (gRPC/Protobuf)**

#### `proto/orchestrator.v1.proto`
Defines communication between **Python agent** and **Go orchestrator**:

```protobuf
service OrchestratorService {
  rpc SubmitTask (SubmitTaskRequest) returns (SubmitTaskResponse);
  rpc GetTaskStatus (GetTaskStatusRequest) returns (GetTaskStatusResponse);
}
```

**Python Usage:**
```python
from dspy_agent.infra import OrchestratorClient

client = OrchestratorClient(host="localhost", port=50052)
response = await client.submit_task("task-1", "inference", {"prompt": "hello"})
status = await client.get_task_status("task-1")
```

#### `proto/env_manager.v1.proto`
Defines communication between **Go orchestrator** and **Rust env-manager**:

```protobuf
service EnvManagerService {
  rpc StartServices(StartServicesRequest) returns (stream ServiceStatusUpdate);
  rpc StopServices(StopServicesRequest) returns (StopServicesResponse);
  rpc GetServicesStatus(GetServicesStatusRequest) returns (ServicesStatusResponse);
  rpc StreamHealth(StreamHealthRequest) returns (stream HealthUpdate);
  rpc ExecuteTask(ExecuteTaskRequest) returns (ExecuteTaskResponse);
  // ... more RPCs
}
```

**Go Usage:**
```go
envClient := envmanager.NewGrpcClient("localhost:50051")
resp, err := envClient.StartEnvironment(ctx, "reddb", config)
```

---

### 2. **Rust Environment Manager** (`env_manager_rs/`)

**Responsibility:** Direct Docker container lifecycle management

**Key Features:**
- **Service Registry:** Pre-configured definitions for RedDB, InferMesh, Ollama, Redis, etc.
- **Dependency Resolution:** Automatically starts services in correct order (e.g., Redis before RedDB)
- **Health Checking:** Built-in health probes for all services
- **Resource Management:** CPU/memory limits, GPU allocation
- **gRPC Server:** Port 50051 (configurable via `ENV_MANAGER_GRPC_ADDR`)

**Services Managed:**
```rust
// Pre-configured in service_registry.rs
- reddb: SQLite-backed storage (port 6380)
- redis: Cache (port 6379)
- infermesh: Inference gateway (port 8000)
- ollama: Local LLM runtime (port 11434)
- model_server: Custom model serving (port 8001)
```

**Example gRPC Call:**
```python
# Python code will NOT call this directly
# The Go orchestrator handles env-manager communication
```

**Binary Location After Build:**
```
env_manager_rs/target/release/env-manager
```

---

### 3. **Go Orchestrator Updates**

**New Capabilities:**
- **gRPC Server:** Exposes `OrchestratorService` on port 50052
- **EnvManager Client:** Connects to Rust env-manager via gRPC
- **Task Queueing:** Enhanced with gRPC streaming support
- **Lifecycle Management:** Automatically starts/stops env-manager

**Important Ports:**
- `50051`: Rust env-manager gRPC
- `50052`: Go orchestrator gRPC (Python connects here)

**Integration Point:**
```go
// cmd/orchestrator/main.go
func main() {
    // Start gRPC server for Python clients
    startGrpcServer(ctx, orchestrator, ":50052")
    
    // Connect to env-manager
    envClient := envmanager.NewGrpcClient("localhost:50051")
    
    // Start services
    envClient.StartEnvironment(ctx, "reddb", config)
}
```

---

### 4. **Unified CLI** (`cmd/dspy-agent/`)

**Purpose:** Single binary that manages everything

**Commands:**
```bash
dspy-agent start     # Starts env-manager + orchestrator + all services
dspy-agent stop      # Graceful shutdown of all components
dspy-agent status    # Show service status
dspy-agent logs      # Stream logs from services
dspy-agent config    # Manage configuration
```

**What `dspy-agent start` Does:**
1. Spawns `env-manager` as a background daemon
2. Waits for env-manager to be ready (health check)
3. Starts Go orchestrator with gRPC server
4. Triggers service startup (RedDB, InferMesh, etc.)
5. Waits for all services to be healthy
6. Returns control (services run in background)

**Configuration (Environment Variables):**
```bash
ORCHESTRATOR_GRPC_ADDR=0.0.0.0:50052
ENV_MANAGER_GRPC_ADDR=0.0.0.0:50051
ENV_MANAGER_DOCKER_HOST=unix:///var/run/docker.sock
```

---

### 5. **Python Infrastructure Module** (`dspy_agent/infra/`)

**This is what Python developers interact with!**

#### `agent_infra.py`
Context manager for entire infrastructure:

```python
from dspy_agent.infra import AgentInfra

# Simple usage
async def main():
    async with AgentInfra.start() as infra:
        # All services (RedDB, InferMesh, etc.) are now running
        
        # Submit tasks to orchestrator
        result = await infra.submit_task(
            task_id="train-1",
            task_type="rl_training",
            payload={"model": "gpt2", "episodes": 100}
        )
        
        # Check task status
        status = await infra.get_task_status("train-1")
        print(f"Status: {status.status}")  # PENDING, RUNNING, COMPLETED, FAILED
        
        # Result is available when COMPLETED
        if status.status == TaskStatus.COMPLETED:
            result_data = status.result_payload.decode('utf-8')

asyncio.run(main())
# Services automatically stop when exiting context
```

#### `grpc_client.py`
Low-level gRPC client (usually not used directly):

```python
from dspy_agent.infra.grpc_client import OrchestratorClient

client = OrchestratorClient(host="localhost", port=50052)

# Submit task
response = await client.submit_task(
    task_id="task-123",
    task_type="inference",
    payload={"prompt": "What is AI?"},
    metadata={"user": "alice", "priority": "high"}
)

# Get status
status = await client.get_task_status("task-123")
```

---

## 🔄 Migration Guide for Python Code

### Old Pattern (docker-compose)
```python
import subprocess
import time

# Start services manually
subprocess.run(["docker-compose", "up", "-d", "redis"])
time.sleep(5)
subprocess.run(["docker-compose", "up", "-d", "reddb"])
time.sleep(10)

# Use services
from dspy_agent.skills import RedDB
reddb = RedDB(host="localhost", port=6380)
```

### New Pattern (unified infra)
```python
from dspy_agent.infra import AgentInfra
from dspy_agent.skills import RedDB

async def main():
    async with AgentInfra.start() as infra:
        # Services auto-started, no manual coordination needed
        reddb = RedDB(host="localhost", port=6380)
        
        # Use orchestrator for complex tasks
        await infra.submit_task("task-1", "inference", {...})

asyncio.run(main())
```

### Finding AgentInfra Binary
```python
# AgentInfra auto-detects the binary in common locations:
# - cmd/dspy-agent/dspy-agent
# - target/release/dspy-agent
# - System PATH

# Or specify explicitly:
async with AgentInfra.start(dspy_agent_cli_path=Path("/path/to/dspy-agent")) as infra:
    ...
```

---

## 🧪 Testing Integration

### Test Orchestrator Connection
```python
# tests/integration/test_grpc_infra.py
import pytest
from dspy_agent.infra import OrchestratorClient

@pytest.mark.asyncio
async def test_submit_task():
    client = OrchestratorClient(host="localhost", port=50052)
    
    response = await client.submit_task(
        task_id="test-1",
        task_type="test",
        payload={"data": "value"}
    )
    
    assert response.success
    
    # Wait for task
    await asyncio.sleep(6)
    
    status = await client.get_task_status("test-1")
    assert status.status == TaskStatus.COMPLETED
```

### Test Unified Startup
```python
# tests/integration/test_startup.py
@pytest.mark.asyncio
async def test_full_startup():
    async with AgentInfra.start() as infra:
        # Verify orchestrator is reachable
        assert infra.orchestrator_client is not None
        
        # Submit test task
        response = await infra.submit_task("startup-test", "ping", {})
        assert response.success
```

---

## 🚀 What Python Developers Need to Do

### 1. **Update Imports**
```python
# Add this import to agent code
from dspy_agent.infra import AgentInfra
```

### 2. **Wrap Agent Initialization**
```python
# OLD: Direct service access
def run_agent():
    agent = DSPyAgent()
    agent.train()

# NEW: Use AgentInfra context
async def run_agent():
    async with AgentInfra.start() as infra:
        agent = DSPyAgent(infra=infra)  # Pass infra for task submission
        await agent.train()
```

### 3. **Use Task Submission for Long-Running Operations**
```python
# For RL training, model training, etc.
async def train_model(infra, config):
    # Submit to orchestrator instead of blocking locally
    task_id = f"train-{uuid.uuid4()}"
    
    await infra.submit_task(
        task_id=task_id,
        task_type="rl_training",
        payload={
            "model_config": config,
            "episodes": 1000,
            "checkpoint_dir": "/data/checkpoints"
        }
    )
    
    # Poll for completion
    while True:
        status = await infra.get_task_status(task_id)
        if status.status == TaskStatus.COMPLETED:
            return status.result_payload
        elif status.status == TaskStatus.FAILED:
            raise RuntimeError(status.error)
        await asyncio.sleep(1)
```

### 4. **Environment Variable Configuration**
```python
# Set before running agent (optional, has defaults)
import os

os.environ["ORCHESTRATOR_GRPC_ADDR"] = "0.0.0.0:50052"
os.environ["ENV_MANAGER_GRPC_ADDR"] = "0.0.0.0:50051"

# Then use AgentInfra as normal
```

---

## 📊 Service Discovery

All services are auto-discovered. No manual IP/port tracking needed:

```python
# Services available after AgentInfra.start():
# - RedDB:      localhost:6380
# - Redis:      localhost:6379
# - InferMesh:  localhost:8000
# - Ollama:     localhost:11434
# - ModelServer: localhost:8001

# Access as before:
from dspy_agent.skills import RedDB, InferMesh

async with AgentInfra.start() as infra:
    reddb = RedDB()  # Auto-connects to localhost:6380
    infermesh = InferMesh()  # Auto-connects to localhost:8000
```

---

## 🔍 Debugging

### Check Service Status
```bash
# From terminal
dspy-agent status

# From Python
status = await infra.orchestrator_client.get_task_status("...")
```

### View Logs
```bash
# Terminal
dspy-agent logs --service reddb
dspy-agent logs --service infermesh

# Python (via gRPC streaming)
# TODO: Implement log streaming client
```

### Verify gRPC Connection
```python
import grpc
from dspy_agent.infra.pb.orchestrator import orchestrator_pb2_grpc

channel = grpc.aio.insecure_channel("localhost:50052")
stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)

try:
    # Try a simple call
    response = await stub.GetTaskStatus(
        orchestrator_pb2.GetTaskStatusRequest(task_id="test")
    )
    print("Orchestrator is reachable!")
except grpc.aio.AioRpcError as e:
    print(f"Connection failed: {e}")
```

---

## ⚠️ Breaking Changes

### 1. **No More docker-compose Commands**
```bash
# ❌ OLD - Don't use these anymore
docker-compose up -d
docker-compose down

# ✅ NEW
dspy-agent start
dspy-agent stop
```

### 2. **Async Context Manager Required**
```python
# ❌ OLD - Direct instantiation
agent = DSPyAgent()

# ✅ NEW - Use context manager
async with AgentInfra.start() as infra:
    agent = DSPyAgent()
```

### 3. **Service Startup is Automatic**
```python
# ❌ OLD - Manual service checks
if not redis_running():
    start_redis()

# ✅ NEW - AgentInfra handles it
async with AgentInfra.start() as infra:
    # Redis is already running
    pass
```

---

## 📝 Code Generation (Protobuf Stubs)

**When proto files change, regenerate stubs:**

```bash
# From project root
buf generate

# This updates:
# - dspy_agent/infra/pb/orchestrator/*.py (Python stubs)
# - orchestrator/internal/pb/orchestrator/*.go (Go stubs)
# - env_manager_rs/src/pb/*.rs (Rust stubs - generated during build)
```

**Python developers typically don't need to run this** unless modifying proto files.

---

## 🎓 Training Guide

### Quick Start Example
```python
#!/usr/bin/env python3
import asyncio
from dspy_agent.infra import AgentInfra
from dspy_agent.agents import DSPyAgent

async def main():
    # Start all infrastructure
    async with AgentInfra.start() as infra:
        print("✅ All services running!")
        
        # Initialize agent
        agent = DSPyAgent()
        
        # Run agent loop
        await agent.run()
        
        # Submit background tasks
        await infra.submit_task(
            "background-train",
            "rl_training",
            {"episodes": 100}
        )
        
        print("✅ Agent running, Ctrl+C to stop")
        
        # Keep running until interrupted
        await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Shutting down gracefully...")
```

Save as `agent_runner.py` and run:
```bash
python agent_runner.py
```

---

## 🆘 Common Issues

### Issue 1: "dspy-agent binary not found"
```python
# Solution: Build the CLI first
# In WSL terminal:
cd cmd/dspy-agent
go build -o dspy-agent .

# Or specify path explicitly:
async with AgentInfra.start(dspy_agent_cli_path=Path("./cmd/dspy-agent/dspy-agent")) as infra:
    ...
```

### Issue 2: "Connection refused to localhost:50052"
```bash
# Check if orchestrator is running
ps aux | grep dspy-agent

# Check if port is in use
lsof -i :50052

# Restart infrastructure
dspy-agent stop
dspy-agent start
```

### Issue 3: "Docker daemon not accessible"
```bash
# Ensure Docker is running
docker ps

# Check env-manager logs
dspy-agent logs --service env-manager
```

---

## 📞 Communication Protocols

### Python → Go (port 50052)
```
Python Agent
    ↓ gRPC (orchestrator.v1.proto)
Go Orchestrator
```

### Go → Rust (port 50051)
```
Go Orchestrator
    ↓ gRPC (env_manager.v1.proto)
Rust Env Manager
    ↓ Docker API
Docker Daemon
```

### Complete Flow
```
Python Agent
    ↓ submit_task()
Go Orchestrator
    ↓ start_environment()
Rust Env Manager
    ↓ create_container()
Docker Daemon
    → Container Running
```

---

## ✅ Verification Checklist

Before merging to main branch:

- [ ] `dspy-agent start` successfully starts all services
- [ ] `dspy-agent status` shows all services as "running"
- [ ] Python can connect to orchestrator on port 50052
- [ ] Existing DSPy agent code works with `AgentInfra` wrapper
- [ ] Tests pass: `pytest tests/integration/`
- [ ] Services gracefully shut down with `dspy-agent stop`
- [ ] Documentation is clear and examples work

---

## 🔮 Future Enhancements

**Planned (not yet implemented):**
1. **Log Streaming:** gRPC streaming for real-time logs in Python
2. **Metrics Collection:** Prometheus/Grafana integration
3. **Auto-scaling:** Dynamic container scaling based on load
4. **Health Dashboard:** Web UI for service monitoring
5. **Config Hot-reload:** Change settings without restart
6. **Multi-node:** Distributed orchestration across machines

---

## 📚 Additional Resources

- **Quick Start:** `docs/QUICKSTART.md`
- **Migration Guide:** `docs/MIGRATION.md`
- **Architecture Deep Dive:** `docs/INFRASTRUCTURE.md`
- **Build Instructions:** `BUILD_INSTRUCTIONS.md`
- **Proto Definitions:** `proto/*.proto`

---

## 🤝 Questions?

**For Python-specific questions:**
- Check `dspy_agent/infra/` module code
- Review `tests/integration/test_*.py` examples
- See `docs/QUICKSTART.md` section "Python Integration"

**For infrastructure issues:**
- Check `dspy-agent logs`
- Review `BUILD_INSTRUCTIONS.md`
- Inspect proto definitions in `proto/`

**For orchestrator behavior:**
- See `orchestrator/internal/grpc/server.go`
- Review `proto/orchestrator.v1.proto`

---

**Last Updated:** October 12, 2025  
**Version:** 1.0.0  
**Status:** ✅ Implementation Complete, Ready for Integration

