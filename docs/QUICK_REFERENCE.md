# Quick Reference Card - New Infrastructure

**For Python Developers** - Keep this handy!

---

## Basic Usage

```python
from dspy_agent.infra import AgentInfra

# Start infrastructure
async with AgentInfra.start() as infra:
    # Submit a task
    result = await infra.submit_task(
        task_id="my-task-1",
        payload={"data": "value"},
        task_class="cpu_short",  # See task classes below
        priority=5
    )
    
    # Check status
    status = await infra.get_task_status("my-task-1")
    
    # Get metrics
    metrics = await infra.get_metrics()
    
    # Health check
    health = await infra.health_check()
```

---

## Task Classes

| Class | Duration | Hardware | Use Case |
|-------|----------|----------|----------|
| `cpu_short` | <1 min | CPU | Quick preprocessing |
| `cpu_long` | >1 min | CPU | Spark jobs, vectorization |
| `gpu_short` | <5 min | GPU | Small model training |
| `gpu_long` | >5 min | GPU | RL sweeps, GEPA optimization |
| `gpu_slurm` | Any | GPU (HPC) | Cluster jobs |

---

## Environment Variables

```bash
# Python (AgentInfra)
ORCHESTRATOR_GRPC_ADDR=127.0.0.1:50052
DSPY_AGENT_CLI_PATH=/path/to/dspy-agent
DSPY_AGENT_SKIP_START=0  # Set to 1 to skip auto-start

# Rust env-manager
ENV_MANAGER_GRPC_ADDR=0.0.0.0:50100
ENV_MANAGER_VERBOSE=true
ENV_MANAGER_HEALTH_TIMEOUT=60

# Go orchestrator  
ORCHESTRATOR_GRPC_ADDR=:50052
ENV_MANAGER_ADDR=localhost:50100
```

---

## Common Patterns

### Pattern 1: RL Training

```python
async def train_with_puffer():
    async with AgentInfra.start() as infra:
        result = await infra.submit_task(
            task_id=f"puffer-{uuid.uuid4()}",
            payload={
                "algorithm": "ppo",
                "hyperparams": {"protein": 0.2, "carbs": 0.6}
            },
            task_class="gpu_long",
            priority=10
        )
```

### Pattern 2: Streaming with Kafka

```python
async def init_streaming():
    async with AgentInfra.start() as infra:
        # Ensures Kafka is healthy
        health = await infra.health_check()
        if health["healthy"]:
            producer = KafkaProducer(bootstrap_servers='localhost:9092')
```

### Pattern 3: Spark Vectorization

```python
async def vectorize_embeddings():
    async with AgentInfra.start() as infra:
        result = await infra.submit_task(
            task_id=f"spark-vec-{uuid.uuid4()}",
            payload={"data_path": "/data", "batch_size": 1000},
            task_class="cpu_long"
        )
```

### Pattern 4: GEPA Optimization

```python
async def optimize_program(program):
    async with AgentInfra.start() as infra:
        result = await infra.submit_task(
            task_id=f"gepa-{program.name}",
            payload={
                "program": program.serialize(),
                "n_iterations": 50
            },
            task_class="gpu_short",
            priority=8
        )
```

---

## Port Mappings (Unchanged)

| Service | Port | URL |
|---------|------|-----|
| RedDB | 8082 | `http://localhost:8082` |
| Redis | 6379 | `redis://localhost:6379` |
| InferMesh Router | 19000 | `http://localhost:19000` |
| Ollama | 11435 | `http://localhost:11435` |
| Kafka | 9092 | `localhost:9092` |
| Zookeeper | 2181 | `localhost:2181` |
| Prometheus | 9090 | `http://localhost:9090` |
| Go Orchestrator (HTTP) | 9097 | `http://localhost:9097/metrics` |
| Go Orchestrator (gRPC) | 50052 | `grpc://127.0.0.1:50052` |
| Rust env-manager (gRPC) | 50100 | `grpc://0.0.0.0:50100` |

---

## Troubleshooting Commands

```bash
# Check Docker containers
docker ps -a

# Check orchestrator
curl http://localhost:9097/metrics
curl http://localhost:9097/queue/status

# Check env-manager
netstat -tuln | grep 50100

# View logs
docker logs reddb
docker logs redis

# Test Python connection
python -c "
import asyncio
from dspy_agent.infra import AgentInfra

async def test():
    async with AgentInfra.start() as infra:
        print(await infra.health_check())

asyncio.run(test())
"
```

---

## Migration Checklist

- [ ] Read `docs/PYTHON_INTEGRATION_GUIDE.md`
- [ ] Test basic `AgentInfra.start()` usage
- [ ] Verify Docker containers still work
- [ ] Update RL training code
- [ ] Update streaming initialization
- [ ] Convert Spark jobs to tasks
- [ ] Update GEPA optimization
- [ ] Add error handling
- [ ] Test with real workloads

---

## Key Files

| File | Purpose |
|------|---------|
| `docs/PYTHON_INTEGRATION_GUIDE.md` | **Main reference** - read this |
| `docs/RUST_GO_CHANGES_LOG.md` | Changelog and what changed |
| `docs/INFRASTRUCTURE_STATUS.md` | Current status and architecture |
| `BUILD_INSTRUCTIONS.md` | How to build binaries |
| `dspy_agent/infra/agent_infra.py` | Python infrastructure client |
| `dspy_agent/infra/grpc_client.py` | gRPC orchestrator client |

---

## Support

**Infrastructure (Rust/Go)**: Other developer  
**Python Integration**: You  
**Documentation**: All in `docs/`

---

**Quick Start**: `cat docs/PYTHON_INTEGRATION_GUIDE.md`

