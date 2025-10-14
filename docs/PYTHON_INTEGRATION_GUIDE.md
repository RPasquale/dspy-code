# Python Integration Guide: New Infrastructure

**For Python Developers** - How to use the new Rust/Go infrastructure

---

## Quick Start

### 1. Basic Usage

```python
from dspy_agent.infra import AgentInfra

async def main():
    # Start infrastructure (Rust env-manager + Go orchestrator)
    async with AgentInfra.start() as infra:
        # Submit a task
        result = await infra.submit_task(
            task_id="train-model-1",
            payload={"model": "gpt2", "epochs": 10},
            task_class="gpu_short",
            priority=5
        )
        print(f"Task submitted: {result}")
        
        # Check status
        status = await infra.get_task_status("train-model-1")
        print(f"Status: {status}")
        
        # Get system metrics
        metrics = await infra.get_metrics()
        print(f"Queue depth: {metrics.get('env_queue_depth', 0)}")
```

### 2. Environment Variables

```bash
# Optional - customize if needed
export ORCHESTRATOR_GRPC_ADDR=127.0.0.1:50052    # Go orchestrator gRPC
export ENV_MANAGER_GRPC_ADDR=127.0.0.1:50100     # Rust env-manager gRPC
export DSPY_AGENT_CLI_PATH=/path/to/dspy-agent   # Custom CLI location
export DSPY_AGENT_SKIP_START=0                   # Set to 1 to skip auto-start
```

---

## Architecture Overview

### What Changed

**OLD (Manual):**
```python
# Had to manually start services
import subprocess
subprocess.Popen(["docker-compose", "up", "-d"])
# Wait...
subprocess.Popen(["go", "run", "orchestrator/main.go"])
# Wait...
subprocess.Popen(["cargo", "run", "--bin", "env-manager"])
# Then finally use your agent
```

**NEW (Automatic):**
```python
# Everything starts automatically
async with AgentInfra.start() as infra:
    # Ready to go!
    pass
```

### Services Managed

The new infrastructure manages:
- ✅ **Redis** (port 6379) - Cache
- ✅ **RedDB** (port 8082) - Database
- ✅ **InferMesh** (ports 19000-19002) - Inference mesh
- ✅ **Ollama** (port 11435) - Local LLM
- ✅ **Kafka** (port 9092) - Event streaming
- ✅ **Zookeeper** (port 2181) - Kafka coordination
- ✅ **Prometheus** (port 9090) - Metrics

---

## Migration Examples

### 1. RL Training (PufferLib Sweeps)

**BEFORE:**
```python
# dspy_agent/rl/puffer_sweep.py (OLD)
import subprocess

def run_sweep():
    # Manual process spawning
    proc = subprocess.Popen([
        "cargo", "run", "--bin", "env-manager", 
        "--", "run", "sweep"
    ])
    proc.wait()
```

**AFTER:**
```python
# dspy_agent/rl/puffer_sweep.py (NEW)
from dspy_agent.infra import AgentInfra

async def run_sweep():
    async with AgentInfra.start() as infra:
        # Submit sweep task to orchestrator
        result = await infra.submit_task(
            task_id=f"puffer-sweep-{uuid.uuid4()}",
            payload={
                "type": "rl_sweep",
                "algorithm": "ppo",
                "hyperparams": {
                    "protein": [0.1, 0.2, 0.3],
                    "carbs": [0.5, 0.6, 0.7]
                }
            },
            task_class="gpu_long",  # Long GPU job
            priority=10
        )
        
        # Poll for completion
        while True:
            status = await infra.get_task_status(result["task_id"])
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(5)
        
        return status["result_payload"]
```

### 2. Streaming / Kafka Integration

**BEFORE:**
```python
# dspy_agent/streaming/streaming_kafka.py (OLD)
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092')
# Hope Kafka is running...
```

**AFTER:**
```python
# dspy_agent/streaming/streaming_kafka.py (NEW)
from dspy_agent.infra import AgentInfra
from kafka import KafkaProducer

async def init_streaming():
    # Infrastructure ensures Kafka is ready
    async with AgentInfra.start() as infra:
        # Health check ensures Kafka is up
        health = await infra.health_check()
        if not health.get("healthy"):
            raise RuntimeError("Infrastructure not ready")
        
        # Now safe to connect
        producer = KafkaProducer(bootstrap_servers='localhost:9092')
        return producer
```

### 3. Spark Vectorization Jobs

**BEFORE:**
```python
# dspy_agent/embedding/spark_vectorizer.py (OLD)
import os

def run_spark_job():
    # Hope Spark is running on port 7077...
    os.system("spark-submit --master local[*] vectorize.py")
```

**AFTER:**
```python
# dspy_agent/embedding/spark_vectorizer.py (NEW)
from dspy_agent.infra import AgentInfra

async def run_spark_job():
    async with AgentInfra.start() as infra:
        # Submit as orchestrated task
        result = await infra.submit_task(
            task_id=f"spark-vectorize-{uuid.uuid4()}",
            payload={
                "type": "spark_vectorization",
                "data_path": "/data/embeddings",
                "batch_size": 1000
            },
            task_class="cpu_long",  # CPU-intensive
            priority=5
        )
        
        # Orchestrator handles Spark submission
        return result
```

### 4. GEPA Optimization

**BEFORE:**
```python
# dspy_agent/grpo/gepa.py (OLD)
def run_gepa_optimization(program):
    # Manual process coordination
    optimizer = OptimizerLM()
    # Submit to queue file
    with open("logs/env_queue/pending/task.json", "w") as f:
        json.dump({"program": program}, f)
```

**AFTER:**
```python
# dspy_agent/grpo/gepa.py (NEW)
from dspy_agent.infra import AgentInfra

async def run_gepa_optimization(program):
    async with AgentInfra.start() as infra:
        # Submit optimization task
        result = await infra.submit_task(
            task_id=f"gepa-opt-{program.name}",
            payload={
                "type": "gepa_optimization",
                "program": program.serialize(),
                "metric": "accuracy",
                "n_iterations": 50
            },
            task_class="gpu_short",
            priority=8
        )
        
        # Get optimized program
        status = await infra.get_task_status(result["task_id"])
        optimized = Program.deserialize(status["result_payload"]["program"])
        return optimized
```

---

## Advanced Usage

### 1. Custom Configuration

```python
from dspy_agent.infra import AgentInfra
from pathlib import Path

async def main():
    # Custom workspace and CLI path
    async with AgentInfra.start(
        orchestrator_addr="127.0.0.1:50052",
        workspace=Path("/custom/workspace"),
        auto_start_services=True,
        cli_path=Path("/usr/local/bin/dspy-agent")
    ) as infra:
        # Use infrastructure
        pass
```

### 2. Manual Control (Advanced)

```python
# For testing or special cases
import os
os.environ["DSPY_AGENT_SKIP_START"] = "1"  # Don't auto-start

from dspy_agent.infra import AgentInfra

async def main():
    # Assumes services already running
    infra = AgentInfra(orchestrator_addr="127.0.0.1:50052")
    await infra.start()
    
    # Use infra...
    
    await infra.stop()
```

### 3. Batch Task Submission

```python
async def submit_batch_tasks():
    async with AgentInfra.start() as infra:
        tasks = []
        for i in range(100):
            result = await infra.submit_task(
                task_id=f"batch-task-{i}",
                payload={"index": i, "data": f"item-{i}"},
                task_class="cpu_short",
                priority=5
            )
            tasks.append(result["task_id"])
        
        # Poll all tasks
        results = []
        for task_id in tasks:
            status = await infra.get_task_status(task_id)
            results.append(status)
        
        return results
```

---

## Task Classes

The orchestrator supports different task classes:

| Class | Description | Use Case |
|-------|-------------|----------|
| `cpu_short` | Fast CPU jobs (<1min) | Quick preprocessing |
| `cpu_long` | Long CPU jobs | Spark vectorization |
| `gpu_short` | Fast GPU jobs (<5min) | Small model training |
| `gpu_long` | Long GPU jobs | RL sweeps, GEPA |
| `gpu_slurm` | Slurm-managed GPU | HPC cluster jobs |

---

## Troubleshooting

### 1. "Failed to connect to orchestrator"

```python
# Check if services are running
import subprocess
result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
print(result.stdout)

# Manually start if needed
subprocess.run(["dspy-agent", "start"])
```

### 2. "Task stuck in pending"

```python
# Check orchestrator metrics
async with AgentInfra.start() as infra:
    metrics = await infra.get_metrics()
    print(f"Queue depth: {metrics.get('env_queue_depth')}")
    print(f"GPU wait: {metrics.get('gpu_wait_seconds')}")
```

### 3. "gRPC connection refused"

```bash
# Verify orchestrator is running
curl http://localhost:9097/metrics

# Check logs
dspy-agent logs --service orchestrator
```

---

## Key Takeaways

1. **Use `AgentInfra.start()` context manager** - This handles everything
2. **Submit tasks instead of manual processes** - Let orchestrator manage execution
3. **Use gRPC (port 50052) not HTTP (port 9097)** - Better performance
4. **Check health before critical operations** - `await infra.health_check()`
5. **Don't spawn subprocesses anymore** - Use `submit_task()` instead

---

## Next Steps

1. **Update your code** to use `AgentInfra.start()`
2. **Test with existing Docker containers** - No need to rebuild
3. **Monitor metrics** via `get_metrics()`
4. **Report issues** if something doesn't work

---

## Support

- **Infrastructure issues (Rust/Go)**: Other developer
- **Python integration issues**: You handle this
- **Documentation**: This file

The Rust/Go infrastructure is now production-ready. Focus on migrating your Python code to use the new APIs!

