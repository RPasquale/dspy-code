#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT_DIR/logs"
QUEUE_DIR="$LOG_DIR/env_queue"
PIDS_DIR="$LOG_DIR/pids"
mkdir -p "$QUEUE_DIR/pending" "$QUEUE_DIR/done" "$PIDS_DIR"

echo "[start] base=$ROOT_DIR"

# Check for required tools
check_dependencies() {
  local missing=()
  
  if ! command -v python3 >/dev/null 2>&1; then
    missing+=("python3")
  fi
  
  if ! command -v go >/dev/null 2>&1; then
    echo "[warn] go not found; Go orchestrator will be skipped"
  fi
  
  if ! command -v cargo >/dev/null 2>&1; then
    echo "[warn] cargo not found; Rust env-runner will be skipped"
  fi
  
  if [ ${#missing[@]} -gt 0 ]; then
    echo "[error] Missing required dependencies: ${missing[*]}"
    echo "Please install the missing dependencies and try again."
    exit 1
  fi
}

# Start Redis if available
start_redis() {
  if command -v redis-server >/dev/null 2>&1; then
    if ! pgrep -f "redis-server" >/dev/null 2>&1; then
      echo "[start] redis"
      (nohup redis-server --port 6379 --daemonize yes >/dev/null 2>&1) || true
    else
      echo "[start] redis already running"
    fi
  else
    echo "[warn] redis-server not found; Redis will be skipped"
  fi
}

# Start Kafka if available
start_kafka() {
  if [ -d "$ROOT_DIR/kafka" ] && [ -f "$ROOT_DIR/kafka/bin/kafka-server-start.sh" ]; then
    if ! pgrep -f "kafka.Kafka" >/dev/null 2>&1; then
      echo "[start] kafka"
      (cd "$ROOT_DIR/kafka" && nohup bin/kafka-server-start.sh config/server.properties >/dev/null 2>&1 & echo $! > "$PIDS_DIR/kafka.pid") || true
    else
      echo "[start] kafka already running"
    fi
  else
    echo "[warn] kafka not found; Kafka will be skipped"
  fi
}

# Start Zookeeper if available
start_zookeeper() {
  if [ -d "$ROOT_DIR/kafka" ] && [ -f "$ROOT_DIR/kafka/bin/zookeeper-server-start.sh" ]; then
    if ! pgrep -f "zookeeper" >/dev/null 2>&1; then
      echo "[start] zookeeper"
      (cd "$ROOT_DIR/kafka" && nohup bin/zookeeper-server-start.sh config/zookeeper.properties >/dev/null 2>&1 & echo $! > "$PIDS_DIR/zookeeper.pid") || true
    else
      echo "[start] zookeeper already running"
    fi
  else
    echo "[warn] zookeeper not found; Zookeeper will be skipped"
  fi
}

# Start RedDB if available
start_reddb() {
  if [ -f "$ROOT_DIR/scripts/start_reddb.sh" ]; then
    if ! pgrep -f "reddb" >/dev/null 2>&1; then
      echo "[start] reddb"
      (cd "$ROOT_DIR" && nohup bash scripts/start_reddb.sh >/dev/null 2>&1 & echo $! > "$PIDS_DIR/reddb.pid") || true
    else
      echo "[start] reddb already running"
    fi
  else
    echo "[warn] reddb not found; RedDB will be skipped"
  fi
}

# Start InferMesh stack if available
start_infermesh() {
  if command -v docker >/dev/null 2>&1 && command -v docker-compose >/dev/null 2>&1; then
    if [ -f "$ROOT_DIR/docker/lightweight/docker-compose.yml" ]; then
      echo "[start] infermesh stack (redis + embedder + dual nodes + router)"
      cd "$ROOT_DIR/docker/lightweight"
      
      # Start Redis first
      if ! docker compose ps redis | grep -q "Up"; then
        echo "[start] starting redis cache..."
        docker compose up -d redis
        sleep 2
      fi
      
      # Start DSPy embedder
      if ! docker compose ps dspy-embedder | grep -q "Up"; then
        echo "[start] starting DSPy embedder..."
        docker compose up -d dspy-embedder
        sleep 4
      fi

      # Start InferMesh nodes
      if ! docker compose ps infermesh-node-a | grep -q "Up"; then
        echo "[start] starting infermesh nodes..."
        docker compose up -d infermesh-node-a infermesh-node-b
        sleep 5
      fi
      
      # Start official router
      if ! docker compose ps infermesh-router | grep -q "Up"; then
        echo "[start] starting infermesh router..."
        docker compose up -d infermesh-router
        sleep 4
      fi
      
      echo "[start] infermesh stack started"
    else
      echo "[warn] docker-compose.yml not found; InferMesh will be skipped"
    fi
  else
    echo "[warn] docker/docker-compose not found; InferMesh will be skipped"
  fi
}

# 1) Check dependencies
check_dependencies

# 2) Start infrastructure services
start_redis
start_zookeeper
start_kafka
start_reddb
start_infermesh

# 3) Start dashboard (enhanced)
if ! pgrep -f "enhanced_dashboard_server.py" >/dev/null 2>&1; then
  echo "[start] dashboard"
  (cd "$ROOT_DIR" && nohup python3 enhanced_dashboard_server.py >/dev/null 2>&1 & echo $! > "$PIDS_DIR/dashboard.pid") || true
else
  echo "[start] dashboard already running"
fi

# 4) Build and start Go orchestrator
if command -v go >/dev/null 2>&1; then
  echo "[start] orchestrator"
  (cd "$ROOT_DIR/orchestrator" && GOCACHE=$(pwd)/.gocache GOMODCACHE=$(pwd)/.gomodcache go build -o "$ROOT_DIR/logs/orchestrator" ./cmd/orchestrator)
  if ! pgrep -f "/logs/orchestrator" >/dev/null 2>&1; then
    (ENV_QUEUE_DIR="$QUEUE_DIR" ORCHESTRATOR_DEMO=0 METRICS_ENABLED=true nohup "$ROOT_DIR/logs/orchestrator" >/dev/null 2>&1 & echo $! > "$PIDS_DIR/orchestrator.pid")
  fi
else
  echo "[warn] go not found; skipping orchestrator build"
fi

# 5) Build and start Rust env-runner
if command -v cargo >/dev/null 2>&1; then
  echo "[start] env-runner"
  (cd "$ROOT_DIR/env_runner_rs" && cargo build --release >/dev/null 2>&1)
  if ! pgrep -f "env-runner" >/dev/null 2>&1; then
    (ENV_QUEUE_DIR="$QUEUE_DIR" METRICS_PORT=8080 nohup "$ROOT_DIR/env_runner_rs/target/release/env-runner" >/dev/null 2>&1 & echo $! > "$PIDS_DIR/env-runner.pid")
  fi
else
  echo "[warn] cargo not found; skipping env-runner build"
fi

# 6) Start Slurm services if available
start_slurm() {
  if command -v sbatch >/dev/null 2>&1; then
    echo "[start] slurm services available"
    # Create Slurm job templates if they don't exist
    if [ ! -f "$ROOT_DIR/deploy/slurm/train_agent_methodologies.sbatch" ]; then
      echo "[warn] Slurm templates not found; creating basic templates"
      mkdir -p "$ROOT_DIR/deploy/slurm"
      cat > "$ROOT_DIR/deploy/slurm/train_agent_methodologies.sbatch" << 'EOF'
#!/bin/bash
#SBATCH --job-name=agent_training
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=/workspace/logs/slurm_%j.out
#SBATCH --error=/workspace/logs/slurm_%j.err

# Basic Slurm job template
echo "Starting agent training job"
python scripts/train_agent.py ${TRAINING_METHOD:-grpo} --backend local
EOF
      cat > "$ROOT_DIR/deploy/slurm/train_puffer_rl.sbatch" << 'EOF'
#!/bin/bash
#SBATCH --job-name=puffer_rl_train
#SBATCH --nodes=${NODES:-1}
#SBATCH --gpus-per-node=${GPUS:-1}
#SBATCH --cpus-per-task=${CPUS_PER_TASK:-8}
#SBATCH --mem=${MEMORY_GB:-48}G
#SBATCH --time=${TIME_LIMIT:-02:00:00}
#SBATCH --output=/workspace/logs/puffer_rl_%j.out
#SBATCH --error=/workspace/logs/puffer_rl_%j.err

echo "mock puffer rl job"
EOF
    fi
  else
    echo "[warn] sbatch not found; Slurm will be skipped"
  fi
}

start_slurm

# 7) Wait for services to start
echo "[start] waiting for services to start..."
sleep 3

# 8) Check service health
check_health() {
  local services=()
  
  # Check orchestrator
  if pgrep -f "/logs/orchestrator" >/dev/null 2>&1; then
    if curl -s http://localhost:9097/metrics >/dev/null 2>&1; then
      services+=("✅ orchestrator (http://localhost:9097)")
    else
      services+=("⚠️  orchestrator (starting...)")
    fi
  else
    services+=("❌ orchestrator (not running)")
  fi
  
  # Check env-runner
  if pgrep -f "env-runner" >/dev/null 2>&1; then
    if curl -s http://localhost:8080/health >/dev/null 2>&1; then
      services+=("✅ env-runner (http://localhost:8080)")
    else
      services+=("⚠️  env-runner (starting...)")
    fi
  else
    services+=("❌ env-runner (not running)")
  fi
  
  # Check dashboard
  if pgrep -f "enhanced_dashboard_server.py" >/dev/null 2>&1; then
    if curl -s http://localhost:8080 >/dev/null 2>&1; then
      services+=("✅ dashboard (http://localhost:8080)")
    else
      services+=("⚠️  dashboard (starting...)")
    fi
  else
    services+=("❌ dashboard (not running)")
  fi
  
  # Check InferMesh stack
  if command -v docker >/dev/null 2>&1; then
    cd "$ROOT_DIR/docker/lightweight"
    if docker compose ps dspy-embedder | grep -q "Up"; then
      if curl -s http://localhost:18082/health >/dev/null 2>&1; then
        services+=("✅ dspy-embedder (http://localhost:18082)")
      else
        services+=("⚠️  dspy-embedder (starting...)")
      fi
    else
      services+=("❌ dspy-embedder (not running)")
    fi

    if docker compose ps infermesh-router | grep -q "Up"; then
      if curl -s http://localhost:19000/health >/dev/null 2>&1; then
        services+=("✅ infermesh-router (http://localhost:19000)")
      else
        services+=("⚠️  infermesh-router (starting...)")
      fi
    else
      services+=("❌ infermesh-router (not running)")
    fi

    if docker compose ps redis | grep -q "Up"; then
      services+=("✅ redis-cache (http://localhost:6379)")
    else
      services+=("❌ redis-cache (not running)")
    fi
    
    if docker compose ps infermesh-node-a | grep -q "Up" && docker compose ps infermesh-node-b | grep -q "Up"; then
      services+=("✅ infermesh-nodes (node-a, node-b)")
    else
      services+=("❌ infermesh-nodes (not running)")
    fi
  else
    services+=("❌ infermesh-stack (docker not available)")
  fi
  
  echo ""
  echo "=== Service Status ==="
  for service in "${services[@]}"; do
    echo "  $service"
  done
  echo ""
}

check_health

echo "[start] done!"
echo ""
echo "=== Quick Start Guide ==="
echo "1. Dashboard: http://localhost:8080"
echo "2. Orchestrator API: http://localhost:9097"
echo "3. Env-Runner API: http://localhost:8080"
echo "4. InferMesh Gateway: http://localhost:19000"
echo "5. Redis Cache: http://localhost:6379"
echo "6. Metrics: http://localhost:9097/metrics"
echo "7. Queue Status: http://localhost:9097/queue/status"
echo ""
echo "=== Test InferMesh Embedding ==="
echo "curl -X POST http://localhost:19000/embed \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"BAAI/bge-small-en-v1.5\", \"inputs\": [\"test embedding\"]}'"
echo ""
echo "=== Submit a Test Job ==="
echo "curl -X POST http://localhost:9097/queue/submit \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"id\":\"test_001\",\"class\":\"cpu_short\",\"payload\":{\"test\":\"data\"}}'"
echo ""
echo "=== Submit a Slurm Job ==="
echo "curl -X POST http://localhost:9097/queue/submit \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"id\":\"slurm_001\",\"class\":\"gpu_slurm\",\"payload\":{\"method\":\"grpo\"}}'"
echo ""
echo "=== InferMesh Management ==="
echo "View logs: docker compose -f docker/lightweight/docker-compose.yml logs -f"
echo "Stop stack: docker compose -f docker/lightweight/docker-compose.yml down"
echo "Restart: docker compose -f docker/lightweight/docker-compose.yml restart"
echo ""
