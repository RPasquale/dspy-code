#!/usr/bin/env bash
set -euo pipefail

# Complete DSPy Agent System Setup Script
# This script sets up the entire system with all Go/Rust/Slurm components

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT_DIR/logs"
QUEUE_DIR="$LOG_DIR/env_queue"
PIDS_DIR="$LOG_DIR/pids"

echo "🚀 DSPy Agent Complete System Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS or Linux
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
log_info "Detected OS: $OS"

# Install system dependencies
install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    if [[ "$OS" == "macos" ]]; then
        # Check if Homebrew is installed
        if ! command -v brew >/dev/null 2>&1; then
            log_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install dependencies via Homebrew
        log_info "Installing dependencies via Homebrew..."
        brew install go rust python3 curl jq redis
        
    elif [[ "$OS" == "linux" ]]; then
        # Detect package manager
        if command -v apt-get >/dev/null 2>&1; then
            log_info "Installing dependencies via apt..."
            sudo apt-get update
            sudo apt-get install -y golang-go rustc cargo python3 python3-pip curl jq redis-server build-essential
        elif command -v yum >/dev/null 2>&1; then
            log_info "Installing dependencies via yum..."
            sudo yum install -y golang rust cargo python3 python3-pip curl jq redis build-essential
        elif command -v dnf >/dev/null 2>&1; then
            log_info "Installing dependencies via dnf..."
            sudo dnf install -y golang rust cargo python3 python3-pip curl jq redis build-essential
        else
            log_warning "Unknown package manager. Please install Go, Rust, Python3, curl, jq, and Redis manually."
        fi
    else
        log_warning "Unknown OS. Please install Go, Rust, Python3, curl, jq, and Redis manually."
    fi
}

# Install Python dependencies
install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$ROOT_DIR/venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv "$ROOT_DIR/venv"
    fi
    
    # Activate virtual environment
    source "$ROOT_DIR/venv/bin/activate"
    
    # Install Python packages
    log_info "Installing Python packages..."
    pip install --upgrade pip
    pip install -r requirements.txt 2>/dev/null || {
        log_warning "requirements.txt not found, installing basic packages..."
        pip install fastapi uvicorn requests pydantic
    }
    
    log_success "Python dependencies installed"
}

# Install Go dependencies
install_go_dependencies() {
    if command -v go >/dev/null 2>&1; then
        log_info "Installing Go dependencies..."
        
        # Set up Go environment
        export GOPATH="$ROOT_DIR/go"
        export GOCACHE="$ROOT_DIR/.gocache"
        export GOMODCACHE="$ROOT_DIR/.gomodcache"
        
        # Create directories
        mkdir -p "$GOPATH" "$GOCACHE" "$GOMODCACHE"
        
        # Install Go dependencies
        cd "$ROOT_DIR/orchestrator"
        go mod tidy
        go mod download
        
        log_success "Go dependencies installed"
    else
        log_warning "Go not found, skipping Go dependencies"
    fi
}

# Install Rust dependencies
install_rust_dependencies() {
    if command -v cargo >/dev/null 2>&1; then
        log_info "Installing Rust dependencies..."
        
        # Install Rust dependencies
        cd "$ROOT_DIR/env_runner_rs"
        cargo build --release
        
        log_success "Rust dependencies installed"
    else
        log_warning "Cargo not found, skipping Rust dependencies"
    fi
}

# Set up directories
setup_directories() {
    log_info "Setting up directories..."
    
    # Create necessary directories
    mkdir -p "$LOG_DIR" "$QUEUE_DIR/pending" "$QUEUE_DIR/done" "$PIDS_DIR"
    mkdir -p "$ROOT_DIR/.gocache" "$ROOT_DIR/.gomodcache"
    mkdir -p "$ROOT_DIR/deploy/slurm"
    mkdir -p "$ROOT_DIR/deploy/helm/orchestrator/templates"
    
    log_success "Directories created"
}

# Create environment configuration
create_environment_config() {
    log_info "Creating environment configuration..."
    
    # Create .env file
    cat > "$ROOT_DIR/.env" << EOF
# DSPy Agent Environment Configuration
DSPY_MODE=development
ENV_QUEUE_DIR=logs/env_queue
ORCHESTRATOR_DEMO=0
METRICS_ENABLED=true
REDIS_URL=redis://localhost:6379
KAFKA_BROKERS=localhost:9092
REDDB_URL=http://localhost:8000
INFERMESH_URL=http://localhost:9000
EOF

    # Create .env.production file
    cat > "$ROOT_DIR/.env.production" << EOF
# DSPy Agent Production Environment Configuration
DSPY_MODE=production
ENV_QUEUE_DIR=/app/logs/env_queue
ORCHESTRATOR_DEMO=0
METRICS_ENABLED=true
REDIS_URL=redis://redis:6379
KAFKA_BROKERS=kafka:9092
REDDB_URL=http://reddb:8000
INFERMESH_URL=http://infermesh-router:9000
EOF

    log_success "Environment configuration created"
}

# Create Docker Compose configuration
create_docker_compose() {
    log_info "Creating Docker Compose configuration..."
    
    # Check if docker-compose.yml exists and is up to date
    if [ ! -f "$ROOT_DIR/docker/lightweight/docker-compose.yml" ]; then
        log_warning "Docker Compose file not found, creating basic configuration..."
        mkdir -p "$ROOT_DIR/docker/lightweight"
        
        cat > "$ROOT_DIR/docker/lightweight/docker-compose.yml" << 'EOF'
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "127.0.0.1:6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  go-orchestrator:
    build:
      context: ../..
      dockerfile: docker/lightweight/Dockerfile
    environment:
      - ORCHESTRATOR_PORT=9097
      - ENV_QUEUE_DIR=/app/logs/env_queue
      - ORCHESTRATOR_DEMO=0
      - METRICS_ENABLED=true
    ports:
      - "127.0.0.1:9097:9097"
    volumes:
      - ../../logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped

  rust-env-runner:
    build:
      context: ../..
      dockerfile: docker/lightweight/Dockerfile
    environment:
      - ENV_QUEUE_DIR=/app/logs/env_queue
      - METRICS_PORT=8080
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ../../logs:/app/logs
    depends_on:
      - redis
      - go-orchestrator
    restart: unless-stopped

volumes:
  redis-data:
EOF
    fi
    
    log_success "Docker Compose configuration ready"
}

# Create Helm charts
create_helm_charts() {
    log_info "Creating Helm charts..."
    
    # Create Helm chart structure
    mkdir -p "$ROOT_DIR/deploy/helm/orchestrator/templates"
    
    # Create Chart.yaml
    cat > "$ROOT_DIR/deploy/helm/orchestrator/Chart.yaml" << 'EOF'
apiVersion: v2
name: orchestrator
description: DSPy Agent Orchestrator
type: application
version: 0.1.0
appVersion: "1.0.0"
EOF

    # Create values.yaml
    cat > "$ROOT_DIR/deploy/helm/orchestrator/values.yaml" << 'EOF'
replicaCount: 1

image:
  repository: dspy-agent
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 9097

autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80
  queueDepthTarget: "10"

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

nodeSelector: {}

tolerations: []

affinity: {}
EOF

    log_success "Helm charts created"
}

# Create test scripts
create_test_scripts() {
    log_info "Creating test scripts..."
    
    # Create test script
    cat > "$ROOT_DIR/scripts/test_system.sh" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "🧪 Testing DSPy Agent System"
echo "============================"

# Test orchestrator
echo "Testing orchestrator..."
if curl -s http://localhost:9097/metrics >/dev/null; then
    echo "✅ Orchestrator is running"
else
    echo "❌ Orchestrator is not responding"
    exit 1
fi

# Test env-runner
echo "Testing env-runner..."
if curl -s http://localhost:8080/health >/dev/null; then
    echo "✅ Env-runner is running"
else
    echo "❌ Env-runner is not responding"
    exit 1
fi

# Test queue submission
echo "Testing queue submission..."
response=$(curl -s -X POST http://localhost:9097/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{"id":"test_001","class":"cpu_short","payload":{"test":"data"}}')

if echo "$response" | grep -q '"ok":true'; then
    echo "✅ Queue submission working"
else
    echo "❌ Queue submission failed"
    exit 1
fi

# Test Slurm job submission
echo "Testing Slurm job submission..."
response=$(curl -s -X POST http://localhost:9097/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{"id":"slurm_001","class":"gpu_slurm","payload":{"method":"grpo"}}')

if echo "$response" | grep -q '"ok":true'; then
    echo "✅ Slurm job submission working"
else
    echo "❌ Slurm job submission failed"
    exit 1
fi

echo "🎉 All tests passed!"
EOF

    chmod +x "$ROOT_DIR/scripts/test_system.sh"
    
    log_success "Test scripts created"
}

# Create documentation
create_documentation() {
    log_info "Creating documentation..."
    
    # Create README for the complete system
    cat > "$ROOT_DIR/COMPLETE_SYSTEM_README.md" << 'EOF'
# DSPy Agent Complete System

This is the complete DSPy Agent system with Go/Rust/Slurm integration.

## Quick Start

1. **Setup the system:**
   ```bash
   bash scripts/setup_complete_system.sh
   ```

2. **Start all services:**
   ```bash
   bash scripts/start_local_system.sh
   ```

3. **Test the system:**
   ```bash
   bash scripts/test_system.sh
   ```

## Services

- **Dashboard**: http://localhost:8080
- **Orchestrator API**: http://localhost:9097
- **Env-Runner API**: http://localhost:8080
- **InferMesh Gateway**: http://localhost:19000
- **Redis Cache**: http://localhost:6379
- **Metrics**: http://localhost:9097/metrics
- **Queue Status**: http://localhost:9097/queue/status

## API Usage

### Submit a regular task:
```bash
curl -X POST http://localhost:9097/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{"id":"test_001","class":"cpu_short","payload":{"test":"data"}}'
```

### Submit a Slurm job:
```bash
curl -X POST http://localhost:9097/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{"id":"slurm_001","class":"gpu_slurm","payload":{"method":"grpo"}}'
```

### Test InferMesh embedding:
```bash
curl -X POST http://localhost:19000/embed \
  -H 'Content-Type: application/json' \
  -d '{"model": "BAAI/bge-small-en-v1.5", "inputs": ["test embedding"]}'
```

### Check job status:
```bash
curl http://localhost:9097/slurm/status/slurm_001
```

## Docker Deployment

```bash
cd docker/lightweight
docker-compose up -d
```

## Kubernetes Deployment

```bash
helm install orchestrator ./deploy/helm/orchestrator
```

## Troubleshooting

- Check service status: `bash scripts/start_local_system.sh`
- View logs: `tail -f logs/*.log`
- Test system: `bash scripts/test_system.sh`
EOF

    log_success "Documentation created"
}

# Main setup function
main() {
    log_info "Starting complete system setup..."
    
    # Check if we're in the right directory
    if [ ! -f "$ROOT_DIR/README.md" ]; then
        log_error "Please run this script from the DSPy Agent root directory"
        exit 1
    fi
    
    # Install dependencies
    install_system_dependencies
    install_python_dependencies
    install_go_dependencies
    install_rust_dependencies
    
    # Set up system
    setup_directories
    create_environment_config
    create_docker_compose
    create_helm_charts
    create_test_scripts
    create_documentation
    
    log_success "Complete system setup finished!"
    echo ""
    echo "🎉 DSPy Agent system is ready!"
    echo ""
    echo "Next steps:"
    echo "1. Start the system: bash scripts/start_local_system.sh"
    echo "2. Test the system: bash scripts/test_system.sh"
    echo "3. View documentation: cat COMPLETE_SYSTEM_README.md"
    echo ""
    echo "Services will be available at:"
    echo "- Dashboard: http://localhost:8080"
    echo "- Orchestrator: http://localhost:9097"
    echo "- Env-Runner: http://localhost:8080"
    echo ""
}

# Run main function
main "$@"
