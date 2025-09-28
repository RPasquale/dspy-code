# DSPy Agent Implementation Guide

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows (with WSL2)
- **Memory**: Minimum 8GB RAM, Recommended 16GB+
- **Storage**: Minimum 20GB free space
- **CPU**: Multi-core processor recommended

### Software Dependencies
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Rust**: Version 1.82+
- **Go**: Version 1.22+
- **Python**: Version 3.11+
- **Node.js**: Version 18+ (for frontend development)

## Installation Steps

### 1. Clone Repository
```bash
git clone <repository-url>
cd dspy_stuff
```

### 2. Install System Dependencies

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Go
wget https://go.dev/dl/go1.22.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.0.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc

# Install Python 3.11
sudo apt install python3.11 python3.11-pip python3.11-venv
```

#### macOS
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install docker docker-compose rust go python@3.11 node
```

#### Windows (WSL2)
```bash
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu

# Follow Ubuntu installation steps above
```

### 3. Build System Components

#### Build All Components
```bash
# Build everything
make build

# Or build individually
make build-go      # Build Go orchestrator
make build-rust     # Build Rust components
make build-python   # Build Python components
```

#### Build Go Orchestrator
```bash
cd orchestrator
go mod download
go build -o orchestrator cmd/orchestrator/main.go
```

#### Build Rust Components
```bash
# Build env-runner
cd env_runner_rs
cargo build --release

# Build reddb
cd ../reddb_rs
cargo build --release
```

#### Build Python Components
```bash
# Install Python dependencies
pip install -e .

# Install additional dependencies
pip install fastapi uvicorn
```

### 4. Configure Environment

#### Create Environment File
```bash
# Copy example environment file
cp docker/lightweight/.env.example docker/lightweight/.env

# Edit environment variables
nano docker/lightweight/.env
```

#### Environment Variables
```bash
# Core configuration
REDDB_ADMIN_TOKEN=your_secure_admin_token_here
REDDB_DATA_DIR=/data
ENV_QUEUE_DIR=logs/env_queue
ORCHESTRATOR_DEMO=0

# Service ports
REDDB_PORT=8082
ORCHESTRATOR_PORT=9097
ENV_RUNNER_PORT=8080
INFERMESH_PORT=19000
REDIS_PORT=6379

# Database configuration
REDDB_HOST=127.0.0.1
REDDB_DATABASE_URL=sqlite:///data/reddb.sqlite

# Redis configuration
REDIS_HOST=127.0.0.1
REDIS_PASSWORD=

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### 5. Start Services

#### Start All Services
```bash
# Start complete stack
make up

# Or start specific services
make up-core       # Core services only
make up-full       # All services including monitoring
```

#### Start Individual Services
```bash
# Start Redis
docker compose -f docker/lightweight/docker-compose.yml up -d redis

# Start RedDB
docker compose -f docker/lightweight/docker-compose.yml up -d reddb

# Start Go Orchestrator
docker compose -f docker/lightweight/docker-compose.yml up -d go-orchestrator

# Start Rust Env Runner
docker compose -f docker/lightweight/docker-compose.yml up -d rust-env-runner

# Start InferMesh
docker compose -f docker/lightweight/docker-compose.yml up -d infermesh
```

### 6. Verify Installation

#### Check Service Health
```bash
# Check all services
make health

# Check individual services
curl http://localhost:8082/health    # RedDB
curl http://localhost:9097/metrics   # Go Orchestrator
curl http://localhost:8080/health     # Rust Env Runner
curl http://localhost:19000/health    # InferMesh
```

#### Check Service Logs
```bash
# View all logs
make logs

# View specific service logs
docker logs lightweight-reddb-1
docker logs lightweight-go-orchestrator-1
docker logs lightweight-rust-env-runner-1
```

#### Check Service Status
```bash
# Check running containers
docker ps

# Check service status
make status
```

## Development Setup

### 1. Local Development Environment

#### Start Development Mode
```bash
# Start development environment
make dev

# Or start with hot reloading
make dev-hot
```

#### Development Configuration
```bash
# Set development environment
export DSPY_ENV=development
export DEBUG=true
export LOG_LEVEL=debug
```

### 2. Code Development

#### Go Development
```bash
# Run Go tests
cd orchestrator
go test ./...

# Run Go linter
golangci-lint run

# Format Go code
gofmt -w .
```

#### Rust Development
```bash
# Run Rust tests
cd env_runner_rs
cargo test

# Run Rust linter
cargo clippy

# Format Rust code
cargo fmt
```

#### Python Development
```bash
# Run Python tests
pytest

# Run Python linter
flake8 dspy_agent/

# Format Python code
black dspy_agent/
```

### 3. Testing

#### Run All Tests
```bash
# Run complete test suite
make test

# Run specific test suites
make test-go       # Go tests
make test-rust     # Rust tests
make test-python   # Python tests
```

#### Integration Testing
```bash
# Run integration tests
make test-integration

# Run end-to-end tests
make test-e2e
```

#### Performance Testing
```bash
# Run performance tests
make test-performance

# Run load tests
make test-load
```

## Production Deployment

### 1. Production Configuration

#### Production Environment Variables
```bash
# Production configuration
export DSPY_ENV=production
export DEBUG=false
export LOG_LEVEL=info

# Security
export REDDB_ADMIN_TOKEN=$(openssl rand -hex 32)
export JWT_SECRET=$(openssl rand -hex 32)

# Performance
export WORKER_PROCESSES=4
export MAX_CONNECTIONS=1000
export CACHE_SIZE=1GB
```

#### Production Docker Compose
```yaml
# docker/lightweight/docker-compose.prod.yml
version: '3.8'

services:
  reddb:
    image: dspy-lightweight:latest
    restart: unless-stopped
    environment:
      - REDDB_ADMIN_TOKEN=${REDDB_ADMIN_TOKEN}
      - REDDB_DATA_DIR=/data
    volumes:
      - ./data/reddb:/data
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'
```

### 2. Kubernetes Deployment

#### Create Kubernetes Manifests
```bash
# Generate Kubernetes manifests
helm template dspy-agent ./k8s/helm/dspy-agent > k8s/manifests/dspy-agent.yaml

# Apply manifests
kubectl apply -f k8s/manifests/
```

#### Kubernetes Configuration
```yaml
# k8s/helm/dspy-agent/values.yaml
replicaCount: 3

image:
  repository: dspy-lightweight
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: dspy-agent.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: dspy-agent-tls
      hosts:
        - dspy-agent.example.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80
```

### 3. Monitoring and Observability

#### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'dspy-agent'
    static_configs:
      - targets: ['localhost:9097', 'localhost:8080', 'localhost:8082']
    metrics_path: /metrics
    scrape_interval: 5s
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "DSPy Agent Monitoring",
    "panels": [
      {
        "title": "Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"dspy-agent\"}"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Check port usage
lsof -i :8082
lsof -i :9097
lsof -i :8080

# Kill conflicting processes
sudo kill -9 <PID>
```

#### 2. Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER ./data/
sudo chmod -R 755 ./data/
```

#### 3. Docker Issues
```bash
# Clean Docker system
docker system prune -a

# Rebuild images
docker compose build --no-cache
```

#### 4. Service Health Issues
```bash
# Check service logs
docker logs <container_name>

# Restart services
docker compose restart <service_name>

# Check service health
curl http://localhost:<port>/health
```

### Debug Commands

#### System Debugging
```bash
# Check system resources
docker stats

# Check network connectivity
docker network ls
docker network inspect <network_name>

# Check volume mounts
docker volume ls
docker volume inspect <volume_name>
```

#### Service Debugging
```bash
# Check service configuration
docker inspect <container_name>

# Check service environment
docker exec <container_name> env

# Check service processes
docker exec <container_name> ps aux
```

## Performance Optimization

### 1. System Optimization

#### Memory Optimization
```bash
# Set memory limits
export WORKER_MEMORY_LIMIT=2G
export CACHE_MEMORY_LIMIT=1G
export REDIS_MEMORY_LIMIT=512M
```

#### CPU Optimization
```bash
# Set CPU limits
export WORKER_CPU_LIMIT=2
export CACHE_CPU_LIMIT=1
export REDIS_CPU_LIMIT=0.5
```

#### Network Optimization
```bash
# Set network limits
export MAX_CONNECTIONS=1000
export KEEP_ALIVE_TIMEOUT=30s
export REQUEST_TIMEOUT=30s
```

### 2. Application Optimization

#### Database Optimization
```sql
-- RedDB optimization
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;
PRAGMA temp_store=MEMORY;
```

#### Caching Optimization
```bash
# Redis optimization
redis-cli CONFIG SET maxmemory 1gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET save ""
```

#### Application Optimization
```python
# Python optimization
import os
os.environ['PYTHONOPTIMIZE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
```

## Security Considerations

### 1. Authentication and Authorization

#### Token Management
```bash
# Generate secure tokens
export REDDB_ADMIN_TOKEN=$(openssl rand -hex 32)
export JWT_SECRET=$(openssl rand -hex 32)
export API_KEY=$(openssl rand -hex 32)
```

#### Access Control
```yaml
# Access control configuration
security:
  authentication:
    enabled: true
    method: token
  authorization:
    enabled: true
    roles: [admin, user, guest]
  rate_limiting:
    enabled: true
    requests_per_minute: 100
```

### 2. Data Security

#### Encryption
```bash
# Enable encryption
export ENCRYPTION_ENABLED=true
export ENCRYPTION_KEY=$(openssl rand -hex 32)
```

#### Data Protection
```yaml
# Data protection configuration
data_protection:
  encryption:
    enabled: true
    algorithm: AES-256-GCM
  backup:
    enabled: true
    schedule: "0 2 * * *"
    retention: 30d
```

## Maintenance

### 1. Regular Maintenance

#### System Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update Docker images
docker compose pull
docker compose up -d
```

#### Database Maintenance
```bash
# Optimize database
sqlite3 data/reddb/reddb.sqlite "VACUUM;"

# Backup database
cp data/reddb/reddb.sqlite backups/reddb-$(date +%Y%m%d).sqlite
```

#### Log Rotation
```bash
# Configure log rotation
sudo nano /etc/logrotate.d/dspy-agent

# Log rotation configuration
/var/log/dspy-agent/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
}
```

### 2. Monitoring and Alerting

#### Health Checks
```bash
# Automated health checks
crontab -e

# Add health check cron job
*/5 * * * * /path/to/health-check.sh
```

#### Alerting Configuration
```yaml
# Alerting configuration
alerts:
  - name: "Service Down"
    condition: "up == 0"
    severity: "critical"
    notification: "email"
  - name: "High Memory Usage"
    condition: "memory_usage > 90%"
    severity: "warning"
    notification: "slack"
```

---

This implementation guide provides comprehensive instructions for setting up, developing, and maintaining the DSPy Agent system in various environments, from local development to production deployment.
