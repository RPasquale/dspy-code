# DSPy Agent Deployment Guide

This guide covers deployment strategies for DSPy Agent across different environments, from local development to production-scale deployments.

## Table of Contents

- [Quick Deployment Options](#quick-deployment-options)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Cloud Deployments](#cloud-deployments)
- [Monitoring and Observability](#monitoring-and-observability)
- [Troubleshooting](#troubleshooting)

## Quick Deployment Options

### Option 1: pip/pipx Installation (Recommended for local use)

```bash
# Install globally with pipx (isolated environment)
pipx install dspy-code

# Or install in current environment
pip install dspy-code

# Run from any project directory
dspy-agent --workspace /path/to/your/project
```

### Option 2: Docker Compose Stack (Recommended for production)

```bash
# Generate stack configuration
dspy-agent lightweight_init \
  --workspace $(pwd) \
  --logs ./logs \
  --out-dir docker/lightweight \
  --install-source pip \
  --db auto

# Deploy the stack
export DOCKER_BUILDKIT=1
docker compose -f docker/lightweight/docker-compose.yml build --pull
docker compose -f docker/lightweight/docker-compose.yml up -d
```

### Option 3: Source Development

```bash
# Clone and setup
git clone https://github.com/your-org/dspy-code.git
cd dspy-code
pip install uv
uv sync

# Run from source
uv run dspy-agent
```

## Local Development

### Prerequisites

- Python 3.11+ (3.13 maximum)
- [Ollama](https://ollama.com/download) (recommended) or OpenAI API access
- Git (for repository analysis)
- Docker and Docker Compose (for full stack)

### Basic Setup

1. **Install Dependencies**
   ```bash
   # Using pipx (recommended)
   pipx install dspy-code
   
   # Or using pip
   pip install dspy-code
   ```

2. **Configure LLM Backend**
   
   **Option A: Ollama (Local)**
   ```bash
   # Install Ollama from https://ollama.com/download
   ollama pull qwen3:1.7b
   
   # Set environment variables (optional, these are defaults)
   export USE_OLLAMA=true
   export OLLAMA_MODEL=qwen3:1.7b
   export OLLAMA_BASE_URL=http://localhost:11434
   ```

   *Only a single Ollama model is loaded per agent process. The first model that initializes becomes the resident model. To allow a hot swap at runtime, start the process with `OLLAMA_ALLOW_SWITCH=1`; otherwise later configuration attempts reuse the existing model.*

   *Max-token caps are disabled for Ollama calls; responses stream until the model chooses to stop. If you need a ceiling for a specific workflow, apply it at the caller layer (e.g., post-process the stream or use a wrapper that truncates output).*
   
   **Option B: OpenAI API**
   ```bash
   export USE_OLLAMA=false
   export OPENAI_API_KEY=your_api_key_here
   export OPENAI_BASE_URL=https://api.openai.com/v1
   export MODEL_NAME=gpt-4o-mini
   ```

3. **Start Agent**
   ```bash
   # Navigate to your project
   cd /path/to/your/project
   
   # Start interactive session
   dspy-agent
   ```

### Development with Source

For contributing or customizing DSPy Agent:

```bash
# Clone repository
git clone https://github.com/your-org/dspy-code.git
cd dspy-code

# Install uv (fast Python package manager)
pip install uv

# Sync dependencies
uv sync

# Install in development mode
uv pip install -e .

# Run tests
uv run python -m unittest discover -s tests -v

# Run from source
uv run dspy-agent --workspace /path/to/project
```

## Docker Deployment

### Single Container Deployment

For simple deployments without streaming capabilities:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dspy-code
RUN pip install dspy-code

# Create workspace
VOLUME ["/workspace"]

# Expose port for HTTP status endpoint
EXPOSE 8000

# Set default command
CMD ["dspy-agent", "--workspace", "/workspace"]
```

```bash
# Build and run
docker build -t dspy-agent .
docker run -v $(pwd):/workspace -p 8000:8000 dspy-agent
```

### Full Stack Deployment (Recommended)

The full stack includes Kafka, Spark, Ollama, and agent services for production use.

#### Generate Stack Configuration

```bash
# Create configuration
dspy-agent lightweight_init \
  --workspace $(pwd) \
  --logs ./logs \
  --out-dir docker/lightweight \
  --install-source pip \
  --pip-spec "dspy-code==0.1.0" \
  --db redis
```

This generates:
- `docker/lightweight/docker-compose.yml` - Main orchestration
- `docker/lightweight/Dockerfile` - Agent container definition
- `docker/lightweight/entrypoints/` - Service startup scripts
- `docker/lightweight/scripts/` - Spark processing jobs

#### Deploy the Stack

```bash
# Set Docker build options
export DOCKER_BUILDKIT=1

# Build all services
docker compose -f docker/lightweight/docker-compose.yml build --pull

# Start services
docker compose -f docker/lightweight/docker-compose.yml up -d

# Check service status
docker compose -f docker/lightweight/docker-compose.yml ps

# View logs
docker compose -f docker/lightweight/docker-compose.yml logs -f dspy-agent
```

#### Stack Services

The full stack includes:

- **Zookeeper**: Kafka coordination
- **Kafka**: Event streaming and message queue
- **Spark**: Log processing and context extraction
- **Redis**: State storage and caching
- **Ollama**: Local LLM inference
- **DSPy Agent**: Main agent service
- **DSPy Worker**: Background processing workers
- **DSPy Router**: Request routing and load balancing

#### Environment Variables

Configure the stack using environment variables:

```bash
# Core configuration
export WORKSPACE=/path/to/project
export LOGS_DIR=/path/to/logs
export KAFKA_BOOTSTRAP=localhost:9092

# LLM configuration
export USE_OLLAMA=true
export OLLAMA_MODEL=qwen3:1.7b

# RL configuration
export RL_BACKGROUND_STEPS=50
export RL_WEIGHTS='{"pass_rate": 1.0, "lint_score": 0.5}'

# Database configuration
export REDIS_URL=redis://localhost:6379

# Monitoring
export ENABLE_METRICS=true
export METRICS_PORT=9090
```

#### Scaling Services

Scale individual services based on load:

```bash
# Scale worker services
docker compose -f docker/lightweight/docker-compose.yml up -d --scale dspy-worker=3

# Scale specific worker types
docker compose -f docker/lightweight/docker-compose.yml up -d --scale dspy-worker-backend=2
docker compose -f docker/lightweight/docker-compose.yml up -d --scale dspy-worker-frontend=2

# Check scaled services
docker compose -f docker/lightweight/docker-compose.yml ps
```

#### Health Checks and Monitoring

The stack includes built-in health checks:

```bash
# Check service health
curl http://localhost:8000/health

# Check Kafka topics
docker exec -it $(docker compose ps -q kafka) kafka-topics --bootstrap-server localhost:9092 --list

# Check Redis
docker exec -it $(docker compose ps -q redis) redis-cli ping

# Check Ollama
curl http://localhost:11434/api/tags
```

## Production Deployment

### Infrastructure Requirements

#### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB (16GB recommended with Ollama)
- **Storage**: 50GB SSD
- **Network**: 1Gbps

#### Recommended Production Setup
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 200GB+ NVMe SSD
- **Network**: 10Gbps
- **GPU**: Optional, for larger LLM models

### Production Configuration

#### 1. Environment Setup

```bash
# Create production environment file
cat > .env.production << EOF
# Core settings
WORKSPACE=/opt/dspy/workspace
LOGS_DIR=/opt/dspy/logs
DB_URL=redis://redis-cluster:6379

# Kafka cluster
KAFKA_BOOTSTRAP=kafka-1:9092,kafka-2:9092,kafka-3:9092

# LLM configuration
USE_OLLAMA=true
OLLAMA_MODEL=qwen3:7b
OLLAMA_BASE_URL=http://ollama-cluster:11434

# Performance tuning
RL_BACKGROUND_STEPS=100
MAX_CONCURRENT_REQUESTS=50
REQUEST_TIMEOUT=60

# Security
ENABLE_AUTH=true
JWT_SECRET_KEY=your-secret-key-here
ALLOWED_ORIGINS=https://your-domain.com

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
EOF
```

#### 2. Production Docker Compose

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  # Load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - dspy-agent
    restart: always

  # Main agent service (multiple instances)
  dspy-agent:
    image: dspy-code:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    environment:
      - WORKSPACE=/opt/workspace
      - KAFKA_BOOTSTRAP=kafka:9092
      - REDIS_URL=redis://redis:6379
    volumes:
      - workspace_data:/opt/workspace
      - logs_data:/opt/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: always

  # Redis cluster
  redis:
    image: redis:alpine
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: always

  # Kafka cluster (simplified single node for example)
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'
      KAFKA_LOG_RETENTION_HOURS: 168
      KAFKA_LOG_SEGMENT_BYTES: 1073741824
    volumes:
      - kafka_data:/var/lib/kafka/data
    depends_on:
      - zookeeper
    healthcheck:
      test: ["CMD", "kafka-topics", "--bootstrap-server", "localhost:9092", "--list"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: always

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
    restart: always

  # Ollama service
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_HOST=0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: always

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: always

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: always

volumes:
  workspace_data:
  logs_data:
  redis_data:
  kafka_data:
  zookeeper_data:
  ollama_data:
  prometheus_data:
  grafana_data:
```

#### 3. Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream dspy_agents {
        server dspy-agent:8000;
        # Add more agent instances for load balancing
        # server dspy-agent-2:8000;
        # server dspy-agent-3:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/ssl/cert.pem;
        ssl_certificate_key /etc/ssl/key.pem;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # Proxy to agent services
        location / {
            proxy_pass http://dspy_agents;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://dspy_agents/health;
            access_log off;
        }

        # Metrics endpoint (restrict access)
        location /metrics {
            proxy_pass http://dspy_agents/metrics;
            allow 10.0.0.0/8;
            deny all;
        }
    }
}
```

#### 4. Deployment Script

```bash
#!/bin/bash
# deploy.sh

set -e

echo "Deploying DSPy Agent to production..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed."; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed."; exit 1; }

# Set environment
export DOCKER_BUILDKIT=1
export COMPOSE_PROJECT_NAME=dspy-prod

# Pull latest images
echo "Pulling latest images..."
docker-compose -f docker-compose.production.yml pull

# Build custom images
echo "Building custom images..."
docker-compose -f docker-compose.production.yml build

# Start services
echo "Starting services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Health checks
echo "Running health checks..."
curl -f http://localhost/health || { echo "Health check failed"; exit 1; }

# Initialize Ollama model
echo "Initializing Ollama model..."
docker exec $(docker-compose -f docker-compose.production.yml ps -q ollama) ollama pull qwen3:7b

echo "Deployment completed successfully!"
echo "Services available at: https://your-domain.com"
echo "Monitoring available at: http://localhost:3000 (admin/admin)"
```

### Security Considerations

#### 1. Network Security

```bash
# Firewall rules (ufw example)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 9092/tcp   # Block external Kafka access
sudo ufw deny 6379/tcp   # Block external Redis access
sudo ufw enable
```

#### 2. Authentication and Authorization

```python
# Add to environment configuration
ENABLE_AUTH=true
JWT_SECRET_KEY=$(openssl rand -base64 32)
JWT_EXPIRATION_HOURS=24

# API key authentication
API_KEYS='["api-key-1", "api-key-2"]'

# RBAC configuration
RBAC_ENABLED=true
ADMIN_USERS='["admin@company.com"]'
```

#### 3. Data Encryption

```bash
# Encrypt sensitive data at rest
ENCRYPT_STATE=true
ENCRYPTION_KEY=$(openssl rand -base64 32)

# Use TLS for all communications
KAFKA_SSL_ENABLED=true
REDIS_TLS_ENABLED=true
```

## Cloud Deployments

### AWS Deployment

#### 1. ECS Fargate Deployment

```yaml
# ecs-task-definition.json
{
  "family": "dspy-agent",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/dspyAgentRole",
  "containerDefinitions": [
    {
      "name": "dspy-agent",
      "image": "your-account.dkr.ecr.region.amazonaws.com/dspy-agent:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "WORKSPACE",
          "value": "/opt/workspace"
        },
        {
          "name": "KAFKA_BOOTSTRAP",
          "value": "your-msk-cluster.kafka.region.amazonaws.com:9092"
        },
        {
          "name": "REDIS_URL",
          "value": "redis://your-elasticache-cluster.cache.amazonaws.com:6379"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:dspy/openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/dspy-agent",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### 2. Infrastructure as Code (Terraform)

```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# VPC and networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "dspy-agent-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
}

# ECS Cluster
resource "aws_ecs_cluster" "dspy_cluster" {
  name = "dspy-agent"
  
  capacity_providers = ["FARGATE"]
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Application Load Balancer
resource "aws_lb" "dspy_alb" {
  name               = "dspy-agent-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets           = module.vpc.public_subnets
}

# MSK (Managed Kafka)
resource "aws_msk_cluster" "dspy_kafka" {
  cluster_name           = "dspy-kafka"
  kafka_version          = "2.8.0"
  number_of_broker_nodes = 3
  
  broker_node_group_info {
    instance_type   = "kafka.m5.large"
    ebs_volume_size = 100
    client_subnets  = module.vpc.private_subnets
    security_groups = [aws_security_group.kafka_sg.id]
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "dspy_redis" {
  name       = "dspy-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_cluster" "dspy_redis" {
  cluster_id           = "dspy-redis"
  engine              = "redis"
  node_type           = "cache.m5.large"
  num_cache_nodes     = 1
  parameter_group_name = "default.redis6.x"
  port                = 6379
  subnet_group_name   = aws_elasticache_subnet_group.dspy_redis.name
  security_group_ids  = [aws_security_group.redis_sg.id]
}

# ECS Service
resource "aws_ecs_service" "dspy_service" {
  name            = "dspy-agent"
  cluster         = aws_ecs_cluster.dspy_cluster.id
  task_definition = aws_ecs_task_definition.dspy_task.arn
  desired_count   = 3
  launch_type     = "FARGATE"
  
  network_configuration {
    security_groups = [aws_security_group.ecs_sg.id]
    subnets         = module.vpc.private_subnets
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.dspy_tg.arn
    container_name   = "dspy-agent"
    container_port   = 8000
  }
}
```

### Google Cloud Platform (GKE)

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dspy-agent
  labels:
    app: dspy-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dspy-agent
  template:
    metadata:
      labels:
        app: dspy-agent
    spec:
      containers:
      - name: dspy-agent
        image: gcr.io/your-project/dspy-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: WORKSPACE
          value: "/opt/workspace"
        - name: KAFKA_BOOTSTRAP
          value: "kafka-service:9092"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: dspy-secrets
              key: openai-api-key
        resources:
          requests:
            cpu: 1
            memory: 2Gi
          limits:
            cpu: 2
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: workspace-storage
          mountPath: /opt/workspace
      volumes:
      - name: workspace-storage
        persistentVolumeClaim:
          claimName: workspace-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: dspy-agent-service
spec:
  selector:
    app: dspy-agent
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### Azure Container Instances

```yaml
# azure-container-group.yaml
apiVersion: 2021-03-01
location: eastus
name: dspy-agent-group
properties:
  containers:
  - name: dspy-agent
    properties:
      image: your-registry.azurecr.io/dspy-agent:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 4
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: WORKSPACE
        value: /opt/workspace
      - name: KAFKA_BOOTSTRAP
        value: your-eventhub-namespace.servicebus.windows.net:9093
      - name: REDIS_URL
        secureValue: redis://your-redis-cache.redis.cache.windows.net:6380
      volumeMounts:
      - name: workspace-volume
        mountPath: /opt/workspace
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
  volumes:
  - name: workspace-volume
    azureFile:
      shareName: dspy-workspace
      storageAccountName: your-storage-account
      storageAccountKey: your-storage-key
tags:
  environment: production
  application: dspy-agent
```

## Monitoring and Observability

### Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'dspy-agent'
    static_configs:
      - targets: ['dspy-agent:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka:9092']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
```

### Grafana Dashboards

Key metrics to monitor:

- **Agent Performance**
  - Request latency and throughput
  - Success/error rates
  - Active sessions
  - Memory and CPU usage

- **Learning Metrics**
  - RL reward trends
  - Policy performance
  - Learning rate convergence
  - Exploration vs exploitation ratios

- **Infrastructure Metrics**
  - Kafka message throughput
  - Redis cache hit rates
  - Ollama inference latency
  - Docker container health

### Logging Strategy

```yaml
# logging-config.yaml
version: 1
formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: json
    filename: /opt/logs/dspy-agent.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

  syslog:
    class: logging.handlers.SysLogHandler
    level: WARNING
    formatter: json
    address: ['localhost', 514]

loggers:
  dspy_agent:
    level: DEBUG
    handlers: [console, file, syslog]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

## Troubleshooting

### Common Deployment Issues

#### 1. Container Startup Failures

```bash
# Check container logs
docker logs dspy-agent

# Check resource constraints
docker stats

# Verify environment variables
docker exec dspy-agent env | grep DSPY

# Test connectivity
docker exec dspy-agent curl -f http://localhost:8000/health
```

#### 2. Service Discovery Issues

```bash
# Check network connectivity
docker network ls
docker network inspect lightweight_default

# Test inter-service communication
docker exec dspy-agent ping kafka
docker exec dspy-agent telnet redis 6379
```

#### 3. Performance Issues

```bash
# Monitor resource usage
docker stats --no-stream

# Check for memory leaks
docker exec dspy-agent ps aux --sort=-%mem

# Analyze slow queries
docker exec redis redis-cli --latency-history

# Check Kafka lag
docker exec kafka kafka-consumer-groups --bootstrap-server localhost:9092 --describe --all-groups
```

#### 4. Data Persistence Issues

```bash
# Check volume mounts
docker volume ls
docker volume inspect lightweight_redis_data

# Verify permissions
docker exec dspy-agent ls -la /opt/workspace

# Check disk space
docker exec dspy-agent df -h
```

### Recovery Procedures

#### 1. Service Recovery

```bash
# Restart individual service
docker compose -f docker/lightweight/docker-compose.yml restart dspy-agent

# Full stack restart
docker compose -f docker/lightweight/docker-compose.yml down
docker compose -f docker/lightweight/docker-compose.yml up -d

# Clean restart (removes containers but preserves data)
docker compose -f docker/lightweight/docker-compose.yml down --remove-orphans
docker compose -f docker/lightweight/docker-compose.yml up -d
```

#### 2. Data Recovery

```bash
# Backup important data
docker run --rm -v lightweight_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis-backup.tar.gz /data

# Restore from backup
docker run --rm -v lightweight_redis_data:/data -v $(pwd):/backup alpine tar xzf /backup/redis-backup.tar.gz -C /

# Reset learning state (if corrupted)
docker exec dspy-agent rm -f /opt/workspace/.dspy_rl_state.json
docker compose -f docker/lightweight/docker-compose.yml restart dspy-agent
```

#### 3. Emergency Procedures

```bash
# Scale down to single instance
docker compose -f docker/lightweight/docker-compose.yml up -d --scale dspy-worker=1

# Emergency stop
docker compose -f docker/lightweight/docker-compose.yml stop

# Force cleanup (WARNING: loses all data)
docker compose -f docker/lightweight/docker-compose.yml down --volumes
docker system prune -af
```

This deployment guide provides comprehensive coverage for deploying DSPy Agent across different environments. Choose the deployment strategy that best fits your requirements and infrastructure constraints.
