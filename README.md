# DSPy Agent - Production-Ready AI Coding Assistant

A high-performance, production-ready AI coding assistant with advanced orchestration, distributed training, and real-time monitoring capabilities.

## 🏗️ System Architecture

### Core Components

#### 1. **DSPy Agent Core** (Python)
- **Orchestrator**: Central component for routing user queries to appropriate tools
- **Skills System**: Modular AI capabilities (Controller, CodeContext, TaskAgent, FileLocator, DataRAG)
- **Streaming Engine**: Real-time data processing with Kafka integration
- **Reinforcement Learning**: GRPO (Group Relative Preference Optimization) training methodology
- **Training System**: GEPA, Teleprompt, and GRPO for agent module optimization

#### 2. **Go Orchestrator** (Go)
- **Structured Concurrency**: Advanced task orchestration with adaptive semaphores
- **Metrics Integration**: Prometheus metrics for monitoring and optimization
- **Slurm Integration**: GPU cluster job submission and management
- **Queue Management**: Environment task queuing and processing
- **Event Bus**: Real-time event publishing and subscription

#### 3. **Rust Environment Runner** (Rust)
- **High-Performance Execution**: Low-latency environment task execution
- **File System Monitoring**: Real-time file change detection with `notify` crate
- **Metrics Server**: HTTP metrics endpoint for monitoring
- **Workload Classes**: Support for different task types (CPU, GPU, Slurm)

#### 4. **Rust RedDB Server** (Rust)
- **High-Performance Database**: Key-value storage, vector operations, document search
- **SQLite Backend**: Persistent storage with SQLite
- **HTTP API**: RESTful API for database operations
- **Authentication**: Token-based authentication for security
- **Streaming Support**: Real-time data streaming capabilities

#### 5. **InferMesh Integration** (Python)
- **High-Throughput Embeddings**: Microservice-style text embedding service
- **Model Serving**: Efficient model loading and inference
- **Caching**: HuggingFace model caching for performance
- **Health Monitoring**: Built-in health checks and metrics

#### 6. **Redis Cache** (Redis)
- **Session Management**: User session storage and management
- **Caching Layer**: High-performance caching for frequently accessed data
- **Pub/Sub**: Real-time message broadcasting

### Infrastructure Components

#### Docker Compose Stack
- **Multi-Container Architecture**: All services containerized for easy deployment
- **Service Discovery**: Automatic service discovery and networking
- **Health Checks**: Built-in health monitoring for all services
- **Resource Management**: CPU and memory limits for optimal performance

#### Monitoring & Observability
- **Prometheus Metrics**: Comprehensive metrics collection
- **Health Endpoints**: Service health monitoring
- **Log Aggregation**: Centralized logging with structured output
- **Performance Monitoring**: Real-time performance metrics

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Rust (for local development)
- Go (for local development)
- Python 3.11+

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd dspy_stuff
```

2. **Build the system**:
```bash
# Build all components (Go, Rust, Python)
make build

# Or build individual components
make build-go      # Build Go orchestrator
make build-rust     # Build Rust components
make build-python   # Build Python components
```

3. **Start the system**:
```bash
# Start all services
make up

# Or start specific services
make up-core       # Start core services only
make up-full       # Start all services including monitoring
```

4. **Verify installation**:
```bash
# Check service health
make health

# View service logs
make logs

# Check service status
make status
```

## 🔧 Development

### Local Development Setup

1. **Install dependencies**:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Go
# Follow official Go installation guide

# Install Python dependencies
pip install -e .
```

2. **Run tests**:
```bash
# Run all tests
make test

# Run specific test suites
make test-go       # Go tests
make test-rust     # Rust tests
make test-python   # Python tests
```

3. **Development workflow**:
```bash
# Start development environment
make dev

# Run linting
make lint

# Format code
make fmt
```

### Service Configuration

#### Environment Variables
```bash
# Core configuration
REDDB_ADMIN_TOKEN=your_admin_token
REDDB_DATA_DIR=/data
ENV_QUEUE_DIR=logs/env_queue
ORCHESTRATOR_DEMO=0

# Service ports
REDDB_PORT=8080
ORCHESTRATOR_PORT=9097
ENV_RUNNER_PORT=8080
INFERMESH_PORT=9000
```

#### Docker Compose Configuration
The system uses a multi-stage Docker build with:
- **Go Builder**: Compiles Go orchestrator
- **Rust Builder**: Compiles Rust components (env-runner, reddb)
- **Python Runtime**: Final runtime with all components

## 📊 Monitoring & Metrics

### Health Endpoints
- **RedDB**: `http://localhost:8082/health`
- **Go Orchestrator**: `http://localhost:9097/metrics`
- **Rust Env Runner**: `http://localhost:8080/health`
- **InferMesh**: `http://localhost:19000/health`

### Metrics Collection
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Custom Metrics**: Service-specific metrics and KPIs

### Logging
- **Structured Logs**: JSON-formatted logs for all services
- **Log Aggregation**: Centralized log collection
- **Log Rotation**: Automatic log rotation and cleanup

## 🎯 Advanced Features

### Distributed Training
- **Slurm Integration**: GPU cluster job submission
- **Multi-GPU Training**: Distributed training across multiple GPUs
- **Job Management**: Queue management and job monitoring
- **Resource Allocation**: Dynamic resource allocation based on workload

### Performance Optimization
- **Adaptive Concurrency**: Dynamic concurrency limits based on system load
- **Caching**: Multi-level caching for improved performance
- **Batch Processing**: Efficient batch processing for large datasets
- **Resource Management**: Intelligent resource allocation and management

### Security
- **Authentication**: Token-based authentication for all services
- **Authorization**: Role-based access control
- **Encryption**: Data encryption in transit and at rest
- **Audit Logging**: Comprehensive audit trail

## 🛠️ API Reference

### RedDB API
```bash
# Health check
curl http://localhost:8082/health

# Stream operations
curl -X POST http://localhost:8082/streams/{namespace}/{stream} \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{"data": "your_data"}'
```

### Go Orchestrator API
```bash
# Metrics
curl http://localhost:9097/metrics

# Queue status
curl http://localhost:9097/queue/status

# Submit task
curl -X POST http://localhost:9097/queue/submit \
  -H "Content-Type: application/json" \
  -d '{"class": "cpu_short", "payload": {"task": "example"}}'
```

### Rust Env Runner API
```bash
# Health check
curl http://localhost:8080/health

# Metrics
curl http://localhost:8080/metrics
```

## 🚀 Deployment

### Production Deployment
```bash
# Build production images
make build-prod

# Deploy to production
make deploy-prod

# Scale services
make scale-up
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

## 📈 Performance Tuning

### System Optimization
- **Memory Management**: Optimized memory usage across all components
- **CPU Optimization**: Efficient CPU utilization
- **Network Optimization**: Optimized network communication
- **Storage Optimization**: Efficient storage usage and I/O

### Monitoring & Alerting
- **Real-time Monitoring**: Continuous system monitoring
- **Alerting**: Automated alerting for critical issues
- **Performance Metrics**: Detailed performance metrics and KPIs
- **Capacity Planning**: Resource capacity planning and optimization

## 🔍 Troubleshooting

### Common Issues
1. **Port Conflicts**: Ensure no other services are using required ports
2. **Permission Issues**: Check file permissions for data directories
3. **Resource Limits**: Monitor system resources and adjust limits
4. **Network Issues**: Verify network connectivity between services

### Debug Commands
```bash
# Check service logs
docker logs <service_name>

# Check service health
curl http://localhost:<port>/health

# Check system resources
docker stats

# Check network connectivity
docker network ls
```

## 📚 Documentation

- **Architecture Guide**: `docs/architecture.md`
- **API Documentation**: `docs/api.md`
- **Deployment Guide**: `docs/deployment.md`
- **Troubleshooting Guide**: `docs/troubleshooting.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

Apache-2.0 (see LICENSE)

## 🆘 Support

For support and questions:
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: Project documentation
- **Community**: Join our community forum

---

**Note**: This system is production-ready and includes comprehensive monitoring, security, and performance optimization features. All components are designed to work together seamlessly for optimal performance and reliability.