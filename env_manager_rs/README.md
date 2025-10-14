# Environment Manager (Rust)

High-performance container lifecycle manager for DSPy agent infrastructure.

## Purpose

`env_manager_rs` is responsible for:
- Managing Docker containers (RedB, Redis, InferMesh, Ollama, etc.)
- Health checking and dependency resolution
- Providing gRPC API for orchestrator communication
- Replacing docker-compose with embedded service definitions

## Distinct from env_runner_rs

- **env_manager_rs**: Manages container **lifecycle** (start/stop services)
- **env_runner_rs**: Executes **tasks** within containers (actual workload execution)

## Building

```bash
cd env_manager_rs
cargo build --release
```

The binary will be at `target/release/env-manager`.

## Running

```bash
# Start with default settings
./target/release/env-manager

# With custom gRPC address
ENV_MANAGER_GRPC_ADDR=0.0.0.0:50100 ./target/release/env-manager

# With custom Docker host
DOCKER_HOST=unix:///var/run/docker.sock ./target/release/env-manager
```

## Environment Variables

- `ENV_MANAGER_GRPC_ADDR`: gRPC server address (default: `0.0.0.0:50100`)
- `ENV_MANAGER_METRICS_ADDR`: HTTP metrics/health address (default: `0.0.0.0:50101`)
- `DOCKER_HOST`: Docker socket/endpoint (default: system default)
- `ENV_MANAGER_CONFIG`: Optional TOML file to override service definitions (tenants, images, mounts)
- `RUST_LOG`: Logging level (default: `info`)

## gRPC API

See `proto/env_manager.v1.proto` for full API definition.

Key services:
- `StartServices`: Start containers with dependency resolution
- `StopServices`: Gracefully stop all containers
- `GetServicesStatus`: Query current status of all services
- `StreamHealth`: Real-time health updates
- `PullImages`: Pull required Docker images with progress

## Service Registry

Embedded service definitions for:
- **RedB**: Lightweight database (port 8080)
- **Redis**: Cache and pub/sub (port 6379)
- **InferMesh Nodes**: Inference services (ports 19001, 19002)
- **InferMesh Router**: Load balancer (port 19000)
- **Ollama**: Local LLM (port 11434)

## Architecture

```
env_manager_rs
├── container.rs        # Docker API integration (Bollard)
├── service_registry.rs # Service definitions
├── health.rs          # Health checking logic
├── manager.rs         # Orchestration engine
├── grpc_server.rs     # gRPC API implementation
└── main.rs           # Entry point
```

## Metrics & Health

`env-manager` exposes two HTTP endpoints (defaults to `0.0.0.0:50101`):

- `GET /health` – readiness/liveness probe used by systemd/Kubernetes.
- `GET /metrics` – Prometheus-friendly JSON (queue depth, active services, Docker timings).

Service startups are reported with Docker API duration histograms (`docker_api_duration_seconds`) and `active_services` gauge, which keep the orchestrator and dashboards in sync.

