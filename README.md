# DSPy Agent

## Build

```bash
# Build all components
make build

# Or build individual components
make build-go      # Build Go orchestrator
make build-rust     # Build Rust components
make build-python   # Build Python components
```

## Run

```bash
# Start all services
make up

# Or start specific services
make up-core       # Start core services only
make up-full       # Start all services including monitoring
```

## Development

```bash
# Start development environment
make dev

# Run tests
make test

# Check service health
make health

# View service logs
make logs
```