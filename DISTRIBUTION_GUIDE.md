# DSPy Agent Distribution Guide

## 🚀 One-Command Setup for Users

The DSPy Agent system now includes a comprehensive packaging system that allows users to download and run everything with a single command.

## 📦 Creating Distribution Packages

### For Developers (Building the Package)

```bash
# Create a complete distribution package
make package

# Or run the packager directly
bash scripts/dspy_stack_packager.sh
```

This creates:
- `dist/dspy_stack_bundle_<timestamp>/` - Complete bundle directory
- `dist/dspy_stack_bundle_<timestamp>.tar.gz` - Compressed archive
- `dist/dspy_stack_bundle_<timestamp>.tar.gz.sha256` - Checksum file

### Package Contents

The distribution package includes:
- **Pre-built binaries**: Go orchestrator and Rust env-runner
- **Docker Compose stack**: Complete containerized environment
- **Slurm integration**: GPU job submission templates
- **Startup scripts**: One-command bootstrap
- **Documentation**: Complete usage guides
- **Source code**: For advanced customization

## 👥 For End Users (Using the Package)

### Quick Start

1. **Download the package**:
   ```bash
   # Extract the archive
   tar -xzf dspy_stack_bundle_<timestamp>.tar.gz
   cd dspy_stack_bundle_<timestamp>
   ```

2. **Run the system**:
   ```bash
   # One command to start everything
   ./start_bundle.sh
   ```

3. **Access services**:
   - Dashboard: http://localhost:8080
   - Orchestrator API: http://localhost:9097
   - Env-Runner API: http://localhost:8080
   - Metrics: http://localhost:9097/metrics

### Prerequisites

The bundle automatically checks for:
- **Docker** (20+)
- **Docker Compose** (plugin or legacy)
- **OpenSSL** (for secure tokens)
- **curl** (optional, for health checks)
- **Slurm tools** (optional, for GPU jobs)

### What the Bundle Does

The `start_bundle.sh` script:
1. ✅ Validates all dependencies
2. ✅ Generates secure environment configuration
3. ✅ Creates necessary directories
4. ✅ Builds Docker images
5. ✅ Starts all services
6. ✅ Runs health checks
7. ✅ Provides usage instructions

## 🛠️ Advanced Usage

### Custom Configuration

```bash
# Set custom RedDB token
export REDDB_ADMIN_TOKEN="your-custom-token"
./start_bundle.sh

# Use existing environment file
cp your-config.env docker/lightweight/.env
./start_bundle.sh
```

### Service Management

```bash
# Check service status
docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env ps

# View logs
docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env logs -f

# Stop services
docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env down
```

### Slurm Integration

If Slurm is available:
```bash
# Submit GPU training job
curl -X POST http://localhost:9097/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "id": "my-training-job",
    "class": "gpu_slurm",
    "payload": {
      "script": "train_grpo_agent.sbatch",
      "nodes": 2,
      "gpus": 4
    }
  }'

# Check job status
curl http://localhost:9097/slurm/status/my-training-job
```

## 🔧 Development Workflow

### Building Packages

```bash
# Clean previous packages
make package-clean

# Create new package
make package

# Test the package locally
cd dist/dspy_stack_bundle_<timestamp>
./start_bundle.sh
```

### Continuous Integration

The packaging system is designed for CI/CD:
- Builds Go and Rust binaries
- Creates reproducible packages
- Generates checksums for verification
- Includes all necessary dependencies

## 📋 Package Structure

```
dspy_stack_bundle_<timestamp>/
├── bin/                          # Pre-built binaries
│   ├── orchestrator              # Go orchestrator
│   └── env_runner                # Rust env-runner
├── orchestrator/                 # Go source + binaries
├── env_runner_rs/                # Rust source + binaries
├── deploy/slurm/                 # Slurm templates
├── docker/lightweight/          # Docker Compose stack
├── scripts/                     # Utility scripts
├── start_bundle.sh              # Main startup script
├── README_BUNDLE.md             # Bundle documentation
└── ... (all other project files)
```

## 🚨 Troubleshooting

### Common Issues

1. **Docker not found**:
   ```bash
   # Install Docker
   # macOS: brew install docker
   # Ubuntu: sudo apt install docker.io
   ```

2. **OpenSSL missing**:
   ```bash
   # Install OpenSSL
   # macOS: brew install openssl
   # Ubuntu: sudo apt install openssl
   ```

3. **Permission denied**:
   ```bash
   chmod +x start_bundle.sh
   ```

4. **Port conflicts**:
   ```bash
   # Check what's using ports
   lsof -i :8080 -i :9097
   ```

### Health Checks

```bash
# Check orchestrator
curl http://localhost:9097/metrics

# Check env-runner
curl http://localhost:8080/health

# Check queue status
curl http://localhost:9097/queue/status
```

## 📚 Additional Resources

- **Complete System README**: `COMPLETE_SYSTEM_README.md`
- **InferMesh Optimization**: `INFERMESH_OPTIMIZATION_GUIDE.md`
- **Slurm Integration**: `deploy/slurm/README.md`
- **Go Orchestrator**: `orchestrator/README.md`
- **Rust Env-Runner**: `env_runner_rs/README.md`

## 🎯 Next Steps

1. **Test the package**: Run `make package` and test locally
2. **Distribute**: Share the `.tar.gz` file with users
3. **Document**: Update your project documentation
4. **Automate**: Integrate with your CI/CD pipeline

The packaging system makes the DSPy Agent accessible to anyone with Docker, regardless of their Go/Rust/Python setup!
