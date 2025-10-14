# Build Instructions for DSPy Infrastructure

**Production-ready build process for Rust env-manager and Go orchestrator**

---

## Prerequisites

### System Requirements

- **OS**: Linux (WSL2 on Windows, or native Linux)
- **Docker**: 20.10+ (for container management)
- **Disk Space**: ~5GB for build artifacts
- **Memory**: 4GB+ RAM recommended

### Software Dependencies

#### 1. Rust (Required for env-manager)

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Source the cargo environment
source ~/.cargo/env

# Verify installation
rustc --version  # Should be 1.70+
cargo --version
```

#### 2. Go (Required for orchestrator)

```bash
# Download Go 1.21+ (orchestrator requires 1.18+, but 1.21+ recommended)
wget https://go.dev/dl/go1.21.6.linux-amd64.tar.gz

# Extract to /usr/local
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.21.6.linux-amd64.tar.gz

# Add to PATH
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
echo 'export PATH=$PATH:$HOME/go/bin' >> ~/.bashrc
source ~/.bashrc

# Verify installation
go version  # Should be 1.21+
```

#### 3. Protocol Buffers (Required for gRPC)

```bash
# Install protoc
sudo apt-get update
sudo apt-get install -y protobuf-compiler

# Verify installation
protoc --version  # Should be 3.x+
```

#### 4. Buf (Required for proto generation)

```bash
# Download buf
wget https://github.com/bufbuild/buf/releases/download/v1.28.1/buf-Linux-x86_64 -O /tmp/buf

# Make executable and install
chmod +x /tmp/buf
sudo mv /tmp/buf /usr/local/bin/buf

# Verify installation
buf --version
```

---

## Build Process

### Option 1: Automated Build (Recommended)

```bash
# From project root
./scripts/build_infrastructure.sh
```

This script will:
1. Build Rust env-manager (release mode)
2. Build Go orchestrator (optimized)
3. Build Go CLI (dspy-agent)
4. Generate proto files
5. Run tests
6. Create artifacts in `dist/`

### Option 2: Manual Build

#### Step 1: Generate Proto Files

```bash
cd /mnt/c/Users/Admin/dspy-code

# Generate proto files for all languages
buf generate proto

# Verify generated files exist
ls -la orchestrator/internal/pb/orchestrator/
ls -la orchestrator/internal/pb/envmanager/
ls -la dspy_agent/infra/pb/orchestrator/
ls -la dspy_agent/infra/pb/env_manager/
```

#### Step 2: Build Rust env-manager

```bash
cd env_manager_rs

# Build in release mode (optimized)
cargo build --release

# Binary location
ls -lh target/release/env-manager

# Test the binary
./target/release/env-manager --version
```

**Build time**: ~1-2 minutes (first build), ~30 seconds (incremental)

**Binary size**: ~15MB (with optimizations and stripped debug symbols)

**Features enabled**:
- ✅ Retry logic with exponential backoff
- ✅ Configuration system (TOML + env vars)
- ✅ Enhanced logging with emojis
- ✅ Health checks for all services
- ✅ gRPC server

#### Step 3: Build Go Orchestrator

```bash
cd orchestrator

# Ensure dependencies are up to date
go mod tidy
go mod download

# Build with optimizations
go build -o orchestrator-linux \
  -ldflags='-s -w -X main.Version=0.1.0 -X main.BuildTime=$(date -u +%Y-%m-%dT%H:%M:%SZ)' \
  ./cmd/orchestrator

# Binary location
ls -lh orchestrator-linux

# Test the binary
./orchestrator-linux --help
```

**Build time**: ~1-2 minutes

**Binary size**: ~20MB (stripped)

**Features**:
- ✅ Adaptive concurrency control
- ✅ Workflow execution
- ✅ Slurm integration
- ✅ Metrics collection (Prometheus)
- ✅ Event bus (Kafka)
- ✅ HTTP API + gRPC server

#### Step 4: Build CLI (dspy-agent)

```bash
cd cmd/dspy-agent

# Build CLI
go build -o dspy-agent \
  -ldflags='-s -w' \
  .

# Make executable
chmod +x dspy-agent

# Test
./dspy-agent --help
```

---

## Build Artifacts

After building, you should have:

```
env_manager_rs/target/release/env-manager  (~15MB)
orchestrator/orchestrator-linux            (~20MB)
cmd/dspy-agent/dspy-agent                  (~25MB)
```

### Artifact Locations

```bash
# Rust env-manager
/mnt/c/Users/Admin/dspy-code/env_manager_rs/target/release/env-manager

# Go orchestrator
/mnt/c/Users/Admin/dspy-code/orchestrator/orchestrator-linux

# CLI
/mnt/c/Users/Admin/dspy-code/cmd/dspy-agent/dspy-agent
```

---

## Verification

### 1. Test Rust env-manager

```bash
cd env_manager_rs

# Check binary exists and is executable
ls -lh target/release/env-manager

# Run with help flag
./target/release/env-manager --help

# Test Docker connection (requires Docker running)
export DOCKER_HOST=unix:///var/run/docker.sock
./target/release/env-manager &
ENV_MANAGER_PID=$!

# Give it time to start
sleep 3

# Check if gRPC server is listening
netstat -tuln | grep 50100

# Kill test process
kill $ENV_MANAGER_PID
```

### 2. Test Go Orchestrator

```bash
cd orchestrator

# Check binary exists
ls -lh orchestrator-linux

# Test startup (Ctrl+C to exit)
./orchestrator-linux &
ORCH_PID=$!

# Wait for startup
sleep 3

# Check HTTP endpoint
curl http://localhost:9097/metrics

# Check if healthy
curl http://localhost:9097/queue/status

# Kill test process
kill $ORCH_PID
```

### 3. Test CLI

```bash
cd cmd/dspy-agent

# Check binary exists
ls -lh dspy-agent

# Test commands
./dspy-agent --version
./dspy-agent --help

# NOTE: Don't run 'start' yet - needs Docker services
```

---

## Current Build Status

✅ **Rust env-manager**: Built successfully  
- Binary: `env_manager_rs/target/release/env-manager`
- Size: ~15MB
- Status: Production ready

⚠️ **Go orchestrator**: Build blocked by Go version  
- Current Go: 1.18.1
- Required Go: 1.21+ (for newer gRPC features)
- Status: Needs Go upgrade

❓ **CLI (dspy-agent)**: Not built yet  
- Depends on: Go orchestrator
- Status: Waiting for Go upgrade

---

## Troubleshooting

### Issue: "cargo: command not found"

**Fix**:
```bash
source ~/.cargo/env
# Or add to ~/.bashrc permanently
echo 'source ~/.cargo/env' >> ~/.bashrc
```

### Issue: "go: package not found"

**Fix**:
```bash
cd orchestrator
go mod tidy
go mod download
```

### Issue: "protoc: command not found"

**Fix**:
```bash
sudo apt-get update
sudo apt-get install -y protobuf-compiler
```

### Issue: Go version too old

**Current Issue**: WSL has Go 1.18.1, but gRPC v1.76 requires 1.21+

**Fix**:
```bash
# Remove old Go
sudo rm -rf /usr/local/go

# Download new Go (1.21.6 or later)
wget https://go.dev/dl/go1.21.6.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.6.linux-amd64.tar.gz

# Update PATH
export PATH=/usr/local/go/bin:$PATH

# Verify
go version
```

### Issue: "Permission denied" when running binary

**Fix**:
```bash
chmod +x target/release/env-manager
chmod +x orchestrator-linux
chmod +x dspy-agent
```

---

## Build Optimization Flags

### Rust (Cargo.toml)

```toml
[profile.release]
opt-level = 3              # Maximum optimization
lto = true                 # Link-time optimization
codegen-units = 1          # Better optimization (slower build)
strip = true               # Strip debug symbols
panic = 'abort'            # Smaller binary (optional)
```

### Go (ldflags)

```bash
-s  # Strip debug symbols
-w  # Strip DWARF debugging info
-X main.Version=0.1.0  # Inject version string
```

---

## Installation (Production)

After building, install binaries system-wide:

```bash
# Create installation directory
sudo mkdir -p /usr/local/bin/dspy-agent/{bin,config}

# Copy binaries
sudo cp env_manager_rs/target/release/env-manager /usr/local/bin/dspy-agent/bin/
sudo cp orchestrator/orchestrator-linux /usr/local/bin/dspy-agent/bin/
sudo cp cmd/dspy-agent/dspy-agent /usr/local/bin/

# Make executable
sudo chmod +x /usr/local/bin/dspy-agent/bin/*
sudo chmod +x /usr/local/bin/dspy-agent

# Verify installation
which dspy-agent
dspy-agent --version
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Infrastructure

on: [push, pull_request]

jobs:
  build-rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Build env-manager
        run: |
          cd env_manager_rs
          cargo build --release
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: env-manager
          path: env_manager_rs/target/release/env-manager

  build-go:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.21'
      - name: Build orchestrator
        run: |
          cd orchestrator
          go build -o orchestrator-linux ./cmd/orchestrator
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: orchestrator
          path: orchestrator/orchestrator-linux
```

---

## Next Steps

1. **Upgrade Go**: Install Go 1.21+ in WSL
2. **Build orchestrator**: Once Go is upgraded
3. **Build CLI**: After orchestrator builds successfully
4. **Test integration**: Run full stack test
5. **Deploy**: Install to production path

---

## Summary

**Current Status**:
- ✅ Rust env-manager: Ready
- ⚠️ Go orchestrator: Needs Go 1.21+
- ❓ CLI: Waiting for orchestrator

**Action Required**:
```bash
# Upgrade Go to 1.21+
sudo rm -rf /usr/local/go
wget https://go.dev/dl/go1.21.6.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.6.linux-amd64.tar.gz
export PATH=/usr/local/go/bin:$PATH

# Then rebuild orchestrator
cd orchestrator
go build -o orchestrator-linux ./cmd/orchestrator

# Then build CLI
cd ../cmd/dspy-agent
go build -o dspy-agent .
```

---

**Last Updated**: October 12, 2025  
**Infrastructure Version**: 0.1.0  
**Build System**: Cargo (Rust) + Go modules
