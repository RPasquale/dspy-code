# Build Instructions

Complete instructions for building the optimized DSPy agent infrastructure.

## Prerequisites

Install the following tools:

- **Docker**: 20.10+ (running and accessible)
- **Rust**: 1.70+ with Cargo
- **Go**: 1.20+
- **Python**: 3.11+
- **Protocol Buffers Compiler** (optional, for regenerating protos)

### Verify Prerequisites

```bash
docker ps                    # Should work without errors
rustc --version             # Should show 1.70+
cargo --version             # Should be present
go version                  # Should show 1.20+
python --version            # Should show 3.11+
```

## Build Steps

### 1. Build Rust Environment Manager

```bash
cd env_manager_rs
cargo build --release
cd ..
```

**Output:** `env_manager_rs/target/release/env-manager` (or `.exe` on Windows)

**Time:** ~2-5 minutes (first build)

### 2. Build Go Orchestrator (Enhanced)

The orchestrator build is optional if you're using the unified CLI, but recommended for standalone use:

```bash
cd orchestrator
go mod tidy
go build -o bin/orchestrator ./cmd/orchestrator
cd ..
```

**Output:** `orchestrator/bin/orchestrator`

**Time:** ~30 seconds

### 3. Build Unified CLI

```bash
cd cmd/dspy-agent
go mod tidy
go build -o dspy-agent
cd ../..
```

**Output:** `cmd/dspy-agent/dspy-agent` (or `.exe` on Windows)

**Time:** ~30 seconds

### 4. Install Python Dependencies

```bash
# Install Python gRPC tools
pip install grpcio grpcio-tools

# Install DSPy agent package (development mode)
pip install -e .
```

### 5. Generate Protocol Buffer Code (Optional)

Only needed if you modified `.proto` files:

```bash
# Using Makefile
make proto-go
make proto-python

# Or manually
protoc --go_out=orchestrator/internal/pb --go-grpc_out=orchestrator/internal/pb proto/*.proto
python -m grpc_tools.protoc -I proto --python_out=dspy_agent/infra/pb --grpc_python_out=dspy_agent/infra/pb proto/*.proto
```

## Installation

### Option 1: Local Install (Recommended for Development)

Keep binaries in project:

```bash
# Binaries are already in their respective directories
# env_manager_rs/target/release/env-manager
# cmd/dspy-agent/dspy-agent

# Add to PATH (optional)
export PATH="$PWD/cmd/dspy-agent:$PATH"
```

### Option 2: System Install

Install globally:

```bash
# Install CLI
sudo cp cmd/dspy-agent/dspy-agent /usr/local/bin/

# Install env_manager
sudo cp env_manager_rs/target/release/env-manager /usr/local/bin/

# Install orchestrator
sudo cp orchestrator/bin/orchestrator /usr/local/bin/
```

### Option 3: User Install (No sudo)

```bash
# Create user bin directory
mkdir -p ~/.local/bin

# Install binaries
cp cmd/dspy-agent/dspy-agent ~/.local/bin/
cp env_manager_rs/target/release/env-manager ~/.local/bin/
cp orchestrator/bin/orchestrator ~/.local/bin/

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"
```

## Verification

### Test Rust Component

```bash
# Run env_manager
env_manager_rs/target/release/env-manager --help

# Or if installed
env-manager --help

# Should show usage information
```

### Test Go Orchestrator

```bash
# Run orchestrator (will fail without services, that's OK)
orchestrator/bin/orchestrator &
PID=$!

# Wait a moment
sleep 2

# Check it's running
ps -p $PID

# Stop it
kill $PID
```

### Test Unified CLI

```bash
# Run CLI
cmd/dspy-agent/dspy-agent --version

# Should show version info
```

### Test Full Stack

```bash
# Initialize configuration
dspy-agent config init

# Start all services
dspy-agent start

# Check status
dspy-agent status

# Stop services
dspy-agent stop
```

## Platform-Specific Notes

### Linux

Standard build process works. Ensure Docker socket is accessible:

```bash
# Check Docker socket
ls -l /var/run/docker.sock

# Add user to docker group if needed
sudo usermod -aG docker $USER
# Log out and back in
```

### macOS

Works with Docker Desktop:

```bash
# Ensure Docker Desktop is running
open -a Docker

# Wait for Docker to start
docker ps
```

### Windows

Use PowerShell or Git Bash:

```powershell
# Build Rust (PowerShell)
cd env_manager_rs
cargo build --release
cd ..

# Build Go
cd cmd\dspy-agent
go build -o dspy-agent.exe
cd ..\..

# Run
.\cmd\dspy-agent\dspy-agent.exe --help
```

Docker Desktop must be running.

## Troubleshooting

### Rust Build Fails

```bash
# Update Rust
rustup update

# Clean and rebuild
cd env_manager_rs
cargo clean
cargo build --release
```

### Go Build Fails

```bash
# Update dependencies
cd cmd/dspy-agent
go mod tidy
go clean
go build
```

### Protobuf Generation Fails

```bash
# Install protoc
# Ubuntu/Debian:
sudo apt install protobuf-compiler

# macOS:
brew install protobuf

# Or download from: https://github.com/protocolbuffers/protobuf/releases

# Install Go plugins
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Install Python tools
pip install grpcio-tools
```

### Docker Permission Denied

```bash
# Linux: Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

### Missing Dependencies

```bash
# Rust dependencies are automatically downloaded by Cargo

# Go dependencies
cd cmd/dspy-agent
go mod download

cd ../../orchestrator
go mod download

# Python dependencies
pip install -r requirements.txt
```

## Development Build

For faster iteration during development:

```bash
# Rust debug build (faster compile, slower runtime)
cd env_manager_rs
cargo build  # Without --release
cd ..

# Go with race detector
cd cmd/dspy-agent
go build -race
cd ../..
```

## Clean Build

To start fresh:

```bash
# Clean Rust
cd env_manager_rs
cargo clean
cd ..

# Clean Go
cd cmd/dspy-agent
go clean
rm -f dspy-agent
cd ../..

cd orchestrator
go clean
rm -rf bin/
cd ..

# Clean Python
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type d -name "*.egg-info" -exec rm -rf {} +

# Rebuild everything
cd env_manager_rs && cargo build --release && cd ..
cd cmd/dspy-agent && go build && cd ../..
```

## Build Artifacts

After successful build, you should have:

```
env_manager_rs/target/release/
└── env-manager                    # Rust binary

cmd/dspy-agent/
└── dspy-agent                     # Go CLI binary

orchestrator/bin/
└── orchestrator                   # Go orchestrator binary

orchestrator/internal/pb/
├── orchestrator/                  # Generated Go code
└── envmanager/                    # Generated Go code

dspy_agent/infra/pb/
├── orchestrator_v1_pb2.py         # Generated Python code
└── orchestrator_v1_pb2_grpc.py    # Generated Python code
```

## Next Steps

After successful build:

1. **Initialize:** `dspy-agent config init`
2. **Start:** `dspy-agent start`
3. **Verify:** `dspy-agent status`
4. **Test:** Run examples from `examples/` directory
5. **Develop:** Write your DSPy agent using `AgentInfra`

See `docs/QUICKSTART.md` for detailed usage instructions.

