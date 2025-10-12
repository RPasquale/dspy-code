#!/bin/bash
set -euo pipefail

# Build Official InferMesh from Source
# This script builds the official InferMesh from the redbco repository

echo "🔧 Building Official InferMesh from Source"
echo "=========================================="

WORKSPACE_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Check if Rust is installed
if ! command -v cargo >/dev/null 2>&1; then
    echo "❌ Rust/Cargo not found. Please install Rust first:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Create build directory
BUILD_DIR="/tmp/infermesh-build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "📥 Cloning official InferMesh repository..."
if [ ! -d "infermesh" ]; then
    git clone https://github.com/redbco/infermesh.git
fi

cd infermesh

# Apply local patches if available
PATCH_FILE="$WORKSPACE_ROOT/docker/lightweight/infermesh-config/official_infermesh.patch"
if [ -f "$PATCH_FILE" ]; then
    echo "🩹 Applying local InferMesh patch set..."
    git reset --hard HEAD >/dev/null
    git clean -fd >/dev/null
    if ! git apply "$PATCH_FILE" >/dev/null 2>&1; then
        echo "⚠️  Patch did not apply cleanly; continuing with upstream sources."
    fi
fi

echo "🔨 Building InferMesh (skipping host build if protoc unavailable)..."
if command -v protoc >/dev/null 2>&1; then
    cargo build --release
else
    echo "⚠️  protoc not found on host; relying on Docker multi-stage build."
fi

echo "🏗️  Creating Docker image..."
cat > Dockerfile << 'EOF'
# Multi-stage build for official InferMesh
FROM rust:1.82-slim as builder

# Resource limits to prevent system crashes
ARG CARGO_BUILD_JOBS=2
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}
ENV CARGO_INCREMENTAL=0
ENV RUSTFLAGS="-C codegen-units=4"

# Install dependencies including protobuf compiler
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code
COPY . .

# Fix the compilation error in mesh-metrics
RUN sed -i 's/registry\.register(Box::new(pc))/_registry.register(Box::new(pc))/' crates/mesh-metrics/src/prometheus_metrics.rs

# Build the project with limited parallelism to prevent system crashes
# Enable prometheus process feature for Linux
RUN cargo build --release -j ${CARGO_BUILD_JOBS} --features prometheus/process

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built binaries
COPY --from=builder /app/target/release/meshd /usr/local/bin/meshd
COPY --from=builder /app/target/release/mesh-sim /usr/local/bin/mesh-sim
COPY --from=builder /app/target/release/mesh /usr/local/bin/mesh
COPY --from=builder /app/target/release/mesh-router /usr/local/bin/mesh-router

# Expose ports
EXPOSE 9000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Run InferMesh (meshd is the main daemon)
CMD ["/usr/local/bin/meshd"]
EOF

# Build Docker image with resource limits
DOCKER_BUILDKIT=0 docker build --build-arg CARGO_BUILD_JOBS=2 -t official-infermesh:latest .

echo "✅ Official InferMesh built successfully!"
echo "   Image: official-infermesh:latest"
echo ""
echo "🚀 To use the official InferMesh:"
echo "   1. Update docker-compose.yml to use 'official-infermesh:latest'"
echo "   2. Start services: docker compose up -d"
echo "   3. Test: curl http://localhost:19000/health"
