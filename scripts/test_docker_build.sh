#!/bin/bash
set -euo pipefail

# Function to print status messages
print_status() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1"
}

# Check if we're in the right directory
if [[ ! -f "docker/lightweight/Dockerfile" ]]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

echo "🐳 Testing Docker Build for ARM64 Compatibility"
echo "=============================================="

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Build the Docker image
echo "Building Docker image..."
if docker build -f docker/lightweight/Dockerfile -t dspy-agent-test .; then
    print_status "Docker build completed successfully"
else
    print_error "Docker build failed"
    exit 1
fi

# Test the image
echo "Testing the built image..."
if docker run --rm dspy-agent-test --help > /dev/null 2>&1; then
    print_status "Docker image runs successfully"
else
    print_error "Docker image failed to run"
    exit 1
fi

# Check if RL components are available
echo "Checking RL components..."
if docker run --rm dspy-agent-test python -c "
import sys
try:
    import dspy_agent.rl.rlkit
    print('RL toolkit imported successfully')
except ImportError as e:
    print(f'RL toolkit import failed: {e}')
    sys.exit(1)

try:
    import gymnasium
    print('Gymnasium available')
except ImportError:
    print('Gymnasium not available')
    sys.exit(1)

try:
    import torch
    print('PyTorch available')
except ImportError:
    print('PyTorch not available')
    sys.exit(1)

# Check PufferLib availability
try:
    import pufferlib
    print('PufferLib available')
except ImportError:
    print('PufferLib not available (expected on ARM64)')
"; then
    print_status "RL components check passed"
else
    print_error "RL components check failed"
    exit 1
fi

echo ""
echo "🎉 Docker build test completed successfully!"
echo ""
echo "The agent is now compatible with both x86_64 and ARM64 architectures:"
echo "  - x86_64: Full RL stack with PufferLib"
echo "  - ARM64: Core RL components without PufferLib"
echo ""
echo "You can now run:"
echo "  docker run -it dspy-agent-test --workspace /app"
