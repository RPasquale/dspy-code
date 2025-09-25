#!/bin/bash
set -euo pipefail

# Full Agent Deploy Script
# This script:
# 1. Rebuilds all containers with no cache
# 2. Publishes containers to registry (if configured)
# 3. Starts the full stack
# 4. Runs health checks and smoke tests
# 5. Opens the agent interface
# 6. Launches the React frontend

echo "[full-deploy] Starting comprehensive agent deployment..."

# Change to project root
cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[full-deploy]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[full-deploy]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[full-deploy]${NC} $1"
}

print_error() {
    echo -e "${RED}[full-deploy]${NC} $1"
}

# 1. Rebuild all containers with no cache
print_status "Rebuilding all containers with no cache..."
DOCKER_BUILDKIT=1 docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env build --no-cache

# 2. Publish containers (if INFERMESH_IMAGE is configured)
if grep -q "^INFERMESH_IMAGE=" docker/lightweight/.env 2>/dev/null; then
    print_status "Publishing containers to registry..."
    make infermesh-push || print_warning "Failed to push InferMesh image (optional)"
else
    print_warning "INFERMESH_IMAGE not configured, skipping container publishing"
fi

# 3. Start the full stack
print_status "Starting the full lightweight stack..."
make stack-up

# 4. Wait for services to be ready
print_status "Waiting for services to initialize..."
sleep 10

# 5. Run health checks
print_status "Running health checks..."
make health-check || print_warning "Some health checks failed (continuing...)"

# 6. Run smoke tests
print_status "Running smoke tests..."
make smoke || print_warning "Smoke tests had issues (continuing...)"

# 7. Check container status
print_status "Checking container status..."
make stack-ps

# 8. Test agent functionality
print_status "Testing agent functionality..."
if docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env exec dspy-agent dspy-agent --help >/dev/null 2>&1; then
    print_success "Agent is working correctly!"
else
    print_error "Agent test failed!"
    exit 1
fi

# 9. Open agent interface (if available)
print_status "Opening agent interface..."
if command -v open >/dev/null 2>&1; then
    # macOS
    open "http://127.0.0.1:8765" 2>/dev/null || print_warning "Could not open agent interface"
elif command -v xdg-open >/dev/null 2>&1; then
    # Linux
    xdg-open "http://127.0.0.1:8765" 2>/dev/null || print_warning "Could not open agent interface"
else
    print_warning "No browser opener found, agent interface available at: http://127.0.0.1:8765"
fi

# 10. Launch React frontend
print_status "Launching React frontend..."
if command -v open >/dev/null 2>&1; then
    # macOS
    open "http://127.0.0.1:18081/dashboard" 2>/dev/null || print_warning "Could not open dashboard"
elif command -v xdg-open >/dev/null 2>&1; then
    # Linux
    xdg-open "http://127.0.0.1:18081/dashboard" 2>/dev/null || print_warning "Could not open dashboard"
else
    print_warning "No browser opener found, dashboard available at: http://127.0.0.1:18081/dashboard"
fi

# 11. Show useful URLs
print_success "Deployment completed successfully!"
echo ""
echo "🌐 Available Services:"
echo "   • Agent Interface: http://127.0.0.1:8765"
echo "   • Dashboard: http://127.0.0.1:18081/dashboard"
echo "   • InferMesh: http://127.0.0.1:19000/health"
echo "   • Spark UI: http://127.0.0.1:4041"
echo "   • Embed Worker: http://127.0.0.1:9101/metrics"
echo ""
echo "📊 Monitoring:"
echo "   • Container logs: make stack-logs"
echo "   • Health check: make health-check"
echo "   • Container status: make stack-ps"
echo ""
echo "🔧 Management:"
echo "   • Stop stack: make stack-down"
echo "   • Restart agent: make stack-reload"
echo ""

# 12. Optional: Start monitoring logs in background
if [[ "${1:-}" == "--monitor" ]]; then
    print_status "Starting log monitoring (press Ctrl+C to stop)..."
    make stack-logs
fi
