#!/bin/bash
set -euo pipefail

# Enhanced Deploy Script with Comprehensive Testing
# This script:
# 1. Rebuilds all containers with no cache
# 2. Publishes containers to registry (if configured)
# 3. Starts the full stack
# 4. Runs comprehensive test suite
# 5. Feeds test results back to agent for learning
# 6. Opens interfaces only if tests pass
# 7. Provides detailed test reporting

echo "[enhanced-deploy] Starting comprehensive agent deployment with testing..."

# Change to project root
cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[enhanced-deploy]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[enhanced-deploy]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[enhanced-deploy]${NC} $1"
}

print_error() {
    echo -e "${RED}[enhanced-deploy]${NC} $1"
}

print_test() {
    echo -e "${PURPLE}[test-suite]${NC} $1"
}

# Test results tracking
TEST_RESULTS_FILE="/tmp/test_results.json"
TEST_FAILED=false

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
sleep 15

# 5. Run health checks
print_status "Running health checks..."
make health-check || print_warning "Some health checks failed (continuing...)"

# 6. Run comprehensive test suite
print_test "🧪 Starting comprehensive test suite..."

# Create test suite service container
print_test "Building test suite service..."
docker build -f docker/lightweight/test_suite.Dockerfile -t test-suite-service .

# Run test suite
print_test "Running test suite..."
if docker run --rm \
    --network host \
    -v "$(pwd):/workspace" \
    -v "$(pwd)/logs:/workspace/logs" \
    test-suite-service > "$TEST_RESULTS_FILE" 2>&1; then
    
    print_success "✅ Test suite completed successfully!"
    TEST_FAILED=false
else
    print_error "❌ Test suite failed!"
    TEST_FAILED=true
fi

# 7. Parse and display test results
if [[ -f "$TEST_RESULTS_FILE" ]]; then
    print_test "📊 Test Results Summary:"
    echo "----------------------------------------"
    
    # Extract summary from test results
    if command -v jq >/dev/null 2>&1; then
        echo "Test results saved to: $TEST_RESULTS_FILE"
        echo "Summary:"
        jq -r '.summary | "Total: \(.total_tests), Passed: \(.passed), Failed: \(.failed), Skipped: \(.skipped)"' "$TEST_RESULTS_FILE" 2>/dev/null || echo "Could not parse JSON results"
    else
        echo "Test results saved to: $TEST_RESULTS_FILE"
        echo "Install 'jq' for better result parsing"
    fi
    
    echo "----------------------------------------"
fi

# 8. Feed test results to agent (if available)
if [[ -f "$TEST_RESULTS_FILE" ]]; then
    print_test "🤖 Feeding test results to agent for learning..."
    
    # Publish test results to agent learning stream
    if docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env ps dspy-agent | grep -q "Up"; then
        # Copy test results to agent container
        docker cp "$TEST_RESULTS_FILE" "$(docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env ps -q dspy-agent):/tmp/test_results.json"
        
        # Trigger agent learning from test results
        docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env exec dspy-agent python -c "
import json
import asyncio
from dspy_agent.streaming.bus import LocalBus

async def feed_results():
    bus = LocalBus()
    with open('/tmp/test_results.json', 'r') as f:
        results = json.load(f)
    await bus.publish('agent.test_results', results)
    print('Test results fed to agent learning system')

asyncio.run(feed_results())
" 2>/dev/null || print_warning "Could not feed results to agent"
    else
        print_warning "Agent not available for learning integration"
    fi
fi

# 9. Only proceed with interface opening if tests passed
if [[ "$TEST_FAILED" == "false" ]]; then
    print_success "🎉 All tests passed! Opening interfaces..."
    
    # Test agent functionality
    print_status "Testing agent functionality..."
    if docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env exec dspy-agent dspy-agent --help >/dev/null 2>&1; then
        print_success "Agent is working correctly!"
    else
        print_error "Agent test failed!"
        TEST_FAILED=true
    fi
    
    # Open interfaces only if everything is working
    if [[ "$TEST_FAILED" == "false" ]]; then
        # Open agent interface
        print_status "Opening agent interface..."
        if command -v open >/dev/null 2>&1; then
            open "http://127.0.0.1:8765" 2>/dev/null || print_warning "Could not open agent interface"
        elif command -v xdg-open >/dev/null 2>&1; then
            xdg-open "http://127.0.0.1:8765" 2>/dev/null || print_warning "Could not open agent interface"
        fi
        
        # Launch React frontend
        print_status "Launching React frontend..."
        if command -v open >/dev/null 2>&1; then
            open "http://127.0.0.1:18081/dashboard" 2>/dev/null || print_warning "Could not open dashboard"
        elif command -v xdg-open >/dev/null 2>&1; then
            xdg-open "http://127.0.0.1:18081/dashboard" 2>/dev/null || print_warning "Could not open dashboard"
        fi
    fi
else
    print_error "❌ Tests failed! Not opening interfaces."
    print_error "Please check the test results and fix issues before proceeding."
fi

# 10. Show final status
echo ""
if [[ "$TEST_FAILED" == "false" ]]; then
    print_success "🚀 Deployment completed successfully with all tests passing!"
    echo ""
    echo "🌐 Available Services:"
    echo "   • Agent Interface: http://127.0.0.1:8765"
    echo "   • Dashboard: http://127.0.0.1:18081/dashboard"
    echo "   • InferMesh: http://127.0.0.1:19000/health"
    echo "   • Spark UI: http://127.0.0.1:4041"
    echo "   • Embed Worker: http://127.0.0.1:9101/metrics"
    echo ""
    echo "📊 Test Results:"
    echo "   • Results file: $TEST_RESULTS_FILE"
    echo "   • All tests passed ✅"
    echo "   • Agent learning enabled 🤖"
else
    print_error "❌ Deployment completed with test failures!"
    echo ""
    echo "🔍 Debugging Information:"
    echo "   • Test results: $TEST_RESULTS_FILE"
    echo "   • Container logs: make stack-logs"
    echo "   • Health check: make health-check"
    echo "   • Container status: make stack-ps"
    echo ""
    echo "🔧 Management:"
    echo "   • Stop stack: make stack-down"
    echo "   • Restart agent: make stack-reload"
    echo "   • Re-run tests: ./scripts/enhanced_deploy_with_tests.sh"
fi

echo ""
echo "📋 Management Commands:"
echo "   • View logs: make stack-logs"
echo "   • Health check: make health-check"
echo "   • Container status: make stack-ps"
echo "   • Stop stack: make stack-down"
echo "   • Restart agent: make stack-reload"
echo ""

# Exit with appropriate code
if [[ "$TEST_FAILED" == "false" ]]; then
    exit 0
else
    exit 1
fi
