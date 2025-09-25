#!/bin/bash
set -euo pipefail

# Test Suite Runner Script
# Runs comprehensive tests and feeds results back to agent

echo "🧪 Starting Test Suite Runner..."

# Change to project root
cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_test() {
    echo -e "${PURPLE}[test-suite]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[test-suite]${NC} $1"
}

print_error() {
    echo -e "${RED}[test-suite]${NC} $1"
}

# Test results file
TEST_RESULTS_FILE="/tmp/test_results_$(date +%Y%m%d_%H%M%S).json"

# 1. Run test suite service
print_test "Running comprehensive test suite..."
if docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env run --rm test-suite > "$TEST_RESULTS_FILE" 2>&1; then
    print_success "✅ Test suite completed successfully!"
    TEST_FAILED=false
else
    print_error "❌ Test suite failed!"
    TEST_FAILED=true
fi

# 2. Display results
if [[ -f "$TEST_RESULTS_FILE" ]]; then
    print_test "📊 Test Results:"
    echo "----------------------------------------"
    cat "$TEST_RESULTS_FILE"
    echo "----------------------------------------"
fi

# 3. Feed results to agent
if [[ -f "$TEST_RESULTS_FILE" ]]; then
    print_test "🤖 Feeding test results to agent for learning..."
    
    # Copy results to agent container
    if docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env ps dspy-agent | grep -q "Up"; then
        docker cp "$TEST_RESULTS_FILE" "$(docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env ps -q dspy-agent):/tmp/test_results.json"
        print_success "Test results fed to agent learning system"
    else
        print_error "Agent not available for learning integration"
    fi
fi

# 4. Exit with appropriate code
if [[ "$TEST_FAILED" == "false" ]]; then
    print_success "🎉 All tests passed!"
    exit 0
else
    print_error "❌ Some tests failed!"
    exit 1
fi
