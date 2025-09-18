#!/bin/bash
# Easy test runner for DSPy Agent
# Usage: ./scripts/test.sh [category]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Function to show help
show_help() {
    echo "DSPy Agent Test Runner"
    echo "====================="
    echo ""
    echo "Usage: $0 [category]"
    echo ""
    echo "Categories:"
    echo "  all         - Run all tests (comprehensive suite)"
    echo "  simple      - Run simple functionality test"
    echo "  unit        - Run unit tests only"
    echo "  rl          - Run RL component tests"
    echo "  integration - Run integration tests"
    echo "  quick       - Run quick validation tests"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 all       # Run comprehensive test suite"
    echo "  $0 simple    # Quick functionality check"
    echo "  $0 rl        # Test RL components"
    echo "  $0 unit      # Unit tests only"
}

# Function to run simple tests
run_simple() {
    print_info "Running simple functionality test..."
    uv run python scripts/test_agent_simple.py
}

# Function to run unit tests
run_unit() {
    print_info "Running unit tests..."
    uv run pytest tests/ -v
}

# Function to run RL tests
run_rl() {
    print_info "Running RL component tests..."
    if [[ -f "scripts/test_rl.py" ]]; then
        uv run python scripts/test_rl.py
    else
        print_warning "RL test script not found, running RL unit tests..."
        uv run pytest tests/test_rl_tooling.py -v
    fi
}

# Function to run integration tests
run_integration() {
    print_info "Running integration tests..."
    uv run python scripts/run_all_tests.py
}

# Function to run all tests
run_all() {
    print_info "Running comprehensive test suite..."
    uv run python scripts/run_all_tests.py
}

# Function to run quick tests
run_quick() {
    print_info "Running quick validation tests..."
    echo ""
    print_info "1. Simple functionality test..."
    uv run python scripts/test_agent_simple.py
    echo ""
    print_info "2. Unit tests..."
    uv run pytest tests/ -x --tb=short
    echo ""
    print_info "3. RL basic test..."
    uv run pytest tests/test_rl_tooling.py -x --tb=short
}

# Main logic
case "${1:-all}" in
    "all")
        run_all
        ;;
    "simple")
        run_simple
        ;;
    "unit")
        run_unit
        ;;
    "rl")
        run_rl
        ;;
    "integration")
        run_integration
        ;;
    "quick")
        run_quick
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Unknown test category: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

print_status "Test run completed!"
