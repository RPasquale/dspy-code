#!/bin/bash
# Quick start script for DSPy Agent
# This script provides an easy way to get started with the agent

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 DSPy Agent Quick Start${NC}"
echo "=========================="
echo ""

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

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install it first:"
    echo "  pip install uv"
    exit 1
fi

print_status "uv is installed"

# Create virtual environment if it doesn't exist
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    uv venv
    print_status "Virtual environment created"
fi

# Install dependencies
echo "Installing dependencies..."
uv sync
print_status "Dependencies installed"

# Install the package in the virtual environment
echo "Installing package..."
uv pip install -e .
print_status "Package installed"

# Check if Ollama is available
if command -v ollama &> /dev/null; then
    print_status "Ollama is available"
    
    # Check if deepseek-coder:1.3b model is available
    if ollama list | grep -q "deepseek-coder:1.3b"; then
        print_status "deepseek-coder:1.3b model is available"
    else
        print_warning "deepseek-coder:1.3b model not found. You can pull it with:"
        echo "  ollama pull deepseek-coder:1.3b"
    fi
else
    print_warning "Ollama not installed. You can install it from: https://ollama.com/download"
    print_warning "Or use OpenAI-compatible endpoints by setting environment variables"
fi

# Create logs directory
mkdir -p logs
print_status "Logs directory created"

echo ""
echo "🎯 DSPy Agent is ready to use!"
echo ""
echo "Quick commands:"
echo "  uv run dspy-agent --workspace \$(pwd)                    # Start interactive session"
echo "  uv run dspy-agent --help                                 # Show help"
echo "  uv run python scripts/run_all_tests.py                   # Run all tests"
echo "  ./scripts/deploy-dev.sh                                  # Development setup"
echo "  ./scripts/deploy-test.sh                                 # Test environment"
echo "  ./scripts/deploy-prod.sh                                 # Production deployment"
echo ""

# Ask if user wants to start the agent
read -p "Start the agent now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting DSPy Agent..."
    echo ""
    uv run dspy-agent --workspace "$(pwd)"
else
    echo "You can start the agent later with:"
    echo "  uv run dspy-agent --workspace \$(pwd)"
fi
