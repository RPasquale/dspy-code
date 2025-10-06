#!/bin/bash
# Development deployment script for DSPy Agent
# This script sets up a development environment with hot reloading and debugging

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-$PROJECT_ROOT}"
LOGS_DIR="${LOGS_DIR:-$WORKSPACE/logs}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

echo -e "${BLUE}🚀 DSPy Agent Development Deployment${NC}"
echo "=================================="
echo "Project Root: $PROJECT_ROOT"
echo "Workspace: $WORKSPACE"
echo "Logs Directory: $LOGS_DIR"
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

# Check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install it first:"
        echo "  pip install uv"
        exit 1
    fi
    print_status "uv is installed"
    
    # Check if Python 3.11+ is available
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
        print_error "Python 3.11+ is required"
        exit 1
    fi
    print_status "Python 3.11+ is available"
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        print_error "pyproject.toml not found. Are you in the right directory?"
        exit 1
    fi
    print_status "Project structure looks good"
}

# Setup development environment
setup_dev_environment() {
    echo ""
    echo "🔧 Setting up development environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create logs directory
    mkdir -p "$LOGS_DIR"
    print_status "Created logs directory: $LOGS_DIR"
    
    # Install dependencies with uv
    echo "Installing dependencies..."
    uv sync --dev
    print_status "Dependencies installed"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d ".venv" ]]; then
        echo "Creating virtual environment..."
        uv venv
        print_status "Virtual environment created"
    fi

    # Install the package in development mode
    echo "Installing package in development mode..."
    uv pip install -e .
    print_status "Package installed in development mode"
}

# Setup Ollama (optional)
setup_ollama() {
    echo ""
    echo "🤖 Setting up Ollama (optional)..."
    
    if command -v ollama &> /dev/null; then
        print_status "Ollama is already installed"
        
        # Check if deepseek-coder:1.3b model is available
        if ollama list | grep -q "deepseek-coder:1.3b"; then
            print_status "deepseek-coder:1.3b model is available"
        else
            print_warning "deepseek-coder:1.3b model not found. Pulling it..."
            ollama pull deepseek-coder:1.3b
            print_status "deepseek-coder:1.3b model pulled"
        fi
    else
        print_warning "Ollama not installed. You can install it from: https://ollama.com/download"
        print_warning "Or use OpenAI-compatible endpoints by setting environment variables:"
        echo "  export OPENAI_API_KEY=your_key"
        echo "  export OPENAI_BASE_URL=your_endpoint"
        echo "  export MODEL_NAME=your_model"
    fi
}

# Run development tests
run_dev_tests() {
    echo ""
    echo "🧪 Running development tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run the comprehensive test suite
    if [[ -f "scripts/run_all_tests.py" ]]; then
        echo "Running comprehensive test suite..."
        uv run python scripts/run_all_tests.py
        print_status "Test suite completed"
    else
        print_warning "Comprehensive test suite not found, running basic tests..."
        uv run python -m unittest discover -s tests -v
        print_status "Basic tests completed"
    fi
}

# Start development server
start_dev_server() {
    echo ""
    echo "🚀 Starting development server..."
    
    cd "$PROJECT_ROOT"
    
    # Set development environment variables
    export DSPY_DEV_MODE=true
    export DSPY_LOG_LEVEL=DEBUG
    export DSPY_WORKSPACE="$WORKSPACE"
    export DSPY_LOGS="$LOGS_DIR"
    
    print_status "Environment variables set for development"
    
    echo ""
    echo "🎯 Development environment is ready!"
    echo ""
    echo "Available commands:"
    echo "  dspy-agent --workspace $WORKSPACE                    # Start interactive session"
    echo "  dspy-agent --workspace $WORKSPACE --help             # Show help"
    echo "  dspy-agent lightweight_init --workspace $WORKSPACE   # Generate Docker stack"
    echo ""
    echo "Environment variables:"
    echo "  DSPY_DEV_MODE=true"
    echo "  DSPY_LOG_LEVEL=DEBUG"
    echo "  DSPY_WORKSPACE=$WORKSPACE"
    echo "  DSPY_LOGS=$LOGS_DIR"
    echo ""
    echo "Logs will be written to: $LOGS_DIR"
    echo ""
    
    # Ask if user wants to start the agent immediately
    read -p "Start the agent now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting DSPy Agent..."
        uv run dspy-agent --workspace "$WORKSPACE"
    else
        echo "You can start the agent later with:"
        echo "  cd $PROJECT_ROOT"
        echo "  uv run dspy-agent --workspace $WORKSPACE"
    fi
}

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Development deployment completed!"
    echo ""
    echo "Next steps:"
    echo "1. Start coding with: uv run dspy-agent --workspace $WORKSPACE"
    echo "2. Check logs in: $LOGS_DIR"
    echo "3. Run tests with: uv run python scripts/run_all_tests.py"
    echo "4. Generate Docker stack with: uv run dspy-agent lightweight_init"
}

# Main execution
main() {
    check_prerequisites
    setup_dev_environment
    setup_ollama
    run_dev_tests
    start_dev_server
    cleanup
}

# Handle script interruption
trap 'print_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"
