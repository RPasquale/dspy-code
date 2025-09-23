#!/usr/bin/env bash
set -euo pipefail

# Automated Agent Setup Script
# This script sets up the complete DSPy agent stack with all dependencies and services

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SETUP_DIR="$PROJECT_ROOT/setup"
LOG_DIR="$PROJECT_ROOT/logs"
DOCKER_DIR="$PROJECT_ROOT/docker/lightweight"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Error handling
handle_error() {
    log_error "Setup failed at line $1"
    cleanup
    exit 1
}

trap 'handle_error $LINENO' ERR

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -rf "$SETUP_DIR" 2>/dev/null || true
}

# Check system requirements
check_system_requirements() {
    log_step "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "Detected macOS"
        PLATFORM="macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_info "Detected Linux"
        PLATFORM="linux"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check Python version (3.11+ required)
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
        log_error "Python 3.11+ is required. Current version: $(python3 --version)"
        log_info "Please install Python 3.11+ and try again"
        exit 1
    fi
    
    log_success "Python version check passed: $(python3 --version)"
    
    # Check available memory
    if [[ "$PLATFORM" == "macos" ]]; then
        MEMORY_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    else
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    fi
    
    if [[ $MEMORY_GB -lt 8 ]]; then
        log_warning "Recommended: 8GB+ RAM. Current: ${MEMORY_GB}GB"
    fi
    
    # Check disk space
    if [[ "$PLATFORM" == "macos" ]]; then
        DISK_GB=$(df -g . | awk 'NR==2 {print $4}')
    else
        DISK_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    fi
    if [[ $DISK_GB -lt 10 ]]; then
        log_warning "Recommended: 10GB+ free disk space. Current: ${DISK_GB}GB"
    fi
    
    log_success "System requirements check completed"
}

# Install system dependencies
install_system_dependencies() {
    log_step "Installing system dependencies..."
    
    if [[ "$PLATFORM" == "macos" ]]; then
        # Check for Homebrew
        if ! command -v brew &> /dev/null; then
            log_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install dependencies
        log_info "Installing macOS dependencies..."
        brew install docker docker-compose git curl wget jq
        
    elif [[ "$PLATFORM" == "linux" ]]; then
        # Detect package manager
        if command -v apt-get &> /dev/null; then
            log_info "Installing Ubuntu/Debian dependencies..."
            sudo apt-get update
            sudo apt-get install -y docker.io docker-compose git curl wget jq build-essential
        elif command -v yum &> /dev/null; then
            log_info "Installing CentOS/RHEL dependencies..."
            sudo yum install -y docker docker-compose git curl wget jq gcc gcc-c++ make
        elif command -v pacman &> /dev/null; then
            log_info "Installing Arch Linux dependencies..."
            sudo pacman -S --noconfirm docker docker-compose git curl wget jq base-devel
        else
            log_error "Unsupported Linux distribution"
            exit 1
        fi
    fi
    
    log_success "System dependencies installed"
}

# Setup Python environment
setup_python_environment() {
    log_step "Setting up Python environment..."
    
    # Create virtual environment
    log_info "Creating Python virtual environment..."
    python3 -m venv "$PROJECT_ROOT/venv"
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    python3 -m pip install --upgrade pip setuptools wheel
    
    # Install uv for faster package management
    log_info "Installing uv package manager..."
    python3 -m pip install uv
    
    # Install project dependencies
    log_info "Installing project dependencies..."
    cd "$PROJECT_ROOT"
    uv pip install -e .
    
    # Install development dependencies
    log_info "Installing development dependencies..."
    uv pip install pytest pytest-html pytest-xdist pytest-cov black flake8 mypy
    
    log_success "Python environment setup completed"
}

# Setup Docker environment
setup_docker_environment() {
    log_step "Setting up Docker environment..."
    
    # Start Docker daemon
    if [[ "$PLATFORM" == "macos" ]]; then
        log_info "Starting Docker Desktop..."
        open -a Docker
        sleep 10
    elif [[ "$PLATFORM" == "linux" ]]; then
        log_info "Starting Docker service..."
        sudo systemctl start docker
        sudo systemctl enable docker
    fi
    
    # Add user to docker group (Linux only)
    if [[ "$PLATFORM" == "linux" ]]; then
        sudo usermod -aG docker "$USER"
        log_warning "Please log out and log back in for Docker group changes to take effect"
    fi
    
    # Test Docker
    log_info "Testing Docker installation..."
    docker --version
    docker-compose --version
    
    log_success "Docker environment setup completed"
}

# Setup Ollama (optional local LLM)
setup_ollama() {
    log_step "Setting up Ollama (optional local LLM)..."
    
    if command -v ollama &> /dev/null; then
        log_info "Ollama already installed"
    else
        log_info "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    
    # Start Ollama service
    log_info "Starting Ollama service..."
    if [[ "$PLATFORM" == "macos" ]]; then
        ollama serve &
    else
        systemctl --user start ollama
    fi
    
    # Wait for Ollama to start
    sleep 5
    
    # Pull default model
    log_info "Pulling default model (qwen3:1.7b)..."
    ollama pull qwen3:1.7b || log_warning "Failed to pull model, continuing without it"
    
    log_success "Ollama setup completed"
}

# Setup workspace
setup_workspace() {
    log_step "Setting up workspace..."
    
    # Create workspace directories
    log_info "Creating workspace directories..."
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/.dspy_cache"
    mkdir -p "$PROJECT_ROOT/vectorized/embeddings"
    mkdir -p "$PROJECT_ROOT/.dspy_checkpoints"
    
    # Set permissions
    log_info "Setting workspace permissions..."
    chmod -R 755 "$PROJECT_ROOT/logs"
    chmod -R 755 "$PROJECT_ROOT/.dspy_cache"
    
    # Create environment file
    log_info "Creating environment configuration..."
    cat > "$PROJECT_ROOT/.env" << EOF
# DSPy Agent Environment Configuration
WORKSPACE_DIR=$PROJECT_ROOT
DSPY_LOG_LEVEL=INFO
DSPY_AUTO_TRAIN=false
USE_OLLAMA=true
OLLAMA_MODEL=qwen3:1.7b
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
REDDB_URL=http://localhost:8080
REDDB_NAMESPACE=dspy
REDDB_TOKEN=development-token
EOF
    
    log_success "Workspace setup completed"
}

# Setup Docker Compose stack
setup_docker_stack() {
    log_step "Setting up Docker Compose stack..."
    
    cd "$DOCKER_DIR"
    
    # Create environment file for Docker Compose
    log_info "Creating Docker Compose environment..."
    cat > .env << EOF
WORKSPACE_DIR=$PROJECT_ROOT
REDDB_URL=http://localhost:8080
REDDB_NAMESPACE=dspy
REDDB_TOKEN=development-token
EOF
    
    # Build Docker images
    log_info "Building Docker images..."
    make stack-build
    
    log_success "Docker stack setup completed"
}

# Run initial tests
run_initial_tests() {
    log_step "Running initial tests..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run basic tests
    log_info "Running basic functionality tests..."
    python3 -m pytest tests/test_comprehensive_agent.py::TestAgentComprehensive::test_agent_launcher_basic_functionality -v
    
    # Test CLI
    log_info "Testing CLI functionality..."
    python3 -m dspy_agent.cli --help
    
    # Test Docker build
    log_info "Testing Docker build..."
    cd "$DOCKER_DIR"
    docker-compose config
    
    log_success "Initial tests completed"
}

# Start services
start_services() {
    log_step "Starting services..."
    
    cd "$DOCKER_DIR"
    
    # Start Docker Compose stack
    log_info "Starting Docker Compose stack..."
    make stack-up
    
    # Wait for services to be ready
    log_info "Waiting for services to start..."
    sleep 60
    
    # Run health checks
    log_info "Running health checks..."
    make health-check || log_warning "Some health checks failed"
    
    log_success "Services started successfully"
}

# Generate setup report
generate_setup_report() {
    log_step "Generating setup report..."
    
    REPORT_FILE="$LOG_DIR/setup_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$REPORT_FILE" << EOF
# DSPy Agent Setup Report

**Date:** $(date)
**Platform:** $PLATFORM
**Python Version:** $(python3 --version)
**Docker Version:** $(docker --version)

## Installation Summary

### System Dependencies
- Docker: $(docker --version)
- Docker Compose: $(docker-compose --version)
- Git: $(git --version)

### Python Environment
- Virtual Environment: $PROJECT_ROOT/venv
- Python Version: $(python3 --version)
- Package Manager: uv

### Services Status
$(docker-compose -f "$DOCKER_DIR/docker-compose.yml" ps 2>/dev/null || echo "Services not running")

### Workspace Structure
\`\`\`
$PROJECT_ROOT/
├── logs/                    # Application logs
├── .dspy_cache/            # Agent cache
├── vectorized/             # Vector embeddings
├── .dspy_checkpoints/     # Training checkpoints
├── venv/                  # Python virtual environment
└── .env                   # Environment configuration
\`\`\`

## Next Steps

1. **Test the installation:**
   \`\`\`bash
   cd $PROJECT_ROOT
   source venv/bin/activate
   python3 -m dspy_agent.cli --help
   \`\`\`

2. **Start the agent:**
   \`\`\`bash
   cd $PROJECT_ROOT
   source venv/bin/activate
   python3 -m dspy_agent.launcher --workspace .
   \`\`\`

3. **Access the dashboard:**
   - Open http://localhost:8765 in your browser

4. **Monitor services:**
   \`\`\`bash
   cd $DOCKER_DIR
   make stack-logs
   \`\`\`

## Troubleshooting

- **Check service health:** \`make health-check\`
- **View logs:** \`make stack-logs\`
- **Restart services:** \`make stack-down && make stack-up\`
- **Clean installation:** \`make clean\`

## Support

- Documentation: $PROJECT_ROOT/docs/
- Issues: https://github.com/RPasquale/dspy_stuff/issues
- Logs: $LOG_DIR/

EOF

    log_success "Setup report generated: $REPORT_FILE"
}

# Main setup function
main() {
    local mode="${1:-full}"
    
    log_info "Starting automated DSPy agent setup..."
    log_info "Mode: $mode"
    
    case "$mode" in
        "setup")
            check_system_requirements
            ;;
        "minimal")
            check_system_requirements
            setup_python_environment
            setup_workspace
            ;;
        "docker")
            check_system_requirements
            install_system_dependencies
            setup_docker_environment
            setup_docker_stack
            ;;
        "full")
            check_system_requirements
            install_system_dependencies
            setup_python_environment
            setup_docker_environment
            setup_ollama
            setup_workspace
            setup_docker_stack
            run_initial_tests
            start_services
            generate_setup_report
            ;;
        "test")
            run_initial_tests
            ;;
        "start")
            start_services
            ;;
        "stop")
            cd "$DOCKER_DIR"
            make stack-down
            ;;
        "clean")
            cleanup
            cd "$DOCKER_DIR"
            make stack-down
            docker system prune -f
            ;;
        *)
            echo "Usage: $0 {setup|minimal|docker|full|test|start|stop|clean}"
            echo ""
            echo "Modes:"
            echo "  setup    - Check system requirements only"
            echo "  minimal  - Basic Python environment only"
            echo "  docker   - Docker environment only"
            echo "  full     - Complete setup with all services"
            echo "  test     - Run initial tests"
            echo "  start    - Start all services"
            echo "  stop     - Stop all services"
            echo "  clean    - Clean up everything"
            exit 1
            ;;
    esac
    
    log_success "Setup completed successfully!"
    
    if [[ "$mode" == "full" ]]; then
        echo ""
        log_info "🎉 DSPy Agent is ready to use!"
        log_info "Run 'source venv/bin/activate && python3 -m dspy_agent.launcher --workspace .' to start"
        log_info "Or visit http://localhost:8765 for the dashboard"
    fi
}

# Run main function with all arguments
main "$@"
