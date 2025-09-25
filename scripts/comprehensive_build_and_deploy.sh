#!/usr/bin/env bash
set -euo pipefail

# Comprehensive Build and Deploy Script for DSPy Agent
# This script handles the complete build, test, and deployment pipeline

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
DIST_DIR="$PROJECT_ROOT/dist"
LOG_DIR="$PROJECT_ROOT/logs"
DOCKER_DIR="$PROJECT_ROOT/docker/lightweight"

# Advanced deployment configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-development}"
ENABLE_AUTO_SCALING="${ENABLE_AUTO_SCALING:-true}"
ENABLE_PERFORMANCE_MONITORING="${ENABLE_PERFORMANCE_MONITORING:-true}"
ENABLE_INTELLIGENT_CACHING="${ENABLE_INTELLIGENT_CACHING:-true}"
ENABLE_ADAPTIVE_LEARNING="${ENABLE_ADAPTIVE_LEARNING:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Error handling
handle_error() {
    log_error "Build failed at line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Advanced deployment functions
setup_advanced_features() {
    log_info "Setting up advanced features..."
    
    # Create environment configuration
    cat > "$PROJECT_ROOT/.env.advanced" << EOF
# Advanced DSPy Agent Configuration
DEPLOYMENT_ENV=$DEPLOYMENT_ENV
ENABLE_AUTO_SCALING=$ENABLE_AUTO_SCALING
ENABLE_PERFORMANCE_MONITORING=$ENABLE_PERFORMANCE_MONITORING
ENABLE_INTELLIGENT_CACHING=$ENABLE_INTELLIGENT_CACHING
ENABLE_ADAPTIVE_LEARNING=$ENABLE_ADAPTIVE_LEARNING

# Performance optimization
DSPY_PERFORMANCE_MODE=optimized
DSPY_INTELLIGENT_CACHING=true
DSPY_ADAPTIVE_LEARNING=true
DSPY_ENABLE_AUTO_SCALING=true

# Monitoring configuration
AUTO_SCALER_INTERVAL=30
AUTO_SCALER_CPU_THRESHOLD=80
AUTO_SCALER_MEMORY_THRESHOLD=85
PERFORMANCE_MONITOR_INTERVAL=30
EOF
    
    log_success "Advanced features configuration created"
}

deploy_with_advanced_features() {
    log_info "Deploying with advanced features enabled..."
    
    # Set environment variables for advanced features
    export DSPY_ENABLE_AUTO_SCALING=$ENABLE_AUTO_SCALING
    export DSPY_PERFORMANCE_MODE=optimized
    export DSPY_INTELLIGENT_CACHING=$ENABLE_INTELLIGENT_CACHING
    export DSPY_ADAPTIVE_LEARNING=$ENABLE_ADAPTIVE_LEARNING
    
    # Deploy the stack
    cd "$DOCKER_DIR"
    docker compose up -d --build
    
    # Start advanced services if enabled
    if [ "$ENABLE_AUTO_SCALING" = "true" ]; then
        log_info "Starting auto-scaler service..."
        docker compose up -d auto-scaler
    fi
    
    if [ "$ENABLE_PERFORMANCE_MONITORING" = "true" ]; then
        log_info "Starting performance monitoring..."
        python3 -c "
from dspy_agent.monitor.performance_monitor import PerformanceMonitor
import asyncio
monitor = PerformanceMonitor('$PROJECT_ROOT')
monitor.start_monitoring()
print('Performance monitoring started')
" &
    fi
    
    log_success "Advanced deployment completed"
}

# Create necessary directories
setup_directories() {
    log_info "Setting up build directories..."
    mkdir -p "$BUILD_DIR" "$DIST_DIR" "$LOG_DIR"
    log_success "Directories created"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python version
    if ! python3 --version | grep -q "Python 3.11"; then
        log_error "Python 3.11 is required"
        exit 1
    fi
    
    # Check required tools
    for tool in docker docker-compose git; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Run comprehensive tests
run_tests() {
    log_info "Running comprehensive test suite..."
    
    # Unit tests
    log_info "Running unit tests..."
    python3 -m pytest tests/ -v --tb=short --maxfail=5 \
        --junitxml="$LOG_DIR/unit_tests.xml" \
        --html="$LOG_DIR/unit_tests.html" \
        --self-contained-html
    
    # Integration tests
    log_info "Running integration tests..."
    python3 -m pytest tests/test_comprehensive_agent.py::TestAgentComprehensive -v \
        --junitxml="$LOG_DIR/integration_tests.xml" \
        --html="$LOG_DIR/integration_tests.html" \
        --self-contained-html
    
    # Docker tests
    log_info "Running Docker integration tests..."
    python3 -m pytest tests/test_comprehensive_agent.py::TestDockerIntegration -v \
        --junitxml="$LOG_DIR/docker_tests.xml"
    
    # Package tests
    log_info "Running package build tests..."
    python3 -m pytest tests/test_comprehensive_agent.py::TestPackageBuild -v \
        --junitxml="$LOG_DIR/package_tests.xml"
    
    log_success "All tests completed"
}

# Build Python package
build_package() {
    log_info "Building Python package..."
    
    # Clean previous builds
    rm -rf "$DIST_DIR"/* "$BUILD_DIR"/*
    
    # Install build dependencies
    log_info "Installing build dependencies..."
    python3 -m pip install --upgrade pip setuptools wheel build twine
    
    # Build package
    log_info "Building wheel and source distribution..."
    python3 -m build --outdir "$DIST_DIR"
    
    # Verify package
    log_info "Verifying package..."
    python3 -m twine check "$DIST_DIR"/*
    
    log_success "Package built successfully"
}

# Build Docker images
build_docker() {
    log_info "Building Docker images..."
    
    cd "$DOCKER_DIR"
    
    # Build lightweight image
    log_info "Building lightweight Docker image..."
    DOCKER_BUILDKIT=1 docker build -t dspy-lightweight:latest .
    
    # Build embed worker image
    log_info "Building embed worker image..."
    DOCKER_BUILDKIT=1 docker build -f embed_worker.Dockerfile -t dspy-embed-worker:latest ../..
    
    # Test Docker images
    log_info "Testing Docker images..."
    docker run --rm dspy-lightweight:latest --help
    docker run --rm dspy-embed-worker:latest --help
    
    log_success "Docker images built successfully"
}

# Run Docker Compose tests
test_docker_compose() {
    log_info "Testing Docker Compose stack..."
    
    cd "$DOCKER_DIR"
    
    # Validate compose file
    log_info "Validating Docker Compose configuration..."
    docker-compose config
    
    # Test stack startup
    log_info "Testing stack startup..."
    docker-compose up -d --remove-orphans
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Run health checks
    log_info "Running health checks..."
    make health-check || log_warning "Some health checks failed"
    
    # Run smoke tests
    log_info "Running smoke tests..."
    make smoke || log_warning "Smoke tests failed"
    
    # Cleanup
    log_info "Cleaning up test stack..."
    docker-compose down
    
    log_success "Docker Compose tests completed"
}

# Deploy to development environment
deploy_dev() {
    log_info "Deploying to development environment..."
    
    cd "$DOCKER_DIR"
    
    # Set development environment
    export ENVIRONMENT=development
    export WORKSPACE_DIR="$PROJECT_ROOT"
    
    # Start development stack
    log_info "Starting development stack..."
    make stack-up
    
    # Wait for services
    log_info "Waiting for services to start..."
    sleep 45
    
    # Run health checks
    log_info "Running health checks..."
    make health-check
    
    log_success "Development deployment completed"
}

# Deploy to test environment
deploy_test() {
    log_info "Deploying to test environment..."
    
    cd "$DOCKER_DIR"
    
    # Set test environment
    export ENVIRONMENT=test
    export WORKSPACE_DIR="$PROJECT_ROOT"
    
    # Start test stack
    log_info "Starting test stack..."
    make stack-up
    
    # Wait for services
    log_info "Waiting for services to start..."
    sleep 45
    
    # Run comprehensive tests
    log_info "Running test environment tests..."
    make test-lightweight
    
    log_success "Test deployment completed"
}

# Deploy to production environment
deploy_prod() {
    log_info "Deploying to production environment..."
    
    # Confirm production deployment
    read -p "Are you sure you want to deploy to production? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        log_warning "Production deployment cancelled"
        return
    fi
    
    cd "$DOCKER_DIR"
    
    # Set production environment
    export ENVIRONMENT=production
    export WORKSPACE_DIR="$PROJECT_ROOT"
    
    # Tag release
    log_info "Tagging release..."
    git tag -a "v$(date +%Y%m%d-%H%M%S)" -m "Production release $(date)"
    
    # Start production stack
    log_info "Starting production stack..."
    make stack-up
    
    # Wait for services
    log_info "Waiting for services to start..."
    sleep 60
    
    # Run health checks
    log_info "Running production health checks..."
    make health-check
    
    log_success "Production deployment completed"
}

# Publish package
publish_package() {
    log_info "Publishing package..."
    
    # Check if package should be published
    read -p "Publish package to PyPI? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        log_warning "Package publishing cancelled"
        return
    fi
    
    # Upload to PyPI
    log_info "Uploading to PyPI..."
    python3 -m twine upload "$DIST_DIR"/*
    
    log_success "Package published successfully"
}

# Generate deployment report
generate_report() {
    log_info "Generating deployment report..."
    
    REPORT_FILE="$LOG_DIR/deployment_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$REPORT_FILE" << EOF
# DSPy Agent Deployment Report

**Date:** $(date)
**Version:** $(git describe --tags --always)
**Environment:** ${ENVIRONMENT:-development}

## Build Information
- **Python Version:** $(python3 --version)
- **Docker Version:** $(docker --version)
- **Git Commit:** $(git rev-parse HEAD)

## Test Results
- **Unit Tests:** $(find "$LOG_DIR" -name "unit_tests.xml" -exec grep -o 'tests="[0-9]*"' {} \; | head -1 || echo "Not available")
- **Integration Tests:** $(find "$LOG_DIR" -name "integration_tests.xml" -exec grep -o 'tests="[0-9]*"' {} \; | head -1 || echo "Not available")
- **Docker Tests:** $(find "$LOG_DIR" -name "docker_tests.xml" -exec grep -o 'tests="[0-9]*"' {} \; | head -1 || echo "Not available")

## Package Information
- **Package Files:** $(ls -la "$DIST_DIR" | wc -l) files
- **Package Size:** $(du -sh "$DIST_DIR" | cut -f1)

## Docker Images
- **dspy-lightweight:** $(docker images dspy-lightweight:latest --format "table {{.Size}}" | tail -1)
- **dspy-embed-worker:** $(docker images dspy-embed-worker:latest --format "table {{.Size}}" | tail -1)

## Services Status
$(docker-compose -f "$DOCKER_DIR/docker-compose.yml" ps 2>/dev/null || echo "Services not running")

## Health Checks
$(make -C "$DOCKER_DIR" health-check 2>&1 || echo "Health checks not available")

## Next Steps
1. Monitor service health
2. Review logs for any issues
3. Test agent functionality
4. Update documentation if needed

EOF

    log_success "Deployment report generated: $REPORT_FILE"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Stop any running containers
    cd "$DOCKER_DIR"
    docker-compose down 2>/dev/null || true
    
    # Remove temporary files
    rm -rf "$BUILD_DIR"/* 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    local action="${1:-all}"
    
    log_info "Starting comprehensive build and deploy process..."
    log_info "Action: $action"
    
    case "$action" in
        "setup")
            setup_directories
            check_prerequisites
            ;;
        "test")
            run_tests
            ;;
        "build")
            build_package
            build_docker
            ;;
        "deploy-dev")
            deploy_dev
            ;;
        "deploy-test")
            deploy_test
            ;;
        "deploy-prod")
            deploy_prod
            ;;
        "publish")
            publish_package
            ;;
        "all")
            setup_directories
            check_prerequisites
            run_tests
            build_package
            build_docker
            test_docker_compose
            deploy_dev
            generate_report
            ;;
        "clean")
            cleanup
            ;;
        *)
            echo "Usage: $0 {setup|test|build|deploy-dev|deploy-test|deploy-prod|publish|all|clean}"
            exit 1
            ;;
    esac
    
    log_success "Operation completed successfully"
}

# Run main function with all arguments
main "$@"
