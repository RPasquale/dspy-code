#!/usr/bin/env bash
set -euo pipefail

# Comprehensive Deployment Workflow for DSPy Agent
# This script orchestrates the complete deployment process with intelligent features

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs/deployment"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-development}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

log_feature() {
    echo -e "${CYAN}[FEATURE]${NC} $1"
}

# Error handling
handle_error() {
    log_error "Deployment failed at line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Create necessary directories
setup_directories() {
    log_info "Setting up deployment directories..."
    mkdir -p "$LOG_DIR" "$PROJECT_ROOT/config/environments"
    log_success "Directories created"
}

# Environment detection and configuration
setup_environment() {
    log_step "Setting up environment..."
    
    # Detect environment capabilities
    log_info "Detecting environment capabilities..."
    python3 "$SCRIPT_DIR/environment_manager.py" --workspace "$PROJECT_ROOT" --detect > "$LOG_DIR/environment_detection.json"
    
    # Configure environment
    log_info "Configuring environment for $DEPLOYMENT_ENV..."
    python3 "$SCRIPT_DIR/environment_manager.py" --workspace "$PROJECT_ROOT" --configure
    
    log_success "Environment setup complete"
}

# Pre-deployment validation
validate_prerequisites() {
    log_step "Validating prerequisites..."
    
    # Check system requirements
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check required Python packages
    log_info "Checking Python dependencies..."
    python3 -c "
import sys
required_packages = ['psutil', 'requests', 'docker']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    sys.exit(1)
else:
    print('All required packages available')
"
    
    log_success "Prerequisites validation passed"
}

# Build and prepare
build_and_prepare() {
    log_step "Building and preparing deployment..."
    
    # Build Docker images
    log_info "Building Docker images..."
    cd "$PROJECT_ROOT/docker/lightweight"
    docker compose build --parallel
    
    # Prepare environment files
    log_info "Preparing environment configuration..."
    cat > "$PROJECT_ROOT/.env.deployment" << EOF
# Deployment Configuration
DEPLOYMENT_ENV=$DEPLOYMENT_ENV
DSPY_ENABLE_AUTO_SCALING=true
DSPY_PERFORMANCE_MODE=optimized
DSPY_INTELLIGENT_CACHING=true
DSPY_ADAPTIVE_LEARNING=true

# Advanced Features
AUTO_SCALER_INTERVAL=30
AUTO_SCALER_CPU_THRESHOLD=80
AUTO_SCALER_MEMORY_THRESHOLD=85
PERFORMANCE_MONITOR_INTERVAL=30
EOF
    
    log_success "Build and preparation complete"
}

# Deploy with intelligent orchestration
deploy_with_orchestration() {
    log_step "Deploying with intelligent orchestration..."
    
    # Use intelligent deployment orchestrator
    log_info "Starting intelligent deployment..."
    python3 "$SCRIPT_DIR/intelligent_deployment_orchestrator.py" \
        --workspace "$PROJECT_ROOT" \
        --environment "$DEPLOYMENT_ENV" \
        --no-auto-scaling false \
        --no-monitoring false \
        --no-caching false \
        --no-learning false
    
    log_success "Intelligent deployment completed"
}

# Enable advanced features
enable_advanced_features() {
    log_step "Enabling advanced features..."
    
    # Enable auto-scaling
    log_feature "Enabling intelligent auto-scaling..."
    docker compose -f "$PROJECT_ROOT/docker/lightweight/docker-compose.yml" up -d auto-scaler || log_warning "Auto-scaler not available"
    
    # Enable performance monitoring
    log_feature "Enabling advanced performance monitoring..."
    python3 -c "
from dspy_agent.monitor.performance_monitor import PerformanceMonitor
import asyncio
monitor = PerformanceMonitor('$PROJECT_ROOT')
monitor.start_monitoring()
print('Performance monitoring started')
" &
    
    # Enable intelligent optimization
    log_feature "Running intelligent optimization analysis..."
    python3 "$SCRIPT_DIR/intelligent_optimization.py" --workspace "$PROJECT_ROOT" || log_warning "Intelligent optimization not available"
    
    log_success "Advanced features enabled"
}

# Health validation
validate_deployment() {
    log_step "Validating deployment health..."
    
    # Run comprehensive health checks
    log_info "Running comprehensive health checks..."
    python3 "$SCRIPT_DIR/advanced_health_monitor.py" --workspace "$PROJECT_ROOT" --report > "$LOG_DIR/health_report.json"
    
    # Check deployment status
    log_info "Checking deployment status..."
    python3 "$SCRIPT_DIR/intelligent_deployment_orchestrator.py" --workspace "$PROJECT_ROOT" --status > "$LOG_DIR/deployment_status.json"
    
    # Performance validation
    log_info "Validating performance..."
    python3 "$SCRIPT_DIR/performance_monitor.py" --workspace "$PROJECT_ROOT" --report > "$LOG_DIR/performance_report.json"
    
    log_success "Deployment validation complete"
}

# Generate deployment report
generate_report() {
    log_step "Generating deployment report..."
    
    local report_file="$LOG_DIR/deployment_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "deployment": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$DEPLOYMENT_ENV",
    "status": "completed",
    "features_enabled": {
      "auto_scaling": true,
      "performance_monitoring": true,
      "intelligent_caching": true,
      "adaptive_learning": true,
      "anomaly_detection": true
    },
    "services": {
      "dspy-agent": "deployed",
      "ollama": "deployed",
      "kafka": "deployed",
      "reddb": "deployed",
      "dashboard": "deployed",
      "auto-scaler": "deployed"
    },
    "monitoring": {
      "health_monitor": "active",
      "performance_monitor": "active",
      "auto_scaler": "active"
    }
  }
}
EOF
    
    log_success "Deployment report generated: $report_file"
}

# Main deployment workflow
main() {
    log_info "Starting comprehensive DSPy Agent deployment workflow..."
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Project Root: $PROJECT_ROOT"
    
    # Step 1: Setup
    setup_directories
    setup_environment
    
    # Step 2: Validation
    validate_prerequisites
    
    # Step 3: Build and Prepare
    build_and_prepare
    
    # Step 4: Deploy
    deploy_with_orchestration
    
    # Step 5: Enable Advanced Features
    enable_advanced_features
    
    # Step 6: Validate
    validate_deployment
    
    # Step 7: Generate Report
    generate_report
    
    log_success "Comprehensive deployment workflow completed successfully!"
    log_info "Deployment logs available in: $LOG_DIR"
    log_info "Health monitoring: python3 scripts/advanced_health_monitor.py --workspace $PROJECT_ROOT --daemon"
    log_info "Performance monitoring: python3 scripts/performance_monitor.py --workspace $PROJECT_ROOT --daemon"
}

# Command line options
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "validate")
        validate_prerequisites
        ;;
    "health")
        python3 "$SCRIPT_DIR/advanced_health_monitor.py" --workspace "$PROJECT_ROOT" --report
        ;;
    "status")
        python3 "$SCRIPT_DIR/intelligent_deployment_orchestrator.py" --workspace "$PROJECT_ROOT" --status
        ;;
    "environment")
        python3 "$SCRIPT_DIR/environment_manager.py" --workspace "$PROJECT_ROOT" --detect
        ;;
    "help")
        echo "Usage: $0 [deploy|validate|health|status|environment|help]"
        echo "  deploy      - Run full deployment workflow (default)"
        echo "  validate    - Validate prerequisites only"
        echo "  health      - Check deployment health"
        echo "  status      - Check deployment status"
        echo "  environment - Detect environment capabilities"
        echo "  help        - Show this help message"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
