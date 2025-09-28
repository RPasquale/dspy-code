#!/usr/bin/env bash
set -euo pipefail

# Stop DSPy Agent Local System
# This script stops all running services

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT_DIR/logs"
PIDS_DIR="$LOG_DIR/pids"

echo "🛑 Stopping DSPy Agent System"
echo "============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Stop service by PID file
stop_service() {
    local service_name="$1"
    local pid_file="$PIDS_DIR/$service_name.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping $service_name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                log_warning "Force killing $service_name..."
                kill -9 "$pid" 2>/dev/null || true
            fi
            log_success "$service_name stopped"
        else
            log_warning "$service_name was not running"
        fi
        rm -f "$pid_file"
    else
        log_warning "No PID file found for $service_name"
    fi
}

# Stop service by process name
stop_service_by_name() {
    local service_name="$1"
    local process_pattern="$2"
    
    if pgrep -f "$process_pattern" >/dev/null 2>&1; then
        log_info "Stopping $service_name..."
        pkill -f "$process_pattern" 2>/dev/null || true
        sleep 2
        if pgrep -f "$process_pattern" >/dev/null 2>&1; then
            log_warning "Force killing $service_name..."
            pkill -9 -f "$process_pattern" 2>/dev/null || true
        fi
        log_success "$service_name stopped"
    else
        log_warning "$service_name was not running"
    fi
}

# Stop all services
stop_all_services() {
    log_info "Stopping all services..."
    
    # Stop Go orchestrator
    stop_service "orchestrator"
    stop_service_by_name "orchestrator" "/logs/orchestrator"
    
    # Stop Rust env-runner
    stop_service "env-runner"
    stop_service_by_name "env-runner" "env-runner"
    
    # Stop dashboard
    stop_service "dashboard"
    stop_service_by_name "dashboard" "enhanced_dashboard_server.py"
    
    # Stop infrastructure services
    stop_service "redis"
    stop_service_by_name "redis" "redis-server"
    
    stop_service "kafka"
    stop_service_by_name "kafka" "kafka.Kafka"
    
    stop_service "zookeeper"
    stop_service_by_name "zookeeper" "zookeeper"
    
    stop_service "reddb"
    stop_service_by_name "reddb" "reddb"
    
    stop_service "infermesh"
    stop_service_by_name "infermesh" "infermesh"
    
    # Stop any remaining Python processes
    if pgrep -f "python.*dspy" >/dev/null 2>&1; then
        log_info "Stopping remaining Python processes..."
        pkill -f "python.*dspy" 2>/dev/null || true
    fi
    
    # Stop any remaining Go processes
    if pgrep -f "orchestrator" >/dev/null 2>&1; then
        log_info "Stopping remaining Go processes..."
        pkill -f "orchestrator" 2>/dev/null || true
    fi
    
    # Stop any remaining Rust processes
    if pgrep -f "env-runner" >/dev/null 2>&1; then
        log_info "Stopping remaining Rust processes..."
        pkill -f "env-runner" 2>/dev/null || true
    fi
}

# Clean up temporary files
cleanup_temp_files() {
    log_info "Cleaning up temporary files..."
    
    # Remove PID files
    rm -rf "$PIDS_DIR"
    
    # Clean up Go cache
    if [ -d "$ROOT_DIR/.gocache" ]; then
        rm -rf "$ROOT_DIR/.gocache"
    fi
    
    if [ -d "$ROOT_DIR/.gomodcache" ]; then
        rm -rf "$ROOT_DIR/.gomodcache"
    fi
    
    # Clean up Rust target (optional)
    if [ -d "$ROOT_DIR/env_runner_rs/target" ]; then
        log_info "Cleaning up Rust target directory..."
        # Uncomment the next line if you want to clean Rust build artifacts
        # rm -rf "$ROOT_DIR/env_runner_rs/target"
    fi
    
    log_success "Temporary files cleaned up"
}

# Check if services are still running
check_remaining_services() {
    log_info "Checking for remaining services..."
    
    local remaining=()
    
    if pgrep -f "enhanced_dashboard_server.py" >/dev/null 2>&1; then
        remaining+=("dashboard")
    fi
    
    if pgrep -f "/logs/orchestrator" >/dev/null 2>&1; then
        remaining+=("orchestrator")
    fi
    
    if pgrep -f "env-runner" >/dev/null 2>&1; then
        remaining+=("env-runner")
    fi
    
    if pgrep -f "redis-server" >/dev/null 2>&1; then
        remaining+=("redis")
    fi
    
    if [ ${#remaining[@]} -gt 0 ]; then
        log_warning "Some services are still running: ${remaining[*]}"
        log_info "You may need to stop them manually:"
        for service in "${remaining[@]}"; do
            case $service in
                "dashboard")
                    echo "  pkill -f 'enhanced_dashboard_server.py'"
                    ;;
                "orchestrator")
                    echo "  pkill -f '/logs/orchestrator'"
                    ;;
                "env-runner")
                    echo "  pkill -f 'env-runner'"
                    ;;
                "redis")
                    echo "  pkill -f 'redis-server'"
                    ;;
            esac
        done
    else
        log_success "All services stopped successfully"
    fi
}

# Main stop function
main() {
    log_info "Starting system shutdown..."
    
    # Stop all services
    stop_all_services
    
    # Wait a moment for processes to stop
    sleep 2
    
    # Check for remaining services
    check_remaining_services
    
    # Clean up if requested
    if [ "${1:-}" = "--cleanup" ]; then
        cleanup_temp_files
    fi
    
    log_success "System shutdown complete!"
    echo ""
    echo "All DSPy Agent services have been stopped."
    echo ""
    echo "To start the system again:"
    echo "  bash scripts/start_local_system.sh"
    echo ""
    echo "To clean up temporary files:"
    echo "  bash scripts/stop_local_system.sh --cleanup"
    echo ""
}

# Run main function
main "$@"