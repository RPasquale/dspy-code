#!/usr/bin/env bash
# DSPy Stack Startup Script
# Single, clean script to start the entire DSPy stack

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
STACK_COMPOSE="$PROJECT_ROOT/docker/lightweight/docker-compose.yml"
STACK_ENV="$PROJECT_ROOT/docker/lightweight/.env"
DEFAULT_REDDB_HOST_PORT=8082

PORT_SPECS=(
    "REDDB_HOST_PORT:${DEFAULT_REDDB_HOST_PORT}:RedDB HTTP:auto"
    "OLLAMA_HOST_PORT:11435:Ollama API:auto"
    "DSPY_AGENT_HOST_PORT:8765:DSPy agent:auto"
    "FASTAPI_HOST_PORT:8767:FastAPI backend:auto"
    "DSPY_EMBEDDER_HOST_PORT:18082:DSPy embedder:auto"
    "DASHBOARD_HOST_PORT:18081:Dashboard:auto"
    "EMBED_WORKER_METRICS_HOST_PORT:19101:Embed worker metrics:auto"
    "PROMETHEUS_HOST_PORT:9090:Prometheus:auto"
    "KAFKA_HOST_PORT:9092:Kafka broker:auto"
    "ZOOKEEPER_HOST_PORT:2181:ZooKeeper:auto"
    "REDIS_HOST_PORT:6379:Redis:auto"
    "MESH_HUB_GRPC_HOST_PORT:50051:Mesh hub gRPC:auto"
    "MESH_HUB_METRICS_HOST_PORT:9100:Mesh hub metrics:auto"
    "MESH_WORKER_GRPC_HOST_PORT:50052:Mesh worker gRPC:auto"
    "MESH_WORKER_METRICS_HOST_PORT:9101:Mesh worker metrics:auto"
    "MESH_TRAINER_GRPC_HOST_PORT:50053:Mesh trainer gRPC:auto"
    "MESH_TRAINER_METRICS_HOST_PORT:9102:Mesh trainer metrics:auto"
    "MESH_GATEWAY_GRPC_HOST_PORT:50060:Mesh gateway gRPC:auto"
    "MESH_GATEWAY_METRICS_HOST_PORT:9103:Mesh gateway metrics:auto"
    "INFERMESH_HOST_PORT:19000:InferMesh router:auto"
    "GO_ORCHESTRATOR_HTTP_PORT:9097:Go Orchestrator HTTP:auto"
    "GO_ORCHESTRATOR_GRPC_HOST_PORT:50062:Go Orchestrator gRPC:auto"
    "STREAM_SUPERVISOR_GRPC_HOST_PORT:7000:Stream supervisor gRPC:auto"
    "STREAM_SUPERVISOR_METRICS_HOST_PORT:9098:Stream supervisor metrics:auto"
    "ENV_RUNNER_METRICS_HOST_PORT:8081:Environment runner metrics:auto"
    "SPARK_VECTORIZER_HOST_PORT:4041:Spark vectorizer UI:auto"
)

declare -Ag PORT_VALUES=()

# Function to print status messages
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

print_step() {
    echo -e "${PURPLE}🔧 $1${NC}"
}

print_success() {
    echo -e "${CYAN}🎉 $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to generate secure token
generate_token() {
    if command_exists openssl; then
        openssl rand -hex 32
    else
        python3 -c "import secrets; print(secrets.token_hex(32))"
    fi
}

# Function to check if port is in use
port_in_use() {
    local port="$1"
    if command_exists ss; then
        ss -tuln | grep -q ":$port "
    elif command_exists netstat; then
        netstat -tuln | grep -q ":$port "
    else
        return 1
    fi
}

# Function to find available port
find_available_port() {
    local base_port="$1"
    local port="$base_port"
    
    while port_in_use "$port"; do
        port=$((port + 1))
        if [ $port -gt $((base_port + 100)) ]; then
            print_error "Could not find available port starting from $base_port"
            return 1
        fi
    done
    
    echo "$port"
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name="$1"
    local max_attempts=30
    local attempt=1
    
    print_info "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$service_name.*Up"; then
            print_status "$service_name is ready"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within expected time"
    return 1
}

# Function to validate and resolve host port usage
validate_and_remap_ports() {
    print_step "Validating host port availability..."

    declare -A used_ports=()
    PORT_VALUES=()

    for spec in "${PORT_SPECS[@]}"; do
        IFS=":" read -r var_name default_port service_name mode <<< "$spec"
        local requested_value="${!var_name:-}"
        local base_port="$default_port"
        local resolved_port=""

        if [[ -n "$requested_value" ]]; then
            resolved_port="$requested_value"
            if [[ -n "${used_ports[$resolved_port]:-}" ]]; then
                print_error "Port $resolved_port (${service_name}) conflicts with ${used_ports[$resolved_port]}. Update \$$var_name to use a different port."
                exit 1
            fi
            if port_in_use "$resolved_port"; then
                print_error "Required port $resolved_port (${service_name}) is already in use. Set \$$var_name to a free port and re-run the script."
                exit 1
            fi
        else
            local candidate_port="$base_port"
            while port_in_use "$candidate_port" || [[ -n "${used_ports[$candidate_port]:-}" ]]; do
                if [[ "$mode" == "fixed" ]]; then
                    print_error "Required port $candidate_port (${service_name}) is already in use. Stop the conflicting process or set \$$var_name to a different value before re-running."
                    exit 1
                fi
                candidate_port=$((candidate_port + 1))
                if (( candidate_port > base_port + 100 )); then
                    print_error "Could not find an available port for ${service_name} starting from $base_port."
                    exit 1
                fi
            done
            resolved_port="$candidate_port"
            if (( resolved_port != base_port )); then
                print_warning "${service_name} default port $base_port is busy; using $resolved_port instead (override with \$$var_name to pin a value)."
            fi
        fi

        used_ports["$resolved_port"]="$service_name"
        export "$var_name"="$resolved_port"
        PORT_VALUES["$var_name"]="$resolved_port"
    done

    print_status "Host ports validated."
}

# Main startup function
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    DSPy Stack Startup                        ║"
    echo "║              Clean, Single-Command Solution                  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    
    # Step 1: Prerequisites check
    print_step "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Check required commands
    local missing_commands=()
    
    if ! command_exists docker; then
        missing_commands+=("docker")
    fi
    
    if ! command_exists python3; then
        missing_commands+=("python3")
    fi
    
    if [ ${#missing_commands[@]} -ne 0 ]; then
        print_error "Missing required commands: ${missing_commands[*]}"
        print_info "Please install the missing dependencies and try again"
        exit 1
    fi
    
    print_status "All prerequisites satisfied"
    
    # Step 2: Clean up existing containers
    print_step "Cleaning up existing containers..."
    
    # Stop and remove any existing containers
    docker compose -f "$STACK_COMPOSE" down --remove-orphans 2>/dev/null || true
    
    # Clean up any dangling containers
    docker container prune -f 2>/dev/null || true
    
    print_status "Cleanup completed"
    
    # Step 3: Validate and remap ports
    validate_and_remap_ports
    
    # Step 4: Environment setup
    print_step "Setting up environment..."
    
    # Generate secure tokens if not set
    if [[ -z "$REDDB_ADMIN_TOKEN" ]]; then
        export REDDB_ADMIN_TOKEN=$(generate_token)
        print_info "Generated REDDB_ADMIN_TOKEN"
    fi
    
    # Set environment variables
    export REDDB_URL=http://reddb:8080
    export REDDB_NAMESPACE=dspy
    export REDDB_TOKEN="$REDDB_ADMIN_TOKEN"
    export DB_BACKEND=reddb

    print_status "Environment variables configured"
    
    # Step 5: Generate protobuf files
    print_step "Generating protobuf files..."
    
    if command_exists buf; then
        cd "$PROJECT_ROOT"
        buf generate
        print_status "Protobuf files generated"
    else
        print_warning "buf not found, skipping protobuf generation"
        print_info "Install buf with: go install github.com/bufbuild/buf/cmd/buf@latest"
    fi
    
    # Step 6: Create environment file
    print_step "Creating environment file..."
    
    {
        cat <<EOF
WORKSPACE_DIR=$PROJECT_ROOT
# Port Configuration
EOF
        for spec in "${PORT_SPECS[@]}"; do
            IFS=":" read -r var_name _default_port _description _mode <<< "$spec"
            echo "${var_name}=${PORT_VALUES[$var_name]}"
        done
        cat <<EOF
# RedDB Configuration
REDDB_ADMIN_TOKEN=$REDDB_ADMIN_TOKEN
REDDB_URL=http://reddb:8080
REDDB_NAMESPACE=dspy
REDDB_TOKEN=$REDDB_ADMIN_TOKEN
DB_BACKEND=reddb
# Mesh Configuration
MESH_GRPC_ENDPOINT=http://mesh-hub:50051
MESH_WORKER_ENDPOINT=http://mesh-worker:50052
MESH_NODE_ID=9002
MESH_DOMAIN=default
MESH_DOMAIN_ID=1
MESH_HUB_NODE_ID=9001
MESH_TRAINER_NODE_ID=9003
MESH_GATEWAY_NODE_ID=9010
MESH_SERVICES_JSON=[{"id":9001,"endpoint":"http://mesh-hub:50051","domain":"default","tags":["hub"]},{"id":9002,"endpoint":"http://mesh-worker:50052","domain":"default","tags":["worker","inference"]},{"id":9003,"endpoint":"http://mesh-trainer:50053","domain":"default","tags":["trainer"]},{"id":9010,"endpoint":"http://mesh-gateway:50060","domain":"default","tags":["gateway","edge"]}]
MESH_LISTEN_ADDR=0.0.0.0:7000
MESH_GRPC_LISTEN_ADDR=0.0.0.0:50051
MESH_METRICS_ADDR=0.0.0.0:9100
MESH_EXTRA_ARGS=
EOF
    } > "$STACK_ENV"
    
    print_status "Environment file created: $STACK_ENV"
    
    # Step 7: Build Docker images
    print_step "Building Docker images..."
    
    # Set Docker build context
    export DOCKER_BUILDKIT=1
    
    # Build the stack
    docker compose -f "$STACK_COMPOSE" --env-file "$STACK_ENV" build --parallel
    
    print_status "Docker images built successfully"
    
    # Step 8: Start services
    print_step "Starting services..."
    
    # Start all services at once (Docker Compose handles dependencies)
    docker compose -f "$STACK_COMPOSE" --env-file "$STACK_ENV" up -d
    
    print_status "All services started"
    
    # Step 9: Wait for services to be ready
    print_step "Waiting for services to be ready..."
    
    # Wait for core services
    wait_for_service "lightweight-zookeeper-1"
    wait_for_service "lightweight-redis-1"
    wait_for_service "reddb"
    wait_for_service "lightweight-kafka-1"
    
    # Wait for mesh services
    wait_for_service "lightweight-mesh-hub-1"
    wait_for_service "lightweight-mesh-worker-1"
    wait_for_service "lightweight-mesh-trainer-1"
    wait_for_service "lightweight-mesh-gateway-1"
    
    # Wait for application services
    wait_for_service "lightweight-ollama-1"
    wait_for_service "lightweight-go-orchestrator-1"
    wait_for_service "lightweight-dspy-agent-1"
    wait_for_service "lightweight-dashboard-1"
    
    # Step 10: Health checks
    print_step "Running health checks..."
    
    # Wait a bit for services to fully initialize
    sleep 10
    
    # Test RedDB with authentication (exposed on host port)
    if curl -s -H "Authorization: Bearer $REDDB_ADMIN_TOKEN" http://127.0.0.1:${REDDB_HOST_PORT}/health >/dev/null; then
        print_status "RedDB is healthy and authenticated"
    else
        print_warning "RedDB health check failed"
    fi
    
    # Test Dashboard
    if curl -s http://127.0.0.1:${DASHBOARD_HOST_PORT} >/dev/null; then
        print_status "Dashboard is accessible"
    else
        print_warning "Dashboard health check failed"
    fi
    
    # Test Agent
    if curl -s http://127.0.0.1:${DSPY_AGENT_HOST_PORT}/health >/dev/null; then
        print_status "Agent is healthy"
    else
        print_warning "Agent health check failed"
    fi
    
    # Step 11: Show final status
    print_success "DSPy Stack is running successfully!"
    echo ""
    echo -e "${CYAN}📍 Services:${NC}"
    echo "   RedDB:     http://127.0.0.1:${REDDB_HOST_PORT} (with auth)"
    echo "   Agent:     http://127.0.0.1:${DSPY_AGENT_HOST_PORT}"
    echo "   Dashboard: http://127.0.0.1:${DASHBOARD_HOST_PORT}"
    echo "   InferMesh: http://127.0.0.1:${INFERMESH_HOST_PORT}"
    echo ""
    echo -e "${CYAN}🔧 Configuration:${NC}"
    echo "   RedDB URL:      $REDDB_URL"
    echo "   RedDB Host Port: $REDDB_HOST_PORT"
    echo "   RedDB Namespace: $REDDB_NAMESPACE"
    echo "   RedDB Token:     ${REDDB_ADMIN_TOKEN:0:8}...${REDDB_ADMIN_TOKEN: -8}"
    echo "   DB Backend:      $DB_BACKEND"
    echo ""
    echo -e "${CYAN}🛠️  Management Commands:${NC}"
    echo "   View logs:       docker compose -f $STACK_COMPOSE --env-file $STACK_ENV logs -f"
    echo "   Stop stack:      docker compose -f $STACK_COMPOSE --env-file $STACK_ENV down"
    echo "   Restart:         docker compose -f $STACK_COMPOSE --env-file $STACK_ENV restart"
    echo "   Status:          docker compose -f $STACK_COMPOSE --env-file $STACK_ENV ps"
    echo ""
    echo -e "${CYAN}🧪 Test Commands:${NC}"
    echo "   # Test RedDB ingest:"
    echo "   curl -X POST http://127.0.0.1:${REDDB_HOST_PORT}/api/db/ingest \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -H \"Authorization: Bearer \$REDDB_ADMIN_TOKEN\" \\"
    echo "     -d '{\"kind\":\"document\",\"namespace\":\"dspy\",\"collection\":\"test\",\"id\":\"test1\",\"text\":\"Test document\"}'"
    echo ""
    echo "   # Test RedDB query:"
    echo "   curl -X POST http://127.0.0.1:${REDDB_HOST_PORT}/api/db/query \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -H \"Authorization: Bearer \$REDDB_ADMIN_TOKEN\" \\"
    echo "     -d '{\"mode\":\"auto\",\"namespace\":\"dspy\",\"text\":\"test query\",\"top_k\":3}'"
    echo ""
    echo -e "${YELLOW}⚠️  Security Note: Never commit REDDB_ADMIN_TOKEN to version control!${NC}"
    echo ""
}

# Run main function
main "$@"
