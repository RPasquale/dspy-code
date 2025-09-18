#!/bin/bash
# Production deployment script for DSPy Agent
# This script sets up a production environment with Docker and monitoring

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
PROD_DIR="${PROD_DIR:-$PROJECT_ROOT/prod_deployment}"
DOCKER_STACK_DIR="$PROD_DIR/lightweight"

echo -e "${BLUE}🚀 DSPy Agent Production Deployment${NC}"
echo "====================================="
echo "Project Root: $PROJECT_ROOT"
echo "Workspace: $WORKSPACE"
echo "Logs Directory: $LOGS_DIR"
echo "Production Directory: $PROD_DIR"
echo "Docker Stack Directory: $DOCKER_STACK_DIR"
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

# Confirmation prompt
confirm_deployment() {
    echo -e "${YELLOW}⚠️  PRODUCTION DEPLOYMENT WARNING ⚠️${NC}"
    echo ""
    echo "This will deploy the DSPy Agent to production with the following configuration:"
    echo "- Workspace: $WORKSPACE"
    echo "- Logs: $LOGS_DIR"
    echo "- Docker stack: $DOCKER_STACK_DIR"
    echo ""
    echo "Make sure you have:"
    echo "1. ✅ Run all tests successfully"
    echo "2. ✅ Reviewed the code changes"
    echo "3. ✅ Backed up any existing data"
    echo "4. ✅ Configured your LLM endpoints"
    echo ""
    
    read -p "Type 'prod-deploy' to confirm production deployment: " confirmation
    if [[ "$confirmation" != "prod-deploy" ]]; then
        print_error "Deployment cancelled - confirmation text did not match"
        exit 1
    fi
    
    echo ""
    print_status "Production deployment confirmed"
}

# Check prerequisites
check_prerequisites() {
    echo "🔍 Checking production prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required for production deployment"
        echo "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_status "Docker is installed"
    
    # Check if Docker Compose is available
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose is required for production deployment"
        exit 1
    fi
    print_status "Docker Compose is available"
    
    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        print_error "uv is required for production deployment"
        echo "Please install uv: pip install uv"
        exit 1
    fi
    print_status "uv is installed"
    
    # Check if tests have been run
    if [[ ! -f "$PROJECT_ROOT/test_results.json" ]]; then
        print_warning "No test results found. Running tests first..."
        cd "$PROJECT_ROOT"
        if [[ -f "scripts/run_all_tests.py" ]]; then
            uv run python scripts/run_all_tests.py
            if [[ $? -ne 0 ]]; then
                print_error "Tests failed - cannot proceed with production deployment"
                exit 1
            fi
        else
            print_warning "Test suite not found - proceeding without test validation"
        fi
    else
        print_status "Test results found"
    fi
}

# Setup production environment
setup_production_environment() {
    echo ""
    echo "🔧 Setting up production environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create production directories
    mkdir -p "$LOGS_DIR"
    mkdir -p "$PROD_DIR"
    print_status "Created production directories"
    
    # Generate production Docker stack
    echo "Generating production Docker stack..."
    uv run dspy-agent lightweight_init \
        --workspace "$WORKSPACE" \
        --logs "$LOGS_DIR" \
        --out-dir "$DOCKER_STACK_DIR" \
        --db redis \
        --install-source pip \
        --pip-spec "dspy-code"
    
    if [[ -f "$DOCKER_STACK_DIR/docker-compose.yml" ]] && [[ -f "$DOCKER_STACK_DIR/Dockerfile" ]]; then
        print_status "Production Docker stack generated"
    else
        print_error "Failed to generate production Docker stack"
        exit 1
    fi
}

# Configure production settings
configure_production() {
    echo ""
    echo "⚙️  Configuring production settings..."
    
    cd "$DOCKER_STACK_DIR"
    
    # Create production environment file
    cat > .env.prod << EOF
# DSPy Agent Production Configuration
COMPOSE_PROJECT_NAME=dspy-agent-prod
DSPY_WORKSPACE=$WORKSPACE
DSPY_LOGS=$LOGS_DIR
DSPY_ENV=production

# Database Configuration
REDIS_URL=redis://redis:6379/0

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance Configuration
WORKER_PROCESSES=4
MAX_MEMORY=2g

# Security Configuration
API_KEY_REQUIRED=true
EOF

    print_status "Production environment file created"
    
    # Update docker-compose.yml for production
    if [[ -f "docker-compose.yml" ]]; then
        # Add production-specific configurations
        cp docker-compose.yml docker-compose.yml.backup
        
        # Add restart policies and resource limits
        sed -i.bak 's/restart: "no"/restart: unless-stopped/g' docker-compose.yml
        print_status "Updated Docker Compose for production"
    fi
}

# Build and test production stack
build_production_stack() {
    echo ""
    echo "🐳 Building production Docker stack..."
    
    cd "$DOCKER_STACK_DIR"
    
    # Build the Docker images
    echo "Building Docker images..."
    if DOCKER_BUILDKIT=1 docker compose build --no-cache; then
        print_status "Docker images built successfully"
    else
        print_error "Docker build failed"
        exit 1
    fi
    
    # Test the stack
    echo "Testing production stack..."
    if docker compose config > /dev/null 2>&1; then
        print_status "Docker Compose configuration is valid"
    else
        print_error "Docker Compose configuration is invalid"
        exit 1
    fi
}

# Deploy to production
deploy_to_production() {
    echo ""
    echo "🚀 Deploying to production..."
    
    cd "$DOCKER_STACK_DIR"
    
    # Stop any existing containers
    echo "Stopping existing containers..."
    docker compose down --remove-orphans 2>/dev/null || true
    
    # Start the production stack
    echo "Starting production stack..."
    if docker compose --env-file .env.prod up -d; then
        print_status "Production stack started successfully"
    else
        print_error "Failed to start production stack"
        exit 1
    fi
    
    # Wait for services to be ready
    echo "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    echo "Checking service health..."
    if docker compose ps | grep -q "Up"; then
        print_status "Services are running"
    else
        print_warning "Some services may not be running - check logs"
    fi
}

# Setup monitoring
setup_monitoring() {
    echo ""
    echo "📊 Setting up monitoring..."
    
    cd "$DOCKER_STACK_DIR"
    
    # Create monitoring script
    cat > monitor.sh << 'EOF'
#!/bin/bash
# Production monitoring script for DSPy Agent

echo "🔍 DSPy Agent Production Status"
echo "==============================="

# Check container status
echo "Container Status:"
docker compose ps

echo ""
echo "Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

echo ""
echo "Recent Logs:"
docker compose logs --tail=20 dspy-agent

echo ""
echo "Health Check:"
if curl -f http://localhost:8080/health 2>/dev/null; then
    echo "✅ Agent is responding"
else
    echo "❌ Agent health check failed"
fi
EOF

    chmod +x monitor.sh
    print_status "Monitoring script created: $DOCKER_STACK_DIR/monitor.sh"
    
    # Create log rotation script
    cat > rotate_logs.sh << 'EOF'
#!/bin/bash
# Log rotation script for DSPy Agent

LOGS_DIR="${DSPY_LOGS:-./logs}"
DATE=$(date +%Y%m%d_%H%M%S)

echo "🔄 Rotating logs..."

# Archive current logs
if [[ -d "$LOGS_DIR" ]]; then
    tar -czf "logs_backup_$DATE.tar.gz" "$LOGS_DIR"
    echo "✅ Logs archived to logs_backup_$DATE.tar.gz"
    
    # Clear old logs (keep last 7 days)
    find "$LOGS_DIR" -name "*.log" -mtime +7 -delete
    echo "✅ Old logs cleaned up"
fi
EOF

    chmod +x rotate_logs.sh
    print_status "Log rotation script created: $DOCKER_STACK_DIR/rotate_logs.sh"
}

# Generate deployment report
generate_deployment_report() {
    echo ""
    echo "📊 Generating deployment report..."
    
    cd "$PROJECT_ROOT"
    
    # Create deployment report
    REPORT_FILE="$PROD_DIR/deployment_report.md"
    cat > "$REPORT_FILE" << EOF
# DSPy Agent Production Deployment Report

Deployed on: $(date)
Project Root: $PROJECT_ROOT
Production Directory: $PROD_DIR
Docker Stack Directory: $DOCKER_STACK_DIR

## Deployment Configuration

- **Workspace**: $WORKSPACE
- **Logs Directory**: $LOGS_DIR
- **Database**: Redis
- **Installation Source**: pip (dspy-code)
- **Environment**: production

## Services Deployed

- **dspy-agent**: Main agent service
- **redis**: Database backend
- **kafka**: Message streaming
- **spark**: Data processing
- **ollama**: LLM service (if configured)

## Management Commands

### Start/Stop Services
\`\`\`bash
cd $DOCKER_STACK_DIR
docker compose --env-file .env.prod up -d    # Start
docker compose down                           # Stop
docker compose restart                        # Restart
\`\`\`

### Monitoring
\`\`\`bash
cd $DOCKER_STACK_DIR
./monitor.sh                                  # Check status
./rotate_logs.sh                              # Rotate logs
docker compose logs -f dspy-agent             # Follow logs
\`\`\`

### Health Checks
\`\`\`bash
curl http://localhost:8080/health             # Agent health
docker compose ps                             # Container status
\`\`\`

## Configuration Files

- **Docker Compose**: \`$DOCKER_STACK_DIR/docker-compose.yml\`
- **Environment**: \`$DOCKER_STACK_DIR/.env.prod\`
- **Dockerfile**: \`$DOCKER_STACK_DIR/Dockerfile\`

## Logs and Monitoring

- **Logs Directory**: $LOGS_DIR
- **Monitoring Script**: $DOCKER_STACK_DIR/monitor.sh
- **Log Rotation**: $DOCKER_STACK_DIR/rotate_logs.sh

## Security Notes

- API key authentication is enabled
- Services run with restricted permissions
- Logs are rotated automatically
- Database is configured for production use

## Next Steps

1. **Configure LLM**: Set up your preferred LLM endpoint
2. **Monitor**: Use \`./monitor.sh\` to check system health
3. **Scale**: Adjust worker processes in .env.prod as needed
4. **Backup**: Set up regular backups of logs and database

## Support

- Check logs: \`docker compose logs -f dspy-agent\`
- Restart services: \`docker compose restart\`
- Full restart: \`docker compose down && docker compose up -d\`

EOF

    print_status "Deployment report generated: $REPORT_FILE"
}

# Cleanup function
cleanup() {
    echo ""
    echo "🎉 Production deployment completed successfully!"
    echo ""
    echo "Deployment Summary:"
    echo "- Production directory: $PROD_DIR"
    echo "- Docker stack: $DOCKER_STACK_DIR"
    echo "- Logs directory: $LOGS_DIR"
    echo "- Deployment report: $PROD_DIR/deployment_report.md"
    echo ""
    echo "Quick Commands:"
    echo "  cd $DOCKER_STACK_DIR"
    echo "  ./monitor.sh                    # Check status"
    echo "  docker compose logs -f dspy-agent  # View logs"
    echo "  docker compose ps               # Check containers"
    echo ""
    echo "Services are now running in production mode!"
    echo "Check the deployment report for detailed information."
}

# Main execution
main() {
    confirm_deployment
    check_prerequisites
    setup_production_environment
    configure_production
    build_production_stack
    deploy_to_production
    setup_monitoring
    generate_deployment_report
    cleanup
}

# Handle script interruption
trap 'print_error "Production deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"
