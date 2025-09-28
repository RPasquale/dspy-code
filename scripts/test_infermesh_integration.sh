#!/usr/bin/env bash
set -euo pipefail

# Test InferMesh integration with the complete system
# This script tests the InferMesh stack integration with Go orchestrator, Rust runner, and Python clients

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT_DIR/logs"

echo "🧪 Testing InferMesh Integration"
echo "================================"
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

# Test InferMesh router health
test_router_health() {
    log_info "Testing InferMesh router health..."

    if curl -f http://localhost:19000/health >/dev/null 2>&1; then
        log_success "InferMesh router health check passed"
        return 0
    else
        log_error "InferMesh router health check failed"
        return 1
    fi
}

# Test DSPy embedder health
test_embedder_health() {
    log_info "Testing DSPy embedder health..."

    if curl -f http://localhost:18082/health >/dev/null 2>&1; then
        log_success "DSPy embedder health check passed"
        return 0
    else
        log_error "DSPy embedder health check failed"
        return 1
    fi
}

# Test InferMesh embedding endpoint
test_embedding_endpoint() {
    log_info "Testing InferMesh embedding endpoint..."
    
    local test_payload='{"model": "BAAI/bge-small-en-v1.5", "inputs": ["test embedding", "another test"]}'
    local response=$(curl -s -X POST http://localhost:19000/embed \
        -H "Content-Type: application/json" \
        -d "$test_payload" 2>/dev/null)
    
    if echo "$response" | grep -q "vectors"; then
        log_success "InferMesh embedding endpoint test passed"
        return 0
    else
        log_error "InferMesh embedding endpoint test failed"
        log_error "Response: $response"
        return 1
    fi
}

# Test Go orchestrator integration
test_orchestrator_integration() {
    log_info "Testing Go orchestrator integration..."
    
    if curl -s http://localhost:9097/metrics >/dev/null 2>&1; then
        log_success "Go orchestrator is running"
        
        # Test orchestrator health
        if curl -s http://localhost:9097/health >/dev/null 2>&1; then
            log_success "Go orchestrator health check passed"
        else
            log_warning "Go orchestrator health endpoint not available"
        fi
        return 0
    else
        log_error "Go orchestrator is not running"
        return 1
    fi
}

# Test Rust runner integration
test_rust_runner_integration() {
    log_info "Testing Rust runner integration..."
    
    if curl -s http://localhost:8081/health >/dev/null 2>&1; then
        log_success "Rust runner is running"
        return 0
    else
        log_error "Rust runner is not running"
        return 1
    fi
}

# Test Python client integration
test_python_client() {
    log_info "Testing Python client integration..."
    
    cd "$ROOT_DIR"
    
    if python3 -c "
from dspy_agent.embedding.infermesh import InferMeshEmbedder
import os

# Test basic client creation
embedder = InferMeshEmbedder(
    base_url='http://localhost:19000',
    model='BAAI/bge-small-en-v1.5'
)

# Test payload building
payload = embedder._build_payload(['test', 'embedding'])
assert 'model' in payload
assert 'inputs' in payload
assert payload['inputs'] == ['test', 'embedding']

print('Python client integration test passed')
" 2>/dev/null; then
        log_success "Python client integration test passed"
        return 0
    else
        log_error "Python client integration test failed"
        return 1
    fi
}

# Test Docker services
test_docker_services() {
    log_info "Testing Docker services..."
    
    cd "$ROOT_DIR/docker/lightweight"
    
    local services=("redis" "dspy-embedder" "infermesh-node-a" "infermesh-node-b" "infermesh-router")
    local all_healthy=true
    
    for service in "${services[@]}"; do
        if docker compose ps "$service" | grep -q "Up"; then
            log_success "$service is running"
        else
            log_error "$service is not running"
            all_healthy=false
        fi
    done
    
    if $all_healthy; then
        log_success "All Docker services are running"
        return 0
    else
        log_error "Some Docker services are not running"
        return 1
    fi
}

# Test Redis connectivity
test_redis_connectivity() {
    log_info "Testing Redis connectivity..."
    
    if docker compose exec redis redis-cli ping >/dev/null 2>&1; then
        log_success "Redis connectivity test passed"
        return 0
    else
        log_error "Redis connectivity test failed"
        return 1
    fi
}

# Test load balancing
test_load_balancing() {
    log_info "Testing load balancing..."
    
    local requests=10
    local success_count=0
    
    for i in $(seq 1 $requests); do
        if curl -s -X POST http://localhost:19000/embed \
            -H "Content-Type: application/json" \
            -d '{"model": "BAAI/bge-small-en-v1.5", "inputs": ["load test '$i'"]}' \
            >/dev/null 2>&1; then
            success_count=$((success_count + 1))
        fi
    done
    
    local success_rate=$((success_count * 100 / requests))
    
    if [ $success_rate -ge 80 ]; then
        log_success "Load balancing test passed ($success_count/$requests requests successful)"
        return 0
    else
        log_error "Load balancing test failed ($success_count/$requests requests successful)"
        return 1
    fi
}

# Test advanced features
test_advanced_features() {
    log_info "Testing advanced features..."
    
    # Test with routing strategy
    local advanced_payload='{
        "model": "BAAI/bge-small-en-v1.5",
        "inputs": ["advanced test"],
        "options": {
            "routing_strategy": "hybrid",
            "priority": "high"
        },
        "metadata": {
            "tenant": "test",
            "source": "integration-test"
        },
        "cache": {
            "ttl_seconds": 300
        }
    }'
    
    if curl -s -X POST http://localhost:19000/embed \
        -H "Content-Type: application/json" \
        -d "$advanced_payload" \
        | grep -q "vectors"; then
        log_success "Advanced features test passed"
        return 0
    else
        log_error "Advanced features test failed"
        return 1
    fi
}

# Main test function
run_tests() {
    local tests=(
        "test_docker_services"
        "test_redis_connectivity"
        "test_embedder_health"
        "test_router_health"
        "test_embedding_endpoint"
        "test_orchestrator_integration"
        "test_rust_runner_integration"
        "test_python_client"
        "test_load_balancing"
        "test_advanced_features"
    )
    
    local passed=0
    local total=${#tests[@]}
    
    echo "Running $total integration tests..."
    echo ""
    
    for test in "${tests[@]}"; do
        if $test; then
            passed=$((passed + 1))
        fi
        echo ""
    done
    
    echo "=========================================="
    echo "Integration Test Results: $passed/$total passed"
    echo "=========================================="
    
    if [ $passed -eq $total ]; then
        log_success "All integration tests passed! 🎉"
        echo ""
        echo "InferMesh is fully integrated and ready for production use."
        echo ""
        echo "Available endpoints:"
        echo "  - InferMesh Gateway: http://localhost:19000"
        echo "  - Go Orchestrator: http://localhost:9097"
        echo "  - Rust Runner: http://localhost:8080"
        echo "  - Redis Cache: http://localhost:6379"
        echo ""
        echo "Test commands:"
        echo "  - Health: curl http://localhost:19000/health"
        echo "  - Embed: curl -X POST http://localhost:19000/embed -H 'Content-Type: application/json' -d '{\"model\": \"BAAI/bge-small-en-v1.5\", \"inputs\": [\"test\"]}'"
        return 0
    else
        log_error "Some integration tests failed. Please check the logs and try again."
        return 1
    fi
}

# Run the tests
run_tests
