#!/bin/bash
# Infrastructure Verification Script
# Checks that all Rust/Go binaries and Python modules are ready

set -e

COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_RESET='\033[0m'

echo -e "${COLOR_BLUE}=================================${COLOR_RESET}"
echo -e "${COLOR_BLUE}Infrastructure Verification${COLOR_RESET}"
echo -e "${COLOR_BLUE}=================================${COLOR_RESET}"
echo ""

# Track failures
FAILURES=0

# Function to check file existence
check_file() {
    local file=$1
    local description=$2
    
    if [ -f "$file" ]; then
        echo -e "${COLOR_GREEN}✓${COLOR_RESET} $description: $file"
        return 0
    else
        echo -e "${COLOR_RED}✗${COLOR_RESET} $description: $file (NOT FOUND)"
        FAILURES=$((FAILURES + 1))
        return 1
    fi
}

# Function to check if file is executable
check_executable() {
    local file=$1
    local description=$2
    
    if [ -x "$file" ]; then
        echo -e "${COLOR_GREEN}✓${COLOR_RESET} $description is executable"
        return 0
    else
        echo -e "${COLOR_YELLOW}⚠${COLOR_RESET} $description is not executable (run: chmod +x $file)"
        return 1
    fi
}

# Function to test HTTP endpoint
test_endpoint() {
    local url=$1
    local description=$2
    
    if command -v curl &> /dev/null; then
        if curl -sf "$url" > /dev/null 2>&1; then
            echo -e "${COLOR_GREEN}✓${COLOR_RESET} $description is responding"
            return 0
        else
            echo -e "${COLOR_YELLOW}⚠${COLOR_RESET} $description is not responding (service may not be running)"
            return 1
        fi
    else
        echo -e "${COLOR_YELLOW}⚠${COLOR_RESET} curl not available, skipping endpoint test"
        return 1
    fi
}

echo "1. Checking Go binaries..."
check_file "orchestrator/orchestrator-linux" "Go orchestrator binary"
check_file "cmd/dspy-agent/dspy-agent" "Go dspy-agent CLI"

if [ -f "orchestrator/orchestrator-linux" ]; then
    check_executable "orchestrator/orchestrator-linux" "orchestrator-linux"
fi

if [ -f "cmd/dspy-agent/dspy-agent" ]; then
    check_executable "cmd/dspy-agent/dspy-agent" "dspy-agent"
fi

echo ""
echo "2. Checking Rust env-manager..."
if [ -f "env_manager_rs/target/release/env-manager" ]; then
    check_file "env_manager_rs/target/release/env-manager" "Rust env-manager binary"
    check_executable "env_manager_rs/target/release/env-manager" "env-manager"
else
    echo -e "${COLOR_YELLOW}⚠${COLOR_RESET} Rust env-manager not built yet"
    echo -e "  ${COLOR_YELLOW}→${COLOR_RESET} Run: cd env_manager_rs && cargo build --release"
fi

echo ""
echo "3. Checking Python protobuf stubs..."
check_file "dspy_agent/infra/pb/orchestrator/v1_pb2.py" "Orchestrator protobuf"
check_file "dspy_agent/infra/pb/orchestrator/v1_pb2_grpc.py" "Orchestrator gRPC stub"
check_file "dspy_agent/infra/pb/env_manager/v1_pb2.py" "Env-manager protobuf"
check_file "dspy_agent/infra/pb/env_manager/v1_pb2_grpc.py" "Env-manager gRPC stub"

echo ""
echo "4. Checking Python modules..."
if python3 -c "from dspy_agent.infra import AgentInfra" 2>/dev/null; then
    echo -e "${COLOR_GREEN}✓${COLOR_RESET} AgentInfra can be imported"
else
    echo -e "${COLOR_RED}✗${COLOR_RESET} AgentInfra cannot be imported"
    FAILURES=$((FAILURES + 1))
fi

if python3 -c "from dspy_agent.infra.grpc_client import OrchestratorClient" 2>/dev/null; then
    echo -e "${COLOR_GREEN}✓${COLOR_RESET} OrchestratorClient can be imported"
else
    echo -e "${COLOR_RED}✗${COLOR_RESET} OrchestratorClient cannot be imported"
    FAILURES=$((FAILURES + 1))
fi

echo ""
echo "5. Checking systemd service files..."
check_file "deploy/systemd/env-manager.service" "env-manager systemd service"
check_file "deploy/systemd/orchestrator.service" "orchestrator systemd service"
check_file "deploy/systemd/dspy-agent.target" "dspy-agent systemd target"

echo ""
echo "6. Checking documentation..."
check_file "docs/PYTHON_INTEGRATION_GUIDE.md" "Python Integration Guide"
check_file "docs/QUICK_REFERENCE.md" "Quick Reference"
check_file "docs/LATEST_RUST_GO_UPDATES.md" "Latest Updates"
check_file "INFRASTRUCTURE_IMPLEMENTATION_STATUS.md" "Implementation Status"

echo ""
echo "7. Testing live endpoints (if services are running)..."
test_endpoint "http://localhost:50101/health" "env-manager HTTP health"
test_endpoint "http://localhost:50101/metrics" "env-manager metrics"
test_endpoint "http://localhost:9097/queue/status" "orchestrator HTTP"
test_endpoint "http://localhost:9097/metrics" "orchestrator metrics"

echo ""
echo -e "${COLOR_BLUE}=================================${COLOR_RESET}"
if [ $FAILURES -eq 0 ]; then
    echo -e "${COLOR_GREEN}✓ All critical checks passed!${COLOR_RESET}"
    echo ""
    echo "Next steps:"
    echo "  1. Build Rust env-manager (if not done):"
    echo "     cd env_manager_rs && cargo build --release"
    echo ""
    echo "  2. Test the infrastructure:"
    echo "     cd cmd/dspy-agent && ./dspy-agent start"
    echo ""
    echo "  3. Run Python integration test:"
    echo "     python3 scripts/test_python_integration.py"
    exit 0
else
    echo -e "${COLOR_RED}✗ $FAILURES check(s) failed${COLOR_RESET}"
    echo ""
    echo "Please resolve the issues above before proceeding."
    exit 1
fi
