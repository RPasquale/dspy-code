#!/bin/bash
# Production Deployment Script for DSPy Agent Infrastructure
# This script installs the Rust/Go binaries and sets up systemd services

set -e

echo "🚀 DSPy Agent Infrastructure - Production Deployment"
echo "=================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/dspy-agent"
BIN_DIR="$INSTALL_DIR/bin"
DATA_DIR="$INSTALL_DIR/data"
LOG_DIR="$INSTALL_DIR/logs"
CONFIG_DIR="$INSTALL_DIR/config"
SERVICE_USER="dspy"
SERVICE_GROUP="dspy"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}✗ This script must be run as root${NC}"
   exit 1
fi

echo -e "${GREEN}✓ Running as root${NC}"

# Check if binaries exist
echo ""
echo "Checking for binaries..."

ENV_MANAGER_BIN="env_manager_rs/target/release/env-manager"
ORCHESTRATOR_BIN="orchestrator/orchestrator-linux"
CLI_BIN="cmd/dspy-agent/dspy-agent"

if [ ! -f "$ENV_MANAGER_BIN" ]; then
    echo -e "${RED}✗ env-manager binary not found at $ENV_MANAGER_BIN${NC}"
    echo "  Run: cd env_manager_rs && cargo build --release"
    exit 1
fi
echo -e "${GREEN}✓ env-manager binary found${NC}"

if [ ! -f "$ORCHESTRATOR_BIN" ]; then
    echo -e "${YELLOW}⚠ orchestrator binary not found at $ORCHESTRATOR_BIN${NC}"
    echo "  This is optional if running orchestrator separately"
    ORCHESTRATOR_BIN=""
fi
if [ -n "$ORCHESTRATOR_BIN" ]; then
    echo -e "${GREEN}✓ orchestrator binary found${NC}"
fi

if [ ! -f "$CLI_BIN" ]; then
    echo -e "${YELLOW}⚠ CLI binary not found at $CLI_BIN${NC}"
    echo "  This is optional"
    CLI_BIN=""
fi
if [ -n "$CLI_BIN" ]; then
    echo -e "${GREEN}✓ CLI binary found${NC}"
fi

# Create service user
echo ""
echo "Creating service user..."
if id "$SERVICE_USER" &>/dev/null; then
    echo -e "${GREEN}✓ User $SERVICE_USER already exists${NC}"
else
    useradd -r -s /bin/false -d "$INSTALL_DIR" "$SERVICE_USER"
    echo -e "${GREEN}✓ Created user $SERVICE_USER${NC}"
fi

# Add user to docker group
usermod -aG docker "$SERVICE_USER" || true
echo -e "${GREEN}✓ Added $SERVICE_USER to docker group${NC}"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$DATA_DIR"/{workflows,workflow_runs}
mkdir -p "$LOG_DIR"/{env_queue/{pending,done},traces}
mkdir -p "$CONFIG_DIR"

chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
chmod 755 "$INSTALL_DIR"
chmod 755 "$BIN_DIR"
chmod 755 "$DATA_DIR"
chmod 755 "$LOG_DIR"

echo -e "${GREEN}✓ Created directory structure${NC}"

# Copy binaries
echo ""
echo "Installing binaries..."
cp "$ENV_MANAGER_BIN" "$BIN_DIR/env-manager"
chmod +x "$BIN_DIR/env-manager"
echo -e "${GREEN}✓ Installed env-manager${NC}"

if [ -n "$ORCHESTRATOR_BIN" ]; then
    cp "$ORCHESTRATOR_BIN" "$BIN_DIR/orchestrator-linux"
    chmod +x "$BIN_DIR/orchestrator-linux"
    echo -e "${GREEN}✓ Installed orchestrator${NC}"
fi

if [ -n "$CLI_BIN" ]; then
    cp "$CLI_BIN" /usr/local/bin/dspy-agent
    chmod +x /usr/local/bin/dspy-agent
    echo -e "${GREEN}✓ Installed dspy-agent CLI to /usr/local/bin${NC}"
fi

# Install systemd service files
echo ""
echo "Installing systemd services..."

if [ -f "deploy/systemd/env-manager.service" ]; then
    cp deploy/systemd/env-manager.service /etc/systemd/system/
    echo -e "${GREEN}✓ Installed env-manager.service${NC}"
else
    echo -e "${YELLOW}⚠ env-manager.service not found, skipping${NC}"
fi

if [ -n "$ORCHESTRATOR_BIN" ] && [ -f "deploy/systemd/orchestrator.service" ]; then
    cp deploy/systemd/orchestrator.service /etc/systemd/system/
    echo -e "${GREEN}✓ Installed orchestrator.service${NC}"
fi

if [ -f "deploy/systemd/dspy-agent.target" ]; then
    cp deploy/systemd/dspy-agent.target /etc/systemd/system/
    echo -e "${GREEN}✓ Installed dspy-agent.target${NC}"
fi

# Reload systemd
systemctl daemon-reload
echo -e "${GREEN}✓ Reloaded systemd${NC}"

# Create default configuration
echo ""
echo "Creating default configuration..."
cat > "$CONFIG_DIR/env-manager.toml" <<EOF
# DSPy Environment Manager Configuration

grpc_addr = "0.0.0.0:50100"
max_concurrent_starts = 5
health_check_timeout_secs = 60
health_check_max_attempts = 30
verbose_logging = false

# Service overrides (optional)
# [[service_overrides]]
# name = "redis"
# required = true
# health_check_url = "http://localhost:6379"
EOF

chown "$SERVICE_USER:$SERVICE_GROUP" "$CONFIG_DIR/env-manager.toml"
echo -e "${GREEN}✓ Created default configuration${NC}"

# Enable services
echo ""
echo "Enabling services..."
systemctl enable env-manager.service
echo -e "${GREEN}✓ Enabled env-manager.service${NC}"

if [ -n "$ORCHESTRATOR_BIN" ]; then
    systemctl enable orchestrator.service
    echo -e "${GREEN}✓ Enabled orchestrator.service${NC}"
fi

# Summary
echo ""
echo "=================================================="
echo -e "${GREEN}✓ Deployment complete!${NC}"
echo ""
echo "Services installed:"
echo "  • env-manager    (systemctl status env-manager)"
if [ -n "$ORCHESTRATOR_BIN" ]; then
    echo "  • orchestrator   (systemctl status orchestrator)"
fi
echo ""
echo "Installation directory: $INSTALL_DIR"
echo "Configuration: $CONFIG_DIR/env-manager.toml"
echo "Logs: $LOG_DIR"
echo ""
echo "To start services:"
echo "  systemctl start env-manager"
if [ -n "$ORCHESTRATOR_BIN" ]; then
    echo "  systemctl start orchestrator"
fi
echo ""
echo "To view logs:"
echo "  journalctl -u env-manager -f"
if [ -n "$ORCHESTRATOR_BIN" ]; then
    echo "  journalctl -u orchestrator -f"
fi
echo ""
echo "To enable auto-start on boot (already done):"
echo "  systemctl enable env-manager"
if [ -n "$ORCHESTRATOR_BIN" ]; then
    echo "  systemctl enable orchestrator"
fi
echo ""
echo -e "${YELLOW}⚠ Remember to configure Docker to start on boot:${NC}"
echo "  systemctl enable docker"
echo ""
echo "=================================================="

