# Production Deployment Guide

**For DevOps and System Administrators**

---

## Overview

This guide covers deploying the DSPy Agent Infrastructure (Rust env-manager + Go orchestrator) to production using systemd services.

---

## Quick Deployment

```bash
# 1. Build binaries (see BUILD_INSTRUCTIONS.md)
cd env_manager_rs && cargo build --release && cd ..
cd orchestrator && go build -o orchestrator-linux ./cmd/orchestrator && cd ..

# 2. Deploy to production
sudo ./scripts/deploy_production.sh

# 3. Start services
sudo systemctl start env-manager
sudo systemctl start orchestrator

# 4. Verify
sudo systemctl status env-manager
sudo systemctl status orchestrator
curl http://localhost:9097/metrics
```

---

## Architecture

### Service Dependencies

```
docker.service
    ↓
env-manager.service (Rust, port 50100)
    ↓
orchestrator.service (Go, ports 9097, 50052)
    ↓
Python Application (your code)
```

### Communication Ports

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| env-manager | 50100 | gRPC | Container management API |
| orchestrator | 50052 | gRPC | Task submission (Python connects here) |
| orchestrator | 9097 | HTTP | Metrics, status, workflows |

---

## System Requirements

### Hardware

- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ (4GB minimum)
- **Disk**: 20GB+ for logs and data
- **Network**: 1Gbps+ for high-throughput workloads

### Software

- **OS**: Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+)
- **Docker**: 20.10+ (running and enabled)
- **Systemd**: 245+ (for service management)
- **User Permissions**: Root access for installation

---

## Installation Steps

### 1. Build Binaries

See `BUILD_INSTRUCTIONS.md` for detailed build process.

```bash
# Rust env-manager
cd env_manager_rs
cargo build --release
cd ..

# Go orchestrator  
cd orchestrator
go build -o orchestrator-linux ./cmd/orchestrator
cd ..

# Verify binaries exist
ls -lh env_manager_rs/target/release/env-manager
ls -lh orchestrator/orchestrator-linux
```

### 2. Run Deployment Script

```bash
# Clone or navigate to repository
cd /path/to/dspy-code

# Run deployment script as root
sudo ./scripts/deploy_production.sh
```

**What the script does**:
1. Creates service user (`dspy`)
2. Creates directory structure (`/opt/dspy-agent/`)
3. Copies binaries to `/opt/dspy-agent/bin/`
4. Installs systemd service files
5. Creates default configuration
6. Enables services

### 3. Configure Services

#### Environment Manager

Edit `/opt/dspy-agent/config/env-manager.toml`:

```toml
grpc_addr = "0.0.0.0:50100"
max_concurrent_starts = 5
health_check_timeout_secs = 60
health_check_max_attempts = 30
verbose_logging = false

# Override specific services
[[service_overrides]]
name = "redis"
required = true
health_check_url = "http://localhost:6379"

[[service_overrides]]
name = "ollama"
required = false  # Make optional
```

#### Orchestrator

Edit `/etc/systemd/system/orchestrator.service` environment section:

```ini
Environment="ORCHESTRATOR_GRPC_ADDR=:50052"
Environment="ENV_MANAGER_ADDR=localhost:50100"
Environment="WORKFLOW_STORE_DIR=/opt/dspy-agent/data/workflows"
Environment="ENV_QUEUE_DIR=/opt/dspy-agent/logs/env_queue"
```

### 4. Start Services

```bash
# Start env-manager first
sudo systemctl start env-manager

# Wait for it to be ready
sleep 5

# Start orchestrator
sudo systemctl start orchestrator

# Enable auto-start on boot
sudo systemctl enable env-manager
sudo systemctl enable orchestrator
```

---

## Service Management

### Starting Services

```bash
# Start individual services
sudo systemctl start env-manager
sudo systemctl start orchestrator

# Start all (if using target)
sudo systemctl start dspy-agent.target
```

### Stopping Services

```bash
# Stop individual services (graceful, 30s timeout)
sudo systemctl stop orchestrator
sudo systemctl stop env-manager

# Force stop if needed
sudo systemctl kill -s SIGKILL env-manager
```

### Restarting Services

```bash
# Restart individual services
sudo systemctl restart env-manager
sudo systemctl restart orchestrator

# Reload configuration without restart
sudo systemctl reload env-manager
```

### Checking Status

```bash
# Check service status
sudo systemctl status env-manager
sudo systemctl status orchestrator

# Check if running
sudo systemctl is-active env-manager
sudo systemctl is-active orchestrator

# Check if enabled
sudo systemctl is-enabled env-manager
```

### Viewing Logs

```bash
# Follow live logs
sudo journalctl -u env-manager -f
sudo journalctl -u orchestrator -f

# View recent logs
sudo journalctl -u env-manager -n 100
sudo journalctl -u orchestrator -n 100

# View logs since specific time
sudo journalctl -u env-manager --since "1 hour ago"
sudo journalctl -u env-manager --since "2024-01-01"

# Filter by priority
sudo journalctl -u env-manager -p err  # Only errors
sudo journalctl -u env-manager -p warning  # Warnings and above
```

---

## Monitoring

### Health Checks

```bash
# Check env-manager gRPC (should connect)
grpcurl -plaintext localhost:50100 list

# Check orchestrator HTTP
curl http://localhost:9097/metrics
curl http://localhost:9097/queue/status

# Check orchestrator gRPC
grpcurl -plaintext localhost:50052 list
```

### Metrics

**Prometheus Metrics** (orchestrator):
```bash
curl http://localhost:9097/metrics
```

Key metrics:
- `env_queue_depth` - Number of queued tasks
- `gpu_wait_seconds` - Average GPU wait time
- `env_error_rate` - Task error rate
- `workflows_total` - Registered workflows
- `runner_gpu_total` - Detected GPUs

**Service Status** (env-manager):
```bash
# Via gRPC (requires grpcurl)
grpcurl -plaintext localhost:50100 \
  env_manager.v1.EnvManagerService/GetServiceStatus \
  -d '{"service_name": "redis"}'
```

### Resource Usage

```bash
# CPU and memory usage
systemctl status env-manager
systemctl status orchestrator

# Detailed resource info
sudo systemd-cgtop -m

# Service-specific resources
sudo systemctl show env-manager --property=MemoryCurrent
sudo systemctl show env-manager --property=CPUUsageNSec
```

---

## Troubleshooting

### Service Won't Start

**Check logs**:
```bash
sudo journalctl -u env-manager -n 50
sudo journalctl -u orchestrator -n 50
```

**Common issues**:

1. **Port already in use**:
   ```bash
   # Check what's using the port
   sudo netstat -tulpn | grep 50100
   sudo netstat -tulpn | grep 50052
   ```

2. **Docker not accessible**:
   ```bash
   # Check Docker is running
   sudo systemctl status docker
   
   # Test Docker access
   sudo -u dspy docker ps
   ```

3. **Permission denied**:
   ```bash
   # Ensure user is in docker group
   sudo usermod -aG docker dspy
   sudo systemctl restart env-manager
   ```

### Service Keeps Restarting

**Check crash logs**:
```bash
sudo journalctl -u env-manager -p err
sudo coredumpctl list
```

**Check resource limits**:
```bash
sudo systemctl show env-manager --property=LimitNOFILE
sudo systemctl show env-manager --property=MemoryLimit
```

**Increase limits** if needed (edit service file):
```ini
[Service]
LimitNOFILE=100000
MemoryLimit=2G
```

### High Resource Usage

**Check what's consuming resources**:
```bash
sudo systemd-cgtop
htop
```

**Adjust resource limits**:
```ini
# /etc/systemd/system/env-manager.service
[Service]
CPUQuota=200%      # Max 2 cores
MemoryLimit=1G     # Max 1GB RAM
TasksMax=1000      # Max processes
```

Then reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart env-manager
```

---

## Security Hardening

### Service Isolation

Services run with:
- ✅ Dedicated user (`dspy`)
- ✅ Private `/tmp` directory
- ✅ No privilege escalation
- ✅ Protected system directories
- ✅ Limited file access

### File Permissions

```bash
# Secure installation directory
sudo chown -R root:dspy /opt/dspy-agent
sudo chmod 750 /opt/dspy-agent

# Secure binaries
sudo chmod 550 /opt/dspy-agent/bin/*

# Secure configuration
sudo chmod 640 /opt/dspy-agent/config/*.toml
```

### Network Security

```bash
# Bind env-manager to localhost only (if not accessed remotely)
# Edit /etc/systemd/system/env-manager.service:
Environment="ENV_MANAGER_GRPC_ADDR=127.0.0.1:50100"

# Use firewall to restrict access
sudo ufw allow from 10.0.0.0/8 to any port 50100
sudo ufw allow from 10.0.0.0/8 to any port 50052
```

### Audit Logging

```bash
# Enable detailed audit logging
# Edit service files:
Environment="ENV_MANAGER_VERBOSE=true"
Environment="RUST_LOG=debug"

# Centralize logs
# Edit /etc/systemd/system/env-manager.service:
StandardOutput=syslog
StandardError=syslog
SyslogFacility=local1
```

---

## Backup and Recovery

### What to Backup

```bash
# Configuration
/opt/dspy-agent/config/

# Data (workflows, runs)
/opt/dspy-agent/data/

# Logs (optional, for forensics)
/opt/dspy-agent/logs/

# Service definitions
/etc/systemd/system/env-manager.service
/etc/systemd/system/orchestrator.service
/etc/systemd/system/dspy-agent.target
```

### Backup Script

```bash
#!/bin/bash
BACKUP_DIR="/backup/dspy-agent-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup config and data
cp -r /opt/dspy-agent/config "$BACKUP_DIR/"
cp -r /opt/dspy-agent/data "$BACKUP_DIR/"

# Backup service files
mkdir -p "$BACKUP_DIR/systemd"
cp /etc/systemd/system/env-manager.service "$BACKUP_DIR/systemd/"
cp /etc/systemd/system/orchestrator.service "$BACKUP_DIR/systemd/"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"
```

### Restore

```bash
# Extract backup
tar -xzf dspy-agent-20240112-120000.tar.gz

# Stop services
sudo systemctl stop orchestrator env-manager

# Restore files
sudo cp -r dspy-agent-20240112-120000/config/* /opt/dspy-agent/config/
sudo cp -r dspy-agent-20240112-120000/data/* /opt/dspy-agent/data/

# Restore service files
sudo cp dspy-agent-20240112-120000/systemd/* /etc/systemd/system/
sudo systemctl daemon-reload

# Start services
sudo systemctl start env-manager orchestrator
```

---

## Upgrading

### Rolling Update

```bash
# 1. Build new binaries
cd /path/to/updated/code
cargo build --release
go build -o orchestrator-linux ./cmd/orchestrator

# 2. Stop services
sudo systemctl stop orchestrator
sudo systemctl stop env-manager

# 3. Backup old binaries
sudo cp /opt/dspy-agent/bin/env-manager /opt/dspy-agent/bin/env-manager.old
sudo cp /opt/dspy-agent/bin/orchestrator-linux /opt/dspy-agent/bin/orchestrator-linux.old

# 4. Install new binaries
sudo cp env_manager_rs/target/release/env-manager /opt/dspy-agent/bin/
sudo cp orchestrator/orchestrator-linux /opt/dspy-agent/bin/
sudo chmod +x /opt/dspy-agent/bin/*

# 5. Restart services
sudo systemctl start env-manager
sleep 5
sudo systemctl start orchestrator

# 6. Verify
sudo systemctl status env-manager orchestrator
curl http://localhost:9097/metrics
```

### Rollback

```bash
# Stop services
sudo systemctl stop orchestrator env-manager

# Restore old binaries
sudo mv /opt/dspy-agent/bin/env-manager.old /opt/dspy-agent/bin/env-manager
sudo mv /opt/dspy-agent/bin/orchestrator-linux.old /opt/dspy-agent/bin/orchestrator-linux

# Restart
sudo systemctl start env-manager orchestrator
```

---

## Uninstallation

```bash
# Stop and disable services
sudo systemctl stop orchestrator env-manager
sudo systemctl disable orchestrator env-manager

# Remove service files
sudo rm /etc/systemd/system/env-manager.service
sudo rm /etc/systemd/system/orchestrator.service
sudo rm /etc/systemd/system/dspy-agent.target
sudo systemctl daemon-reload

# Remove installation directory (CAUTION: deletes data!)
sudo rm -rf /opt/dspy-agent

# Remove CLI (if installed)
sudo rm /usr/local/bin/dspy-agent

# Remove service user
sudo userdel dspy
```

---

## Production Checklist

### Pre-Deployment

- [ ] Binaries built and tested
- [ ] Docker installed and running
- [ ] Sufficient disk space (20GB+)
- [ ] Firewall rules configured
- [ ] Backup strategy planned

### Deployment

- [ ] Deployment script executed successfully
- [ ] Services started without errors
- [ ] Health checks passing
- [ ] Metrics endpoint accessible
- [ ] Python application can connect

### Post-Deployment

- [ ] Monitoring configured
- [ ] Logs shipping to central system
- [ ] Alerting rules configured
- [ ] Documentation updated
- [ ] Team trained on operations

---

## Support

**For deployment issues**:
1. Check logs: `journalctl -u env-manager -u orchestrator`
2. Verify Docker: `systemctl status docker`
3. Test connectivity: `curl http://localhost:9097/metrics`
4. Review documentation: `docs/PYTHON_INTEGRATION_GUIDE.md`

**Emergency contacts**:
- Infrastructure team: Rust/Go developer
- Application team: Python developer

---

**Last Updated**: October 12, 2025  
**Infrastructure Version**: 0.1.0  
**Deployment Method**: systemd services

