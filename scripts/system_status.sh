#!/bin/bash

# DSPy Agent System Status Script
# Shows comprehensive system architecture and status

set -e

echo "=========================================="
echo "🚀 DSPy Agent - Production System Status"
echo "=========================================="
echo ""

# System Information
echo "📊 System Information:"
echo "  - Architecture: $(uname -m)"
echo "  - OS: $(uname -s)"
echo "  - Kernel: $(uname -r)"
echo "  - Docker: $(docker --version 2>/dev/null || echo 'Not installed')"
echo "  - Docker Compose: $(docker compose version 2>/dev/null || echo 'Not installed')"
echo "  - Rust: $(rustc --version 2>/dev/null || echo 'Not installed')"
echo "  - Go: $(go version 2>/dev/null || echo 'Not installed')"
echo "  - Python: $(python3 --version 2>/dev/null || echo 'Not installed')"
echo ""

# Service Status
echo "🔧 Service Status:"
echo "  - Redis: $(docker ps --filter name=lightweight-redis-1 --format '{{.Status}}' 2>/dev/null || echo 'Not running')"
echo "  - RedDB: $(docker ps --filter name=reddb --format '{{.Status}}' 2>/dev/null || echo 'Not running')"
echo "  - Go Orchestrator: $(docker ps --filter name=lightweight-go-orchestrator-1 --format '{{.Status}}' 2>/dev/null || echo 'Not running')"
echo "  - Rust Env Runner: $(docker ps --filter name=lightweight-rust-env-runner-1 --format '{{.Status}}' 2>/dev/null || echo 'Not running')"
echo "  - InferMesh: $(docker ps --filter name=lightweight-infermesh-1 --format '{{.Status}}' 2>/dev/null || echo 'Not running')"
echo ""

# Health Checks
echo "🏥 Health Checks:"
echo "  - RedDB: $(curl -s http://localhost:8082/health 2>/dev/null || echo '❌ Down')"
echo "  - Go Orchestrator: $(curl -s http://localhost:9097/metrics 2>/dev/null | head -1 || echo '❌ Down')"
echo "  - Rust Env Runner: $(curl -s http://localhost:8080/health 2>/dev/null || echo '❌ Down')"
echo "  - InferMesh: $(curl -s http://localhost:19000/health 2>/dev/null || echo '❌ Down')"
echo "  - Redis: $(docker exec lightweight-redis-1 redis-cli ping 2>/dev/null || echo '❌ Down')"
echo ""

# Port Status
echo "🌐 Port Status:"
echo "  - 6379 (Redis): $(lsof -i :6379 2>/dev/null | wc -l | tr -d ' ') connections"
echo "  - 8080 (Rust Env Runner): $(lsof -i :8080 2>/dev/null | wc -l | tr -d ' ') connections"
echo "  - 8082 (RedDB): $(lsof -i :8082 2>/dev/null | wc -l | tr -d ' ') connections"
echo "  - 9097 (Go Orchestrator): $(lsof -i :9097 2>/dev/null | wc -l | tr -d ' ') connections"
echo "  - 19000 (InferMesh): $(lsof -i :19000 2>/dev/null | wc -l | tr -d ' ') connections"
echo ""

# Resource Usage
echo "💾 Resource Usage:"
echo "  - Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "  - Disk: $(df -h . | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')"
echo "  - CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo ""

# Docker Status
echo "🐳 Docker Status:"
echo "  - Containers: $(docker ps -q | wc -l | tr -d ' ') running"
echo "  - Images: $(docker images -q | wc -l | tr -d ' ') available"
echo "  - Volumes: $(docker volume ls -q | wc -l | tr -d ' ') volumes"
echo "  - Networks: $(docker network ls -q | wc -l | tr -d ' ') networks"
echo ""

# System Architecture
echo "🏗️ System Architecture:"
echo "  - API Gateway Layer:"
echo "    • FastAPI Backend (Python) - Port 8000"
echo "    • Go Orchestrator (Go) - Port 9097"
echo "    • Rust Env Runner (Rust) - Port 8080"
echo "  - Core Services Layer:"
echo "    • DSPy Agent Core (Python)"
echo "    • Skills System (Python)"
echo "    • Streaming Engine (Python)"
echo "    • RL Training System (Python)"
echo "  - Data & Storage Layer:"
echo "    • Rust RedDB Server (Rust) - Port 8082"
echo "    • Redis Cache (Redis) - Port 6379"
echo "    • InferMesh (Python) - Port 19000"
echo "    • File System Monitoring (Rust)"
echo "  - Infrastructure Layer:"
echo "    • Docker Compose (Orchestration)"
echo "    • Kubernetes (Optional)"
echo "    • Monitoring (Prometheus)"
echo "    • Logging (Structured)"
echo ""

# Performance Metrics
echo "📈 Performance Metrics:"
if command -v curl >/dev/null 2>&1; then
    echo "  - RedDB Response Time: $(curl -w '%{time_total}' -s -o /dev/null http://localhost:8082/health 2>/dev/null || echo 'N/A')s"
    echo "  - Go Orchestrator Response Time: $(curl -w '%{time_total}' -s -o /dev/null http://localhost:9097/metrics 2>/dev/null || echo 'N/A')s"
    echo "  - Rust Env Runner Response Time: $(curl -w '%{time_total}' -s -o /dev/null http://localhost:8080/health 2>/dev/null || echo 'N/A')s"
    echo "  - InferMesh Response Time: $(curl -w '%{time_total}' -s -o /dev/null http://localhost:19000/health 2>/dev/null || echo 'N/A')s"
else
    echo "  - Response times: curl not available"
fi
echo ""

# Quick Actions
echo "⚡ Quick Actions:"
echo "  - Start system: make up"
echo "  - Stop system: make down"
echo "  - Check health: make health"
echo "  - View logs: make logs"
echo "  - Clean system: make clean"
echo "  - Quick start: make quickstart"
echo ""

# Service URLs
echo "🔗 Service URLs:"
echo "  - RedDB: http://localhost:8082"
echo "  - Go Orchestrator: http://localhost:9097"
echo "  - Rust Env Runner: http://localhost:8080"
echo "  - InferMesh: http://localhost:19000"
echo "  - Redis: localhost:6379"
echo ""

echo "=========================================="
echo "✅ System status check complete!"
echo "=========================================="
