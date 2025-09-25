#!/bin/bash
set -euo pipefail

# Fix Agent Rebuild Script
# This script addresses two issues:
# 1. PufferLib build fails on ARM (aarch64) - now skipped on non-x86_64
# 2. IndentationError in code_edit.py - fixed with version bump and no-cache rebuild

echo "[fix-agent] Starting agent rebuild and test process..."

# Change to project root
cd "$(dirname "$0")/.."

# Rebuild dspy-agent with no cache to ensure fresh install
echo "[fix-agent] Rebuilding dspy-agent with no cache..."
DOCKER_BUILDKIT=1 docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env build --no-cache dspy-agent

# Restart agent services
echo "[fix-agent] Restarting agent services..."
docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env up -d --no-deps dspy-agent dspy-worker dspy-worker-backend dspy-worker-frontend

# Test the agent
echo "[fix-agent] Testing agent functionality..."
docker compose -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env exec dspy-agent dspy-agent --help

echo "[fix-agent] Agent rebuild and test completed successfully!"
echo "[fix-agent] The agent should now be working properly."
