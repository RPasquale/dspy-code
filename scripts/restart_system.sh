#!/bin/bash
set -e

echo "🚀 Restarting DSPy System with Kafka Fixes..."

# Stop all services
echo "🛑 Stopping all services..."
docker-compose -f docker/lightweight/docker-compose.yml down --remove-orphans

# Clean up
echo "🧹 Cleaning up..."
docker container prune -f
docker volume prune -f

# Start services in proper order
echo "🚀 Starting services in proper order..."

# 1. Start infrastructure first
echo "📦 Starting infrastructure (Zookeeper, Redis)..."
docker-compose -f docker/lightweight/docker-compose.yml up -d zookeeper redis

# Wait a bit for infrastructure
echo "⏳ Waiting for infrastructure to start..."
sleep 10

# 2. Start Kafka
echo "📨 Starting Kafka..."
docker-compose -f docker/lightweight/docker-compose.yml up -d kafka

# Wait for Kafka to be ready
echo "⏳ Waiting for Kafka to be ready..."
sleep 30

# 3. Start core services
echo "🔧 Starting core services..."
docker-compose -f docker/lightweight/docker-compose.yml up -d reddb fastapi-backend ollama infermesh

# Wait for core services
echo "⏳ Waiting for core services..."
sleep 20

# 4. Start workers and agents
echo "👷 Starting workers and agents..."
docker-compose -f docker/lightweight/docker-compose.yml up -d

# Wait for everything to stabilize
echo "⏳ Waiting for all services to stabilize..."
sleep 30

# Check status
echo "🔍 Checking service status..."
docker-compose -f docker/lightweight/docker-compose.yml ps

echo ""
echo "✅ System restart complete!"
echo ""
echo "📍 Services:"
echo "   Frontend:    http://localhost:5176"
echo "   Dashboard:   http://localhost:8080"
echo "   Agent:       http://localhost:8765"
echo ""
echo "🔍 To check logs: docker-compose -f docker/lightweight/docker-compose.yml logs -f"
