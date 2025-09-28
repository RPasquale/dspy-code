#!/bin/bash
set -e

echo "🔧 Fixing Kafka Connectivity Issues..."

# Stop all services
echo "🛑 Stopping all services..."
docker-compose -f docker/lightweight/docker-compose.yml down --remove-orphans

# Clean up any orphaned containers
echo "🧹 Cleaning up orphaned containers..."
docker container prune -f

# Remove Kafka and Zookeeper volumes to start fresh
echo "🗑️ Removing Kafka and Zookeeper data volumes..."
docker volume rm $(docker volume ls -q | grep -E "(kafka|zookeeper)") 2>/dev/null || true

# Start infrastructure services first
echo "🚀 Starting infrastructure services (Zookeeper, Kafka, Redis)..."
docker-compose -f docker/lightweight/docker-compose.yml up -d zookeeper redis

# Wait for Zookeeper to be ready
echo "⏳ Waiting for Zookeeper to be ready..."
for i in {1..30}; do
    if docker-compose -f docker/lightweight/docker-compose.yml exec zookeeper echo "Zookeeper ready" 2>/dev/null; then
        break
    fi
    echo "Waiting for Zookeeper... ($i/30)"
    sleep 2
done

# Start Kafka
echo "📨 Starting Kafka..."
docker-compose -f docker/lightweight/docker-compose.yml up -d kafka

# Wait for Kafka to be ready
echo "⏳ Waiting for Kafka to be ready..."
for i in {1..40}; do
    if docker-compose -f docker/lightweight/docker-compose.yml exec kafka /opt/bitnami/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092 >/dev/null 2>&1; then
        break
    fi
    echo "Waiting for Kafka... ($i/40)"
    sleep 3
done

# Create required Kafka topics
echo "📋 Creating required Kafka topics..."
docker-compose -f docker/lightweight/docker-compose.yml exec kafka /opt/bitnami/kafka/bin/kafka-topics.sh --create --if-not-exists --bootstrap-server localhost:9092 --topic agent.results --partitions 3 --replication-factor 1
docker-compose -f docker/lightweight/docker-compose.yml exec kafka /opt/bitnami/kafka/bin/kafka-topics.sh --create --if-not-exists --bootstrap-server localhost:9092 --topic embedding_input --partitions 3 --replication-factor 1
docker-compose -f docker/lightweight/docker-compose.yml exec kafka /opt/bitnami/kafka/bin/kafka-topics.sh --create --if-not-exists --bootstrap-server localhost:9092 --topic embeddings --partitions 3 --replication-factor 1

# Start remaining services
echo "🚀 Starting remaining services..."
docker-compose -f docker/lightweight/docker-compose.yml up -d

# Wait for all services to be healthy
echo "⏳ Waiting for all services to be healthy..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
docker-compose -f docker/lightweight/docker-compose.yml ps

echo "✅ Kafka connectivity fix complete!"
echo ""
echo "📍 Services should now be available:"
echo "   Frontend:    http://localhost:5176"
echo "   Dashboard:   http://localhost:8080"
echo "   Agent:       http://localhost:8765"
echo ""
echo "🔍 To check logs: docker-compose -f docker/lightweight/docker-compose.yml logs -f"
