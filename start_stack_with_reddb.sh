#!/usr/bin/env bash
set -e

echo "🚀 Starting DSPy Stack with Integrated RedDB..."

# --- Step 1. Check prerequisites ---
echo "🔍 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ ERROR: Docker is required but not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ ERROR: Docker Compose is required but not installed"
    exit 1
fi

if ! command -v openssl &> /dev/null; then
    echo "❌ ERROR: OpenSSL is required for token generation"
    exit 1
fi

# --- Step 2. Set up environment ---
echo "⚙️ Setting up environment..."

# Generate secure admin token if not set
if [[ -z "$REDDB_ADMIN_TOKEN" ]]; then
    export REDDB_ADMIN_TOKEN=$(openssl rand -hex 32)
    echo "🔐 Generated REDDB_ADMIN_TOKEN: $REDDB_ADMIN_TOKEN"
    echo "💾 Save this token: echo 'export REDDB_ADMIN_TOKEN=$REDDB_ADMIN_TOKEN' >> ~/.zshrc"
fi

# Set RedDB environment variables
export REDDB_URL=http://reddb:8080
export REDDB_NAMESPACE=dspy
export REDDB_TOKEN="$REDDB_ADMIN_TOKEN"
export DB_BACKEND=reddb

# --- Step 3. Create/update .env file ---
echo "📝 Creating/updating .env file..."
ENV_FILE="docker/lightweight/.env"

# Create .env file with RedDB configuration
cat > "$ENV_FILE" << EOF
WORKSPACE_DIR=$(pwd)
# RedDB Configuration
REDDB_ADMIN_TOKEN=$REDDB_ADMIN_TOKEN
REDDB_URL=http://reddb:8080
REDDB_NAMESPACE=dspy
REDDB_TOKEN=$REDDB_ADMIN_TOKEN
DB_BACKEND=reddb
EOF

echo "✅ Environment file created: $ENV_FILE"

# --- Step 4. Build and start the stack ---
echo "🏗️ Building and starting the DSPy stack with RedDB..."

# Build the stack
echo "Building Docker images..."
make stack-build

# Start the stack
echo "Starting services..."
make stack-up

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# --- Step 5. Health checks ---
echo "🔍 Running health checks..."

# Check RedDB
echo "Checking RedDB..."
if curl -s -H "Authorization: Bearer $REDDB_ADMIN_TOKEN" http://127.0.0.1:8080/health > /dev/null; then
    echo "✅ RedDB is healthy"
else
    echo "⚠️ RedDB health check failed"
    echo "RedDB logs:"
    docker logs reddb-stack 2>/dev/null || echo "RedDB container not found"
fi

# Run comprehensive health check
echo "Running comprehensive health check..."
make health-check

# --- Step 6. Test RedDB integration ---
echo "🧪 Testing RedDB integration..."

# Test agent CLI with RedDB
echo "Testing agent CLI datasearch..."
if command -v dspy-code &> /dev/null; then
    echo "Testing: dspy-code datasearch 'test query' --ns dspy --top-k 3"
    # Note: This will only work if the agent is properly configured
    echo "✅ Agent CLI available (test manually when agent is ready)"
else
    echo "⚠️ Agent CLI not available in current environment"
fi

# --- Step 7. Show status ---
echo ""
echo "🎉 DSPy Stack with RedDB is running!"
echo ""
echo "📍 Services:"
echo "   RedDB:     http://127.0.0.1:8080 (with auth)"
echo "   Agent:     http://127.0.0.1:8765"
echo "   Dashboard: http://127.0.0.1:18081"
echo "   InferMesh: http://127.0.0.1:19000"
echo ""
echo "🔧 RedDB Configuration:"
echo "   URL: $REDDB_URL"
echo "   Namespace: $REDDB_NAMESPACE"
echo "   Token: ${REDDB_ADMIN_TOKEN:0:8}...${REDDB_ADMIN_TOKEN: -8}"
echo ""
echo "🛑 To stop the stack:"
echo "   make stack-down"
echo ""
echo "📊 To view logs:"
echo "   make stack-logs"
echo ""
echo "🔍 To check health:"
echo "   make health-check"
echo ""
echo "📝 Test RedDB manually:"
echo "   curl -X POST http://127.0.0.1:8080/api/db/ingest \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -H \"Authorization: Bearer \$REDDB_ADMIN_TOKEN\" \\"
echo "     -d '{\"kind\":\"document\",\"namespace\":\"dspy\",\"collection\":\"test\",\"id\":\"test1\",\"text\":\"Test document\"}'"
echo ""
echo "⚠️  SECURITY: Never commit REDDB_ADMIN_TOKEN to version control!"
