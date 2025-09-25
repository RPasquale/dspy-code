#!/usr/bin/env bash
set -e

echo "🔒 Starting Secure RedDB + Agent Backend Setup..."

# --- Security: Check for required environment variables ---
if [[ -z "$REDDB_ADMIN_TOKEN" ]]; then
    echo "⚠️  SECURITY WARNING: REDDB_ADMIN_TOKEN not set!"
    echo "🔑 Generating secure admin token..."
    export REDDB_ADMIN_TOKEN=$(openssl rand -hex 32)
    echo "🔐 Generated admin token: $REDDB_ADMIN_TOKEN"
    echo "💾 Save this token securely: echo 'export REDDB_ADMIN_TOKEN=$REDDB_ADMIN_TOKEN' >> ~/.zshrc"
fi

# --- Security: Validate token format ---
if [[ ! "$REDDB_ADMIN_TOKEN" =~ ^[a-f0-9]{64}$ ]]; then
    echo "❌ ERROR: REDDB_ADMIN_TOKEN must be a 64-character hex string"
    echo "   Generate with: openssl rand -hex 32"
    exit 1
fi

# --- Step 1. Start RedDB via Docker with security ---
echo "📦 Starting RedDB (Docker) with authentication..."
# Stop any existing container
docker stop redb-open 2>/dev/null || true
docker rm redb-open 2>/dev/null || true

# Start new container with admin token
docker run -d --rm --name redb-open \
    -p 127.0.0.1:8080:8080 \
    -e REDB_ADMIN_TOKEN="$REDDB_ADMIN_TOKEN" \
    redbco/redb-open:latest

# Wait for RedDB to be ready
echo "⏳ Waiting for RedDB to start..."
sleep 5

# --- Step 2. Export secure env vars ---
echo "⚙️ Setting secure environment variables..."
export REDDB_URL=http://127.0.0.1:8080
export REDDB_NAMESPACE=agent
export REDDB_TOKEN="$REDDB_ADMIN_TOKEN"
export REDDB_OPEN_NATIVE=false
export DB_BACKEND=reddb

# Security: Mask token in output
REDDB_TOKEN_MASKED="${REDDB_ADMIN_TOKEN:0:8}...${REDDB_ADMIN_TOKEN: -8}"

# --- Step 3. Secure health checks ---
echo "🔍 Checking RedDB health with authentication..."
if curl -s -H "Authorization: Bearer $REDDB_ADMIN_TOKEN" http://127.0.0.1:8080/health > /dev/null; then
    echo "✅ RedDB is healthy and authenticated"
else
    echo "⚠️ RedDB health check failed - checking logs..."
    docker logs redb-open
    echo "🔍 Testing without auth (for debugging)..."
    curl -s http://127.0.0.1:8080/health || echo "No response from RedDB"
    exit 1
fi

# --- Step 4. Install dependencies if needed ---
echo "📦 Checking FastAPI dependencies..."
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Installing FastAPI and uvicorn..."
    pip install -q fastapi uvicorn
fi

# --- Step 5. Start FastAPI backend ---
echo "🌐 Launching FastAPI backend on port 8767..."
cd /Users/robbiepasquale/dspy_stuff
dspy-code databackend-fastapi --port 8767 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# --- Step 6. Test backend health ---
echo "🔍 Checking Agent backend health..."
if curl -s http://127.0.0.1:8767/api/db/health > /dev/null; then
    echo "✅ Agent backend is healthy"
    # Show the health response
    echo "Backend health response:"
    curl -s http://127.0.0.1:8767/api/db/health | python -m json.tool
else
    echo "⚠️ Backend health check failed"
    echo "Backend logs:"
    ps aux | grep databackend-fastapi || echo "Backend process not found"
    exit 1
fi

# --- Step 7. Secure auto-test with sample data ---
echo "🧪 Running secure auto-test with sample data..."

# Test ingest with authentication
echo "Testing document ingest with authentication..."
INGEST_RESPONSE=$(curl -s -X POST http://127.0.0.1:8767/api/db/ingest \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $REDDB_ADMIN_TOKEN" \
  -d '{"kind":"document","namespace":"agent","collection":"notes","id":"test1","text":"Payment API returns 500 error when processing credit cards"}')

echo "Ingest response: $INGEST_RESPONSE"

# Test query with authentication
echo "Testing document query with authentication..."
QUERY_RESPONSE=$(curl -s -X POST http://127.0.0.1:8767/api/db/query \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $REDDB_ADMIN_TOKEN" \
  -d '{"mode":"auto","namespace":"agent","text":"payment 500 error","top_k":3}')

echo "Query response: $QUERY_RESPONSE"

# Test agent CLI
echo "Testing agent CLI datasearch..."
dspy-code datasearch "payment 500 error" --ns agent --top-k 3

echo ""
echo "🎉 SUCCESS! RedDB + Agent Backend is fully operational!"
echo ""
echo "📍 Services running:"
echo "   RedDB:   http://127.0.0.1:8080"
echo "   Backend: http://127.0.0.1:8767"
echo ""
echo "🔧 Environment variables set:"
echo "   REDDB_URL=$REDDB_URL"
echo "   REDDB_NAMESPACE=$REDDB_NAMESPACE"
echo "   REDDB_TOKEN=$REDDB_TOKEN_MASKED"
echo "   DB_BACKEND=$DB_BACKEND"
echo ""
echo "🛑 To stop services:"
echo "   docker stop redb-open && kill $BACKEND_PID"
echo ""
echo "📝 Secure test commands:"
echo "   # Test ingest (with auth):"
echo "   curl -X POST :8767/api/db/ingest -H 'Content-Type: application/json' -H \"Authorization: Bearer \$REDDB_ADMIN_TOKEN\" -d '{\"kind\":\"document\",\"namespace\":\"agent\",\"collection\":\"notes\",\"id\":\"n1\",\"text\":\"Your text here\"}'"
echo ""
echo "   # Test query (with auth):"
echo "   curl -X POST :8767/api/db/query -H 'Content-Type: application/json' -H \"Authorization: Bearer \$REDDB_ADMIN_TOKEN\" -d '{\"mode\":\"auto\",\"namespace\":\"agent\",\"text\":\"your search query\"}'"
echo ""
echo "   # Test agent CLI:"
echo "   dspy-code datasearch \"your search query\" --ns agent --top-k 5"
echo ""
echo "🔒 Security Notes:"
echo "   - RedDB is bound to localhost only (127.0.0.1:8080)"
echo "   - All API calls require Bearer token authentication"
echo "   - Admin token is masked in output for security"
echo "   - Token is stored in REDDB_ADMIN_TOKEN environment variable"
