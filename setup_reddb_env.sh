#!/usr/bin/env bash
# Secure RedDB Environment Setup Script
# Run this to set up your environment for RedDB testing

echo "🔒 Setting up secure RedDB environment variables..."

# Security: Check for existing token
if [[ -z "$REDDB_ADMIN_TOKEN" ]]; then
    echo "⚠️  SECURITY WARNING: REDDB_ADMIN_TOKEN not set!"
    echo "🔑 Generating secure admin token..."
    export REDDB_ADMIN_TOKEN=$(openssl rand -hex 32)
    echo "🔐 Generated admin token: $REDDB_ADMIN_TOKEN"
    echo "💾 Save this token securely: echo 'export REDDB_ADMIN_TOKEN=$REDDB_ADMIN_TOKEN' >> ~/.zshrc"
else
    echo "✅ Using existing REDDB_ADMIN_TOKEN"
fi

# Security: Validate token format
if [[ ! "$REDDB_ADMIN_TOKEN" =~ ^[a-f0-9]{64}$ ]]; then
    echo "❌ ERROR: REDDB_ADMIN_TOKEN must be a 64-character hex string"
    echo "   Generate with: openssl rand -hex 32"
    exit 1
fi

# Export secure RedDB configuration
export REDDB_URL=http://127.0.0.1:8080
export REDDB_NAMESPACE=agent
export REDDB_TOKEN="$REDDB_ADMIN_TOKEN"
export REDDB_OPEN_NATIVE=false
export DB_BACKEND=reddb

# Agent configuration
export MODEL_NAME=gpt-4o-mini
export USE_OLLAMA=true
export LOCAL_MODE=false
export TOOL_APPROVAL=auto

# Security: Mask token in output
REDDB_TOKEN_MASKED="${REDDB_ADMIN_TOKEN:0:8}...${REDDB_ADMIN_TOKEN: -8}"

echo "✅ Secure environment variables set:"
echo "   REDDB_URL=$REDDB_URL"
echo "   REDDB_NAMESPACE=$REDDB_NAMESPACE"
echo "   REDDB_TOKEN=$REDDB_TOKEN_MASKED"
echo "   DB_BACKEND=$DB_BACKEND"
echo ""
echo "🔒 Security features enabled:"
echo "   - Bearer token authentication required"
echo "   - Localhost-only binding (127.0.0.1)"
echo "   - Secure token generation and validation"
echo ""
echo "💡 To make these permanent, add them to your shell profile:"
echo "   echo 'export REDDB_URL=http://127.0.0.1:8080' >> ~/.zshrc"
echo "   echo 'export REDDB_NAMESPACE=agent' >> ~/.zshrc"
echo "   echo 'export DB_BACKEND=reddb' >> ~/.zshrc"
echo "   echo 'export REDDB_ADMIN_TOKEN=$REDDB_ADMIN_TOKEN' >> ~/.zshrc"
echo "   source ~/.zshrc"
echo ""
echo "⚠️  SECURITY WARNING: Never commit REDDB_ADMIN_TOKEN to version control!"
