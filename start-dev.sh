#!/bin/bash
set -e

echo "🚀 Starting DSPy Development Stack..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    echo "   Run: open -a Docker"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "Makefile" ]; then
    echo "❌ Please run this script from the dspy_stuff root directory"
    exit 1
fi

echo "🔧 Setting up environment..."
make stack-env

echo "🏗️ Building the stack..."
make stack-build

echo "🚀 Starting all services..."
make stack-up

echo "⏳ Waiting for services to start..."
sleep 15

echo "🔍 Checking service health..."
make health-check

echo ""
echo "🎉 DSPy Stack is running!"
echo ""
echo "📍 Services:"
echo "   Frontend:    http://localhost:5176"
echo "   Dashboard:   http://localhost:8080"
echo "   RedDB:       http://localhost:8080 (with auth)"
echo "   Agent:       http://localhost:8765"
echo ""
echo "🛑 To stop: make stack-down"
echo "📊 To view logs: make stack-logs"
echo "🔍 To check health: make health-check"
echo ""
echo "✨ Happy coding!"
