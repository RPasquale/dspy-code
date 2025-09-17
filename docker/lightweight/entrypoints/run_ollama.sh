#!/bin/sh
set -e

echo "[entrypoint] starting ollama server"

# Start ollama server in background
ollama serve &
SERVE_PID=$!

# Wait a moment for server to start
sleep 10

# Pull the model if it doesn't exist
echo "[entrypoint] pulling qwen3:1.7b model"
ollama pull qwen3:1.7b || true

echo "[entrypoint] ollama ready, keeping server running"

# Wait for the server process
wait $SERVE_PID