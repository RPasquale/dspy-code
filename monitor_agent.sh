#!/bin/bash

echo "🎯 DSPy Agent Activity Monitor"
echo "=============================="
echo ""

# Function to show status
show_status() {
    echo "📊 Container Status:"
    cd /Users/robbiepasquale/dspy_stuff/docker/lightweight
    docker-compose ps | grep -E "(dspy-agent|ollama)" | while read line; do
        echo "  $line"
    done
    echo ""
}

# Function to show recent logs
show_recent_logs() {
    echo "📝 Recent Agent Activity (last 10 lines):"
    cd /Users/robbiepasquale/dspy_stuff/docker/lightweight
    docker-compose logs dspy-agent --tail=10 | grep -v "MicroBatchExecution" | tail -10
    echo ""
}

# Function to check Ollama
check_ollama() {
    echo "🤖 Ollama Status:"
    if curl -s http://localhost:11435/api/tags >/dev/null 2>&1; then
        echo "  ✅ Ollama is responding"
        models=$(curl -s http://localhost:11435/api/tags | jq -r '.models[].name' 2>/dev/null | head -3)
        if [ ! -z "$models" ]; then
            echo "  📚 Available models:"
            echo "$models" | while read model; do
                echo "    - $model"
            done
        fi
    else
        echo "  ❌ Ollama is not responding"
    fi
    echo ""
}

# Function to check agent responsiveness
test_agent() {
    echo "🧪 Testing Agent Responsiveness:"
    cd /Users/robbiepasquale/dspy_stuff/docker/lightweight
    timeout 5 docker-compose exec dspy-agent dspy-agent tree /app/test_project 2>/dev/null && echo "  ✅ Agent is responsive" || echo "  ⚠️  Agent is slow/unresponsive"
    echo ""
}

# Main monitoring loop
while true; do
    clear
    echo "🎯 DSPy Agent Activity Monitor - $(date)"
    echo "=========================================="
    echo ""
    
    show_status
    check_ollama
    show_recent_logs
    test_agent
    
    echo "Press Ctrl+C to stop monitoring..."
    sleep 5
done
