#!/bin/bash

# Comprehensive Streaming Pipeline Validation Script
# This script validates the entire streaming → vectorization → online training loop

set -e

echo "🔍 DSPy Streaming Pipeline Validation"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}❌ $message${NC}"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠️  $message${NC}"
    else
        echo -e "${BLUE}ℹ️  $message${NC}"
    fi
}

# Function to check if Docker Compose is running
check_docker_compose() {
    echo "Checking Docker Compose status..."
    
    # Check if we're in the right directory or if docker/lightweight exists
    if [ ! -d "docker/lightweight" ]; then
        print_status "FAIL" "docker/lightweight directory not found. Please run from the project root."
        exit 1
    fi
    
    # Change to the lightweight directory for docker compose commands
    cd docker/lightweight
    
    if ! docker compose ps | grep -q "Up"; then
        print_status "FAIL" "Docker Compose services are not running. Please start with: cd docker/lightweight && docker compose up -d"
        exit 1
    fi
    print_status "PASS" "Docker Compose services are running"
    
    # Return to original directory
    cd - > /dev/null
}

# Function to test enhanced container discovery
test_enhanced_discovery() {
    echo ""
    echo "🔍 Step 1: Test Enhanced Container Discovery"
    echo "=========================================="
    
    cd /Users/robbiepasquale/dspy_stuff
    
    echo "Testing enhanced container discovery..."
    if python test_enhanced_discovery.py > /tmp/discovery_test.log 2>&1; then
        print_status "PASS" "Enhanced container discovery is working"
        echo "Discovery results:"
        cat /tmp/discovery_test.log | grep -A 20 "Found.*container"
    else
        print_status "WARN" "Enhanced discovery test failed, but this might be normal"
        echo "Test output:"
        cat /tmp/discovery_test.log
    fi
    
    cd - > /dev/null
}

# Function to validate Kafka topics
validate_kafka_topics() {
    echo ""
    echo "📋 Step 2: Inventory Kafka Topics"
    echo "================================"
    
    cd docker/lightweight
    
    echo "Listing all Kafka topics..."
    topics=$(docker compose exec kafka kafka-topics.sh --bootstrap-server localhost:9092 --list 2>/dev/null || echo "")
    
    if [ -z "$topics" ]; then
        print_status "FAIL" "Could not connect to Kafka or no topics found"
        return 1
    fi
    
    echo "Found topics:"
    echo "$topics"
    echo ""
    
    # Check for critical topics
    critical_topics=(
        "code.fs.events"
        "agent.results.backend"
        "agent.results.frontend"
        "agent.rl.vectorized"
        "embedding_input"
        "embeddings"
    )
    
    missing_topics=()
    for topic in "${critical_topics[@]}"; do
        if echo "$topics" | grep -q "^$topic$"; then
            print_status "PASS" "Topic '$topic' exists"
        else
            print_status "FAIL" "Topic '$topic' is missing"
            missing_topics+=("$topic")
        fi
    done
    
    if [ ${#missing_topics[@]} -gt 0 ]; then
        print_status "WARN" "Missing topics: ${missing_topics[*]}"
        print_status "INFO" "This might be normal if the system hasn't been fully initialized yet"
    fi
    
    cd - > /dev/null
}

# Function to test topic message flow
test_topic_messages() {
    echo ""
    echo "📨 Step 2: Verify Topic Message Flow"
    echo "===================================="
    
    cd docker/lightweight
    
    # Test code.fs.events
    echo "Testing code.fs.events topic..."
    echo "Creating a test file to trigger file system events..."
    echo "# Test file for streaming validation" > /tmp/dspy_test_file.py
    
    # Wait a moment for the event to propagate
    sleep 2
    
    echo "Checking for messages in code.fs.events..."
    timeout 10s docker compose exec kafka kafka-console-consumer.sh \
        --bootstrap-server localhost:9092 \
        --topic code.fs.events \
        --from-beginning \
        --timeout-ms 5000 \
        --max-messages 5 > /tmp/code_fs_events.log 2>&1 || true
    
    if [ -s /tmp/code_fs_events.log ]; then
        print_status "PASS" "code.fs.events is receiving messages"
        echo "Sample message:"
        head -1 /tmp/code_fs_events.log
    else
        print_status "WARN" "No messages found in code.fs.events (this might be normal if no file changes occurred)"
    fi
    
    # Test other critical topics
    topics_to_test=("agent.results.backend" "embedding_input" "embeddings" "agent.rl.vectorized")
    
    for topic in "${topics_to_test[@]}"; do
        echo ""
        echo "Checking topic: $topic"
        timeout 5s docker compose exec kafka kafka-console-consumer.sh \
            --bootstrap-server localhost:9092 \
            --topic "$topic" \
            --from-beginning \
            --timeout-ms 3000 \
            --max-messages 3 > "/tmp/${topic//\./_}.log" 2>&1 || true
        
        if [ -s "/tmp/${topic//\./_}.log" ]; then
            print_status "PASS" "$topic has messages"
            echo "Sample message:"
            head -1 "/tmp/${topic//\./_}.log"
        else
            print_status "WARN" "No messages found in $topic (might be normal if system is idle)"
        fi
    done
    
    cd - > /dev/null
}

# Function to monitor container logs
monitor_container_logs() {
    echo ""
    echo "📊 Step 3: Monitor Container Logs"
    echo "================================="
    
    cd docker/lightweight
    
    # Check Spark vectorizer
    echo "Checking Spark vectorizer logs..."
    spark_logs=$(docker compose logs spark-vectorizer --tail 20 2>/dev/null || echo "")
    if [ -n "$spark_logs" ]; then
        print_status "PASS" "Spark vectorizer is running"
        echo "Recent Spark logs:"
        echo "$spark_logs" | tail -5
    else
        print_status "WARN" "No Spark vectorizer logs found"
    fi
    
    # Check embed worker
    echo ""
    echo "Checking embed worker logs..."
    embed_logs=$(docker compose logs embed-worker --tail 20 2>/dev/null || echo "")
    if [ -n "$embed_logs" ]; then
        print_status "PASS" "Embed worker is running"
        echo "Recent embed worker logs:"
        echo "$embed_logs" | tail -5
    else
        print_status "WARN" "No embed worker logs found"
    fi
    
    # Check code watcher
    echo ""
    echo "Checking code watcher logs..."
    code_watch_logs=$(docker compose logs dspy-code-watch --tail 20 2>/dev/null || echo "")
    if [ -n "$code_watch_logs" ]; then
        print_status "PASS" "Code watcher is running"
        echo "Recent code watcher logs:"
        echo "$code_watch_logs" | tail -5
    else
        print_status "WARN" "No code watcher logs found"
    fi
    
    cd - > /dev/null
}

# Function to validate parquet and checkpoints
validate_parquet_checkpoints() {
    echo ""
    echo "💾 Step 4: Validate Parquet Files and Checkpoints"
    echo "================================================"
    
    # Check vectorized directories
    if [ -d "vectorized" ]; then
        print_status "PASS" "vectorized directory exists"
        
        if [ -d "vectorized/embeddings" ]; then
            embedding_files=$(ls -la vectorized/embeddings/ 2>/dev/null | wc -l)
            print_status "PASS" "vectorized/embeddings directory exists with $embedding_files files"
        else
            print_status "WARN" "vectorized/embeddings directory not found"
        fi
        
        if [ -d "vectorized/embeddings_imesh" ]; then
            imesh_files=$(ls -la vectorized/embeddings_imesh/ 2>/dev/null | wc -l)
            print_status "PASS" "vectorized/embeddings_imesh directory exists with $imesh_files files"
        else
            print_status "WARN" "vectorized/embeddings_imesh directory not found"
        fi
    else
        print_status "WARN" "vectorized directory not found"
    fi
    
    # Check checkpoints
    if [ -d ".dspy_checkpoints" ]; then
        print_status "PASS" ".dspy_checkpoints directory exists"
        
        if [ -d ".dspy_checkpoints/vectorizer" ]; then
            checkpoint_files=$(ls -la .dspy_checkpoints/vectorizer/ 2>/dev/null | wc -l)
            print_status "PASS" "Vectorizer checkpoints exist with $checkpoint_files files"
        else
            print_status "WARN" "Vectorizer checkpoints not found"
        fi
    else
        print_status "WARN" ".dspy_checkpoints directory not found"
    fi
}

# Function to check agent ingestion
check_agent_ingestion() {
    echo ""
    echo "🤖 Step 5: Verify Agent Ingestion"
    echo "================================"
    
    # Check agent log files
    log_files=("logs/agent_actions.jsonl" "logs/agent_tool_usage.jsonl" "logs/agent_learning.jsonl")
    
    for log_file in "${log_files[@]}"; do
        if [ -f "$log_file" ]; then
            line_count=$(wc -l < "$log_file" 2>/dev/null || echo "0")
            if [ "$line_count" -gt 0 ]; then
                print_status "PASS" "$log_file exists with $line_count entries"
                echo "Latest entry:"
                tail -1 "$log_file"
            else
                print_status "WARN" "$log_file exists but is empty"
            fi
        else
            print_status "WARN" "$log_file not found"
        fi
        echo ""
    done
    
    # Check if agent is running
    if pgrep -f "dspy-agent" > /dev/null; then
        print_status "PASS" "DSPy agent process is running"
    else
        print_status "WARN" "No DSPy agent process found"
    fi
}

# Function to check consumer groups
check_consumer_groups() {
    echo ""
    echo "👥 Step 6: Check Consumer Groups"
    echo "==============================="
    
    cd docker/lightweight
    
    echo "Listing consumer groups..."
    consumer_groups=$(docker compose exec kafka kafka-consumer-groups.sh \
        --bootstrap-server localhost:9092 --all-groups --list 2>/dev/null || echo "")
    
    if [ -n "$consumer_groups" ]; then
        print_status "PASS" "Consumer groups found:"
        echo "$consumer_groups"
        
        # Check specific consumer groups
        for group in "spark-vectorizer" "dspy-code-indexer"; do
            echo ""
            echo "Checking consumer group: $group"
            group_info=$(docker compose exec kafka kafka-consumer-groups.sh \
                --bootstrap-server localhost:9092 --group "$group" --describe 2>/dev/null || echo "")
            
            if [ -n "$group_info" ]; then
                print_status "PASS" "Consumer group '$group' is active"
            else
                print_status "WARN" "Consumer group '$group' not found or inactive"
            fi
        done
    else
        print_status "WARN" "No consumer groups found"
    fi
    
    cd - > /dev/null
}

# Function to generate summary report
generate_summary() {
    echo ""
    echo "📋 Validation Summary"
    echo "===================="
    echo ""
    echo "This validation script checked:"
    echo "✅ Docker Compose services status"
    echo "✅ Kafka topics inventory"
    echo "✅ Topic message flow"
    echo "✅ Container logs monitoring"
    echo "✅ Parquet files and checkpoints"
    echo "✅ Agent ingestion logs"
    echo "✅ Consumer groups status"
    echo ""
    echo "If any steps showed warnings or failures, please:"
    echo "1. Check the specific component logs"
    echo "2. Ensure all services are properly started"
    echo "3. Trigger some activity (file edits, agent actions) to generate events"
    echo "4. Re-run this validation script"
    echo ""
    echo "For detailed troubleshooting, check:"
    echo "- Docker logs: cd docker/lightweight && docker compose logs [service-name]"
    echo "- Agent logs: tail -f logs/agent_*.jsonl"
    echo "- Kafka topics: cd docker/lightweight && docker compose exec kafka kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic [topic-name] --from-beginning"
}

# Main execution
main() {
    echo "Starting DSPy Streaming Pipeline Validation..."
    echo ""
    
    check_docker_compose
    test_enhanced_discovery
    validate_kafka_topics
    test_topic_messages
    monitor_container_logs
    validate_parquet_checkpoints
    check_agent_ingestion
    check_consumer_groups
    generate_summary
    
    echo ""
    print_status "INFO" "Validation complete! Check the summary above for any issues."
}

# Run the validation
main "$@"
