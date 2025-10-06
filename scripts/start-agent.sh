#!/bin/bash
# Simple agent launcher with better error handling and user guidance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 DSPy Agent Launcher${NC}"
echo "====================="
echo ""

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    echo -e "${RED}❌ Please run this script from the project root directory${NC}"
    exit 1
fi

# Check if virtual environment exists
if [[ ! -d ".venv" ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating one...${NC}"
    uv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Install dependencies if needed
echo "Installing dependencies..."
uv sync --quiet
echo -e "${GREEN}✅ Dependencies ready${NC}"

# Check if Ollama is available
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Ollama is available${NC}"
    
    # Check if deepseek-coder:1.3b model is available
    if ollama list | grep -q "deepseek-coder:1.3b"; then
        echo -e "${GREEN}✅ deepseek-coder:1.3b model is ready${NC}"
    else
        echo -e "${YELLOW}⚠️  deepseek-coder:1.3b model not found. You can pull it with:${NC}"
        echo "  ollama pull deepseek-coder:1.3b"
        echo ""
        echo -e "${YELLOW}Continuing without the model...${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Ollama not found. You can install it from: https://ollama.com/download${NC}"
    echo -e "${YELLOW}Or use OpenAI-compatible endpoints by setting environment variables${NC}"
fi

# Create logs directory
mkdir -p logs
echo -e "${GREEN}✅ Logs directory ready${NC}"

echo ""
echo -e "${GREEN}🎯 DSPy Agent is ready!${NC}"
echo ""
echo "Starting the agent with basic configuration..."
echo ""

# Set environment variables for better experience
export DSPY_AUTO_TRAIN=false  # Disable auto-training to avoid threading issues
export DSPY_LOG_LEVEL=INFO    # Set reasonable log level

# Start the agent
echo -e "${BLUE}Starting DSPy Agent...${NC}"
echo ""

# Run the agent with error handling
if uv run dspy-agent --workspace "$(pwd)" 2>&1; then
    echo ""
    echo -e "${GREEN}✅ Agent session completed successfully${NC}"
else
    echo ""
    echo -e "${RED}❌ Agent session ended with errors${NC}"
    echo ""
    echo -e "${YELLOW}💡 Troubleshooting tips:${NC}"
    echo "1. Check if Ollama is running: ollama list"
    echo "2. Try pulling the model: ollama pull deepseek-coder:1.3b"
    echo "3. Check logs in the logs/ directory"
    echo "4. Run the simple test: uv run python scripts/test_agent_simple.py"
    echo ""
    exit 1
fi
