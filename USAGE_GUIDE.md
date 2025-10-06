# DSPy Agent Usage Guide

This comprehensive guide will help you get started with the DSPy Agent and use it effectively for your coding tasks.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation Options](#installation-options)
3. [Basic Usage](#basic-usage)
4. [Interactive Commands](#interactive-commands)
5. [LLM Configuration](#llm-configuration)
6. [Advanced Features](#advanced-features)
7. [Deployment Options](#deployment-options)
8. [Troubleshooting](#troubleshooting)
9. [Examples](#examples)

## Quick Start

### Option 1: Install and Use Anywhere
```bash
# Install the agent
pipx install dspy-code

# Install a local LLM (recommended)
ollama pull qwen3:1.7b

# Start the agent in your project
dspy-agent --workspace $(pwd)
```

### Option 2: From Source
```bash
# Clone and setup
git clone <your-repo>
cd dspy_stuff
pip install uv
uv sync

# Start the agent
uv run dspy-agent --workspace $(pwd)
```

## Installation Options

### Prerequisites
- Python 3.11 or higher
- `uv` package manager (recommended) or `pip`
- Optional: Docker (for full stack deployment)
- Optional: Ollama (for local LLM)

### Installation Methods

#### 1. Using pipx (Recommended)
```bash
pipx install dspy-code
```

#### 2. Using pip
```bash
pip install dspy-code
```

#### 3. From Source
```bash
git clone <repository>
cd dspy_stuff
pip install uv
uv sync
uv pip install --system -e .
```

## Basic Usage

### Starting the Agent
```bash
# Basic usage (uses current directory as workspace)
dspy-agent

# Specify workspace
dspy-agent --workspace /path/to/your/project

# With custom logs directory
dspy-agent --workspace /path/to/project --logs /path/to/logs
```

### Command Line Options
```bash
dspy-agent [OPTIONS]

Options:
  --workspace PATH     Set workspace directory (default: current directory)
  --logs PATH          Set logs directory (default: workspace/logs)
  --ollama/--no-ollama Use Ollama by default (default: True)
  --model TEXT         Override model (default: auto-detected)
  --base-url TEXT      Override base URL for LLM
  --api-key TEXT       API key for LLM
  --force-json         Force simple JSON outputs
  --structured         Prefer structured outputs
  --approval TEXT      Tool approval mode: auto|manual
  --help               Show help message
```

## Interactive Commands

Once you start the agent, you'll have access to a rich set of interactive commands:

### 📋 Context & Planning
- `ctx` - Extract key log events with de-noised stack traces
- `plan <task>` - Generate intelligent task plans with commands and risk analysis

### 🔍 Code Search & Analysis
- `grep <regex>` - Fast regex-based code search across the repository
- `index` - Build comprehensive code index for semantic search
- `esearch --q '<query>'` - Semantic search over indexed code
- `tree [path]` - Display directory structure with intelligent filtering

### 🧠 AI-Powered Intelligence
- `emb-index` - Build embeddings index (auto-builds on first miss)
- `vretr --query '<text>'` - Vector-based code retrieval
- `intel --query '<text>'` - Combined knowledge base and vector evidence search

### 📁 File Operations
- `ls [path]` - List directory contents with smart formatting
- `cat <file>` - Display file contents with syntax highlighting

### 🛠️ Code Editing
- `edit <description> --apply` - Propose and apply code changes
- `patch <file> <changes>` - Apply specific patches

### 🔄 Model Management
- `ollama off` - Disable Ollama
- `model gpt-4o-mini` - Switch to different model
- `model list` - List available models

## LLM Configuration

### Option 1: Ollama (Local, Recommended)
```bash
# Install Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen3:1.7b

# Start the agent (Ollama is default)
dspy-agent --workspace $(pwd)
```

### Option 2: OpenAI-Compatible API
```bash
# Set environment variables
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"

# Start the agent
dspy-agent --workspace $(pwd) --no-ollama
```

### Option 3: Custom Configuration
```bash
# Use specific model and endpoint
dspy-agent \
  --workspace $(pwd) \
  --no-ollama \
  --model "gpt-4o-mini" \
  --base-url "https://api.openai.com/v1" \
  --api-key "your-api-key"
```

## Advanced Features

### Reinforcement Learning
The agent learns from your interactions and improves over time:

```bash
# View learning state
cat .dspy_rl_state.json

# View learning events
cat .dspy_rl_events.jsonl
```

### Streaming Integration
For advanced users, the agent supports Kafka streaming:

```bash
# Generate lightweight stack with streaming
dspy-agent lightweight_init \
  --workspace $(pwd) \
  --logs ./logs \
  --out-dir docker/lightweight

# Deploy with Docker
cd docker/lightweight
docker compose up -d
```

### Database Integration
The agent uses RedDB for persistent storage:

```bash
# Initialize database
python -c "from dspy_agent.db import initialize_database; initialize_database()"
```

## Deployment Options

### Development Environment
```bash
# Quick development setup
./scripts/deploy-dev.sh

# This will:
# - Install dependencies
# - Set up development environment
# - Run tests
# - Start the agent
```

### Test Environment
```bash
# Comprehensive testing
./scripts/deploy-test.sh

# This will:
# - Run all tests
# - Validate integration
# - Test Docker stack generation
# - Generate test report
```

### Production Environment
```bash
# Production deployment
./scripts/deploy-prod.sh

# This will:
# - Confirm deployment
# - Generate production Docker stack
# - Deploy with monitoring
# - Set up log rotation
```

## Troubleshooting

### Common Issues

#### 1. "Module not found" errors
```bash
# Make sure you're in the right directory
cd /path/to/dspy_stuff

# Reinstall in development mode
uv pip install --system -e .
```

#### 2. Ollama connection issues
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

#### 3. Database initialization errors
```bash
# Clear existing database
rm -f .dspy_rl_state.json .dspy_rl_events.jsonl

# Reinitialize
python -c "from dspy_agent.db import initialize_database; initialize_database()"
```

#### 4. Docker issues
```bash
# Check Docker status
docker --version
docker compose version

# Clean up Docker resources
docker system prune -f
```

### Getting Help

1. **Check logs**: Look in the `logs/` directory for detailed error messages
2. **Run tests**: Use `./scripts/run_all_tests.py` to validate your setup
3. **Check configuration**: Verify your LLM configuration and environment variables
4. **Review documentation**: Check the API documentation in `docs/API.md`

## Examples

### Example 1: Fix Failing Tests
```bash
# Start the agent
dspy-agent --workspace $(pwd)

# In the interactive session:
> plan "fix the failing tests"
> grep "test_.*fail"
> edit "fix the failing test cases" --apply
```

### Example 2: Add New Feature
```bash
# Start the agent
dspy-agent --workspace $(pwd)

# In the interactive session:
> plan "add user authentication"
> esearch "authentication login"
> edit "implement user authentication system" --apply
```

### Example 3: Code Review
```bash
# Start the agent
dspy-agent --workspace $(pwd)

# In the interactive session:
> intel "review the API endpoints for security issues"
> grep "def.*api"
> edit "add input validation to API endpoints" --apply
```

### Example 4: Refactoring
```bash
# Start the agent
dspy-agent --workspace $(pwd)

# In the interactive session:
> plan "refactor the database layer"
> vretr --query "database connection management"
> edit "refactor database layer for better error handling" --apply
```

## Best Practices

### 1. Workspace Organization
- Keep your project in a clean directory structure
- Use version control (git) for your code
- Keep logs in a separate directory

### 2. Model Selection
- Use local models (Ollama) for privacy and speed
- Use cloud models for complex tasks requiring more context
- Experiment with different models for different tasks

### 3. Learning and Improvement
- Provide feedback on the agent's suggestions
- Review the learning state periodically
- Use the agent consistently to build up its knowledge

### 4. Security
- Never commit API keys to version control
- Use environment variables for sensitive configuration
- Review code changes before applying them

## Next Steps

1. **Start Simple**: Begin with basic commands like `plan` and `grep`
2. **Explore Features**: Try different interactive commands
3. **Customize**: Configure your preferred LLM and settings
4. **Scale Up**: Consider Docker deployment for production use
5. **Contribute**: Check out the contributing guidelines in `CONTRIBUTING.md`

## Support

- **Documentation**: Check `docs/` directory for detailed API documentation
- **Issues**: Report bugs and feature requests on GitHub
- **Community**: Join discussions in the project's community channels

Happy coding with DSPy Agent! 🚀
