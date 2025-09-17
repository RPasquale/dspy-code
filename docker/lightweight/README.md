## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
  - [Install (pip/pipx)](#install-pippipx)
  - [Quick Start (source checkout)](#quick-start-source-checkout)
- [Usage](#usage)
  - [Interactive Commands](#interactive-commands)
  - [Learning & Persistence](#learning--persistence)
- [Configuration](#configuration)
  - [LLM Configuration](#llm-configuration)
  - [Reinforcement Learning](#reinforcement-learning)
- [Deployment](#deployment)
  - [Full Stack: Kafka + Spark + Ollama + Agent (Docker)](#full-stack-kafka--spark--ollama--agent-docker)
  - [Streaming & Kafka](#streaming--kafka)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Overview

DSPy Agent is an intelligent coding assistant that enhances your development workflow through:

### 🤖 **AI-Powered Code Analysis**
- Repository-wide code understanding and context extraction
- Intelligent log analysis and key event detection
- Semantic code search and vector-based retrieval

### 🔧 **Smart Tool Suggestions**
- Context-aware action recommendations (grep, indexing, planning, etc.)
- Safe execution with built-in cooldowns and safeguards
- Extensible tool ecosystem for custom workflows

### 🎯 **Continuous Learning**
- Online learning through bandit policies during interactive sessions
- Offline reinforcement learning with toolchain feedback (tests/lint/build)
- Persistent learning state across sessions

### 🚀 **Production-Ready Stack**
- Full streaming architecture with Kafka + Spark + Ollama
- Docker-based deployment with auto-scaling capabilities
- Enterprise-grade logging and monitoring

**Default LLM:** `qwen3:1.7b` via Ollama (configurable for OpenAI-compatible endpoints)

## Installation

### Install (pip/pipx)

The package name is `dspy-code`. Install it once, then run `dspy-agent` from any project.

```bash
# CLI friendly install
pipx install dspy-code

# or reuse an existing Python environment
pip install dspy-code

# Point the agent at your repo (defaults to current working dir)
dspy-agent --workspace $(pwd)
```

Install [Ollama](https://ollama.com/download) (or bring your own OpenAI-compatible endpoint) and pull a model before first run:

```bash
ollama pull qwen3:1.7b
```

## Quick Start (source checkout)

```bash
pip install uv
uv sync

# Launch the agent from source
uv run dspy-agent
```

You'll see "RL suggested tool: …" before actions, and a session summary with suggested next steps.

## Usage

### Interactive Commands

The agent provides a rich set of interactive commands for code exploration and analysis:

#### 📋 **Context & Planning**
- `ctx` — Extract key log events (with de-noised, collapsed stack traces)
- `plan <task>` — Generate intelligent task plans with commands and risk analysis

#### 🔍 **Code Search & Analysis**
- `grep <regex>` — Fast regex-based code search across the repository
- `index` — Build comprehensive code index for semantic search
- `esearch --q '<query>'` — Semantic search over indexed code
- `tree [path]` — Display directory structure with intelligent filtering

#### 🧠 **AI-Powered Intelligence**
- `emb-index` — Build embeddings index (auto-builds on first miss)
- `vretr --query '<text>'` — Vector-based code retrieval
- `intel --query '<text>'` — Combined knowledge base and vector evidence search

#### 📁 **File Operations**
- `ls [path]` — List directory contents with smart formatting
- `cat <file>` — Display file contents with syntax highlighting

### Learning & Persistence

The agent continuously learns from your interactions:
- **State Persistence**: Learning state saved to `.dspy_rl_state.json`
- **Event Logging**: Per-step rewards logged to `.dspy_rl_events.jsonl`
- **Streaming Integration**: Events streamed via Kafka topic `agent.learning`
- **Cross-Session Learning**: Knowledge accumulates across multiple sessions

## Configuration

### LLM Configuration

Choose between Ollama (recommended for local development) or OpenAI-compatible APIs:

#### Ollama Setup (Recommended)
```bash
export USE_OLLAMA=true
export OLLAMA_MODEL=qwen3:1.7b

# Install Ollama and pull the model
ollama pull qwen3:1.7b
```

#### OpenAI-Compatible API
```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
```

**Notes:**
- The agent uses native Ollama endpoints by default (`http://localhost:11434`)
- LLM timeouts are capped (~30s) with low retries to avoid stalls
- Legacy 600s timeout lines are filtered from `ctx` output

### Reinforcement Learning

Two cooperating paths:

1) Online RL (in-session)
- Bandit suggests a safe tool from: `context, codectx, grep, esearch, plan, tree, ls, index, emb-index, intel, vretr`.
- It auto-fills reasonable args from your query.
- Cooldowns for heavy actions (e.g., `emb-index`: 600s) keep sessions snappy.
- Outcomes are scored and persisted so the policy improves across runs.

2) Offline RL CLI (toolchain)
- Trains a policy over actions: `run_tests, lint, build` using verifiers.
- Observations include Kafka-derived context if available (`logs.ctx.*`), otherwise local logs.

```bash
# Optional extras for neural RL
pip install '.[rl]'

# Config stub
uv run dspy-agent rl config init --out .dspy_rl.json

# Bandit (epsilon-greedy)
uv run dspy-agent rl train --workspace . --steps 300

# Neural REINFORCE (uses PufferLib vectorization if available)
uv run dspy-agent rl train --workspace . --steps 1000 --neural --n-envs 4

# PuffeRL PPO shell (example)
uv run dspy-agent rl ppo --workspace . --n-envs 8 --total-steps 200000
```

## Deployment

### Full Stack: Kafka + Spark + Ollama + Agent (Docker)

This spins up Zookeeper/Kafka, Spark to transform `logs.raw.*` → `logs.ctx.*`, Ollama, and agent services.

```bash
# Write Dockerfile + compose (pip-installing dspy-code inside the image)
dspy-agent lightweight_init \
  --workspace $(pwd) \
  --logs ./logs \
  --out-dir docker/lightweight \
  --install-source pip \
  --db auto

export DOCKER_BUILDKIT=1
docker compose -f docker/lightweight/docker-compose.yml build --pull
docker compose -f docker/lightweight/docker-compose.yml up -d --remove-orphans
docker compose -f docker/lightweight/docker-compose.yml ps

# Use the stack from your shell
export KAFKA_BOOTSTRAP=localhost:9092
dspy-agent --workspace $(pwd)
```

> Tip: use `--install-source local` when iterating on the repo itself; it copies the current checkout into the Docker build context. Add `--pip-spec` to pin a specific wheel or git ref for pip installs.

Space‑friendly rebuilds:
```bash
export DOCKER_BUILDKIT=1
docker compose -f docker/lightweight/docker-compose.yml build --pull
docker compose -f docker/lightweight/docker-compose.yml up -d --no-deps dspy-agent dspy-worker dspy-worker-backend dspy-worker-frontend dspy-router

# Safe cleanup (keeps named volumes like Ollama models)
docker image prune -f
docker builder prune -f
```

## LLM Configuration

Ollama (recommended for local):
```bash
export USE_OLLAMA=true
export OLLAMA_MODEL=qwen3:1.7b
```

OpenAI-compatible API:
```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
```

Notes:
- The agent uses native Ollama endpoints by default (`http://localhost:11434`).
- LLM timeouts are capped (~30s) with low retries to avoid stalls; legacy 600s timeout lines are filtered from `ctx`.

## Streaming & Kafka

- Spark job in the stack reads `logs.raw.*` and publishes JSON `{"ctx": [...]}` to `logs.ctx.*` topics.
- The offline RL env consumes these to add context features.
- The agent publishes per-step learning events to `agent.learning` (and `.dspy_rl_events.jsonl` fallback).
- The background trainer consumes `agent.learning` and updates `.dspy_rl_state.json` continuously.

Environment:
```bash
export KAFKA_BOOTSTRAP=localhost:9092
export RL_BACKGROUND_STEPS=50   # background bandit updates per batch (optional)
```

## Architecture

DSPy Agent is built with a modular architecture designed for scalability and extensibility:

### 🏗️ **Core Modules**

#### `dspy_agent.embedding/`
- **Purpose**: Code indexing and vector-based search capabilities
- **Key Components**: Embeddings generation, semantic search, index management
- **Features**: Auto-building indexes, persistent storage, efficient retrieval

#### `dspy_agent.code_tools/`
- **Purpose**: Code analysis, manipulation, and transformation utilities
- **Key Components**: AST parsing, diff generation, patch application, code evaluation
- **Features**: Safe code modifications, context extraction, symbol resolution

#### `dspy_agent.agents/`
- **Purpose**: Agent orchestration and workflow management
- **Key Components**: Router workers, adapters, knowledge management
- **Features**: Multi-agent coordination, task delegation, runtime optimization

#### `dspy_agent.skills/`
- **Purpose**: DSPy-based AI modules for specific coding tasks
- **Key Components**: Task planning, code context, file location, patch verification
- **Features**: Modular AI capabilities, composable workflows, extensible skill system

#### `dspy_agent.streaming/`
- **Purpose**: Real-time data processing and event streaming
- **Key Components**: Kafka integration, Spark jobs, log processing, background trainers
- **Features**: Scalable data pipelines, real-time learning, distributed processing

#### `dspy_agent.training/`
- **Purpose**: Model training and deployment infrastructure
- **Key Components**: GEPA modules, deployment helpers, dataset generation
- **Features**: Automated training pipelines, model versioning, evaluation frameworks

#### `dspy_agent.rl/`
- **Purpose**: Reinforcement learning environment and policies
- **Key Components**: Bandit algorithms, neural trainers, PufferLib integration
- **Features**: Online/offline learning, policy optimization, reward engineering

### 🔄 **Data Flow Architecture**

```
User Input → CLI Interface → Agent Skills → Code Tools → LLM Integration
     ↓                                                        ↑
Learning Events → Streaming Pipeline → RL Training → Policy Updates
```

### 🎯 **Design Principles**

- **Modularity**: Each component can be used independently or as part of the full stack
- **Extensibility**: Plugin architecture allows custom skills and tools
- **Scalability**: Streaming architecture supports high-throughput scenarios
- **Safety**: Built-in safeguards prevent destructive operations
- **Observability**: Comprehensive logging and monitoring throughout

## Troubleshooting

### Common Issues and Solutions

#### 🔧 **Kafka Connection Issues**
**Problem**: Agent can't connect to Kafka
```
ERROR: Failed to connect to Kafka bootstrap servers
```
**Solution**: 
- Ensure Kafka is running: `docker compose ps`
- Set environment variable: `export KAFKA_BOOTSTRAP=localhost:9092`
- The agent falls back to local logs when Kafka is unavailable

#### 🔍 **Vector Search Not Working**
**Problem**: No results from `intel` or `vretr` commands
```
No vector matches found for query
```
**Solution**:
- Run `emb-index` to build embeddings index manually
- Indexes auto-build on first miss, but may take time
- Check disk space for index storage

#### 🐳 **Docker Build Issues**
**Problem**: Docker containers fail to start or build
**Solutions**:
- **Space Issues**: Run cleanup commands
  ```bash
  docker image prune -f
  docker builder prune -f
  ```
- **Volume Issues**: Avoid `--volumes` flag to preserve Ollama models
- **Build Cache**: Use `export DOCKER_BUILDKIT=1` for better caching

#### 🤖 **LLM Connection Problems**
**Problem**: Agent can't connect to language model
**Solutions**:
- **Ollama**: Ensure service is running (`ollama serve`)
- **OpenAI API**: Check API key and base URL configuration
- **Timeouts**: Increase timeout if needed (default: 30s)

#### 📊 **Performance Issues**
**Problem**: Agent responses are slow
**Solutions**:
- Use smaller models (e.g., `qwen3:1.7b` instead of larger variants)
- Enable index caching for faster subsequent searches
- Adjust cooldown periods for heavy operations

#### 🔄 **Learning State Issues**
**Problem**: Agent doesn't remember previous interactions
**Solutions**:
- Check `.dspy_rl_state.json` file exists and is writable
- Verify `.dspy_rl_events.jsonl` is being updated
- Ensure workspace path is consistent across sessions

## License

See LICENSE for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Quick Start for Contributors

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/dspy-code.git
   cd dspy-code
   ```

2. **Set Up Development Environment**
   ```bash
   pip install uv
   uv sync
   ```

3. **Run Tests**
   ```bash
   uv run python -m unittest discover -s tests -v
   ```

4. **Test the Full Stack**
   ```bash
   uv run dspy-agent lightweight_init --workspace $(pwd) --logs ./logs --db auto
   docker compose -f docker/lightweight/docker-compose.yml build
   docker compose -f docker/lightweight/docker-compose.yml up -d
   ```

### Areas for Contribution

- 🐛 **Bug Fixes**: Check open issues for bugs
- 📚 **Documentation**: Improve guides, examples, and API docs
- 🔧 **New Skills**: Add DSPy modules for specific coding tasks
- 🤖 **LLM Integration**: Support for new language models
- 🔍 **Code Tools**: Enhanced analysis and manipulation utilities
- 📊 **Monitoring**: Better observability and debugging tools

## Support

### Getting Help

- 📖 **Documentation**: Check this README and [docs/](docs/) directory
- 🐛 **Bug Reports**: Open an issue with detailed information
- 💬 **Discussions**: Use GitHub Discussions for questions and ideas
- 📧 **Security Issues**: Email security concerns privately

### Issue Reporting Template

When reporting issues, please include:

```
**Environment:**
- Python version: 
- uv version:
- Ollama version (if applicable):
- Operating System:

**Command that failed:**
```bash
# exact command here
```

**Expected behavior:**
[What you expected to happen]

**Actual behavior:**
[What actually happened]

**Logs:**
```
# Include relevant logs here
# For Docker issues, include: docker compose ps
# For agent issues, include agent logs
```
```
