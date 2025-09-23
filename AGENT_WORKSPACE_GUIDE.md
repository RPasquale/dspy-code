# DSPy Code Agent - Isolated Workspace Guide

The Blampert agent now operates in a **completely isolated environment** that doesn't pollute your main repository. This guide explains how to set up and use the agent's workspace.

## 🏗️ Architecture Overview

```
~/.blampert_workspace/          # Agent's isolated workspace
├── venv/                      # Agent's virtual environment
├── projects/                  # Agent's training projects
│   ├── calculator/           # Calculator project with TODOs
│   ├── math_operations/      # Math operations project
│   └── data_processor/       # Data processing project
├── data/                     # Agent's database files
├── streaming/                # Agent's streaming engine
├── logs/                     # Agent's log files
├── models/                   # Agent's trained models
└── agent_config.json         # Agent configuration
```

## 🚀 Quick Start

### 1. Create Agent Workspace

```bash
# Using CLI
dspy-code workspace create

# Or using Python directly
python scripts/agent_workspace_manager.py
```

### 2. Train the Agent

```bash
# Train on all projects
dspy-code train

# Train on specific project
dspy-code train --project calculator

# Train with more iterations
dspy-code train --project calculator --iterations 5
```

### 3. Monitor Agent Activity

```bash
# Start real-time monitoring
dspy-code monitor

# Start dashboard
dspy-code dashboard
```

## 🔧 Workspace Management

### Create Workspace
```bash
# Default location (~/.blampert_workspace)
dspy-code workspace create

# Custom location
dspy-code workspace create --path /path/to/custom/workspace
```

### Get Workspace Info
```bash
dspy-code workspace info
```

### Clean Workspace
```bash
dspy-code workspace clean
```

## 📁 Project Structure

Each training project contains:

- **Main Python file** with TODO items for the agent to implement
- **Test file** to validate implementations
- **Requirements file** with dependencies
- **README** with project description and success criteria

### Example Project: Calculator

```python
class Calculator:
    def add(self, a: float, b: float) -> float:
        """
        Add two numbers.
        """
        # TODO: Implement addition
        result = 0  # Replace with actual implementation
        self.history.append(f"{a} + {b} = {result}")
        return result
```

The agent will:
1. **Analyze** the TODO items
2. **Generate** real code using DSPy + LLM
3. **Implement** the code in the file
4. **Test** the implementation
5. **Fix** any issues and improve

## 🤖 Agent Training Process

### 1. Project Analysis
- Agent scans for TODO comments
- Identifies methods that need implementation
- Analyzes test requirements

### 2. Code Generation
- Uses DSPy signatures for structured generation
- Leverages LLM (Ollama, OpenAI, etc.) for code creation
- Generates production-ready Python code

### 3. Implementation
- **Actually writes code** to files (not simulated!)
- Replaces TODO placeholders with real implementations
- Maintains proper code structure and documentation

### 4. Testing & Validation
- Runs pytest on generated code
- Analyzes test results
- Identifies failures and issues

### 5. Learning & Improvement
- Records actions and results in RedDB
- Gets feedback on success/failure
- Improves over multiple iterations

## 🎯 Available Training Projects

### 1. Calculator (`calculator/`)
- **Skills**: Basic arithmetic operations
- **TODO Items**: 4 methods (add, subtract, multiply, divide)
- **Difficulty**: Beginner

### 2. Math Operations (`math_operations/`)
- **Skills**: Advanced mathematical functions
- **TODO Items**: 4 methods (factorial, fibonacci, prime_check, gcd)
- **Difficulty**: Intermediate

### 3. Data Processor (`data_processor/`)
- **Skills**: Data processing and analysis
- **TODO Items**: 4 methods (load_csv, clean_data, calculate_statistics, save_processed_data)
- **Difficulty**: Intermediate

## 🔄 Streaming & Monitoring

The agent has its own streaming engine that publishes:

- **Agent Actions**: Code generation, file modifications, test runs
- **Learning Events**: Success/failure feedback, improvement metrics
- **Code Generation**: Generated code snippets and implementations
- **Test Results**: Test outcomes and performance metrics

### Real-time Monitoring
```bash
# Start monitoring system
dspy-code monitor

# View in browser
open http://localhost:8080
```

### Metrics
The status server exposes operational metrics at `/metrics`:

```json
{
  "lm_circuit": {"open": false, "next_retry_sec": 0.0},
  "rl": {"avg_reward": 0.42, "steps": 1200},
  "dlq": {"total": 3, "by_topic": {"agent.metrics": 2, "logs.ctx.backend": 1}, "last_ts": 1726789012.12},
  "bus": {
    "dlq_total": 3,
    "topics": {"agent.results.backend": [0, 1]},
    "groups": {"agent.results": {"worker-A": [0], "worker-B": [0]}}
  }
}
```

Notes:
- `lm_circuit`: LLM adapter circuit breaker state.
- `dlq`: Dead-letter queue counts by topic.
- `bus`: LocalBus queue depths and DLQ total. A background writer also saves snapshots to `.dspy_reports/bus_metrics.json` and history to `.dspy_reports/bus_metrics.jsonl`.

## 🗄️ Database & Storage

The agent uses its own isolated database:

- **RedDB**: For action records and learning data
- **Redis**: For caching and real-time data
- **SQLite**: For persistent storage
- **JSON**: For configuration and results

## 🛠️ Development Workflow

### For Agent Development
```bash
# 1. Create workspace
dspy-code workspace create

# 2. Train agent
dspy-code train --project calculator

# 3. Monitor progress
dspy-code monitor

# 4. Check results
dspy-code workspace info
```

### For Adding New Projects
1. Create project directory in `~/.blampert_workspace/projects/`
2. Add Python file with TODO items
3. Add test file
4. Add requirements.txt
5. Add README.md
6. Train agent on new project

## 🎓 Learning & Improvement

The agent learns through:

- **Reinforcement Learning**: Success/failure feedback
- **Code Analysis**: Understanding patterns and best practices
- **Test Results**: Learning from test failures and successes
- **Iterative Improvement**: Multiple attempts with feedback

### Learning Metrics
- **Success Rate**: Percentage of successful implementations
- **Test Pass Rate**: Percentage of tests that pass
- **Code Quality**: Documentation, error handling, structure
- **Learning Progress**: Improvement over iterations

## 🔧 Configuration

Agent configuration is stored in `~/.blampert_workspace/agent_config.json`:

```json
{
  "workspace": {
    "root": "~/.blampert_workspace",
    "projects": "~/.blampert_workspace/projects"
  },
  "agent": {
    "name": "BlampertAgent",
    "model": "llama3.1:8b",
    "learning_rate": 0.001,
    "max_iterations": 5
  },
  "streaming": {
    "enabled": true,
    "port": 9090
  }
}
```

## 🚨 Troubleshooting

### Workspace Not Found
```bash
# Create workspace first
dspy-code workspace create
```

### Training Fails
```bash
# Check workspace exists
dspy-code workspace info

# Check project exists
ls ~/.blampert_workspace/projects/
```

### Virtual Environment Issues
```bash
# Activate agent environment
source ~/.blampert_workspace/venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

## 🎉 Benefits of Isolated Workspace

✅ **No Repository Pollution**: Agent work doesn't affect your main codebase  
✅ **Complete Isolation**: Agent has its own environment and dependencies  
✅ **Independent Learning**: Agent can experiment without consequences  
✅ **Scalable**: Multiple agents can run in parallel  
✅ **Clean Separation**: Clear boundary between agent and user code  
✅ **Easy Cleanup**: Remove entire workspace if needed  
✅ **Portable**: Workspace can be moved or backed up  

## 📚 Next Steps

1. **Create your first workspace**: `blampert workspace create`
2. **Train the agent**: `blampert train --project calculator`
3. **Monitor progress**: `blampert monitor`
4. **Add custom projects**: Create new projects in the workspace
5. **Scale up**: Train on multiple projects for better learning

The agent is now completely self-contained and ready to learn!
