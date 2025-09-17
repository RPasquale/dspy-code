# DSPy Agent API Documentation

This document provides comprehensive API documentation for DSPy Agent's core modules and classes.

## Table of Contents

- [Core CLI Interface](#core-cli-interface)
- [Agent Skills](#agent-skills)
- [Code Tools](#code-tools)
- [Embedding System](#embedding-system)
- [Reinforcement Learning](#reinforcement-learning)
- [Streaming & Kafka](#streaming--kafka)
- [Configuration](#configuration)

## Core CLI Interface

### Main CLI Entry Point

The primary interface is through the `dspy-agent` command-line tool.

```bash
dspy-agent [OPTIONS] COMMAND [ARGS]...
```

#### Global Options

- `--workspace PATH`: Set workspace directory (default: current directory)
- `--logs PATH`: Set logs directory (default: workspace/logs)
- `--verbose`: Enable verbose logging
- `--config PATH`: Path to configuration file

#### Main Commands

##### Interactive Mode
```bash
dspy-agent [--workspace PATH]
```
Starts interactive session with the following commands:

- `ctx` - Extract key log events
- `plan <task>` - Generate task plans
- `grep <pattern>` - Search code
- `index` - Build code index
- `emb-index` - Build embeddings index
- `esearch --q '<query>'` - Semantic search
- `vretr --query '<text>'` - Vector retrieval
- `intel --query '<text>'` - Knowledge search
- `tree [path]` - Display directory structure
- `ls [path]` - List directory contents
- `cat <file>` - Display file contents

##### Stack Management
```bash
dspy-agent lightweight_init [OPTIONS]
```
Generate Docker stack for production deployment.

**Options:**
- `--workspace PATH`: Target workspace
- `--logs PATH`: Logs directory
- `--out-dir PATH`: Output directory for generated files
- `--install-source {pip,local}`: Installation source
- `--pip-spec TEXT`: Pip package specification
- `--db {auto,redis,sqlite}`: Database backend

##### Reinforcement Learning
```bash
dspy-agent rl COMMAND [OPTIONS]
```

**RL Commands:**
- `train`: Train RL policy
- `eval`: Evaluate trained policy
- `config init`: Initialize RL configuration
- `ppo`: Run PuffeRL PPO training

## Agent Skills

DSPy Agent uses modular skills implemented as DSPy modules for specific tasks.

### TaskAgent

Plans and breaks down complex coding tasks.

```python
from dspy_agent.skills import TaskAgent, PlanTaskSig

class PlanTaskSig(dspy.Signature):
    """Generate a comprehensive plan for a coding task."""
    task: str = dspy.InputField(desc="The coding task to plan")
    context: str = dspy.InputField(desc="Relevant context and constraints")
    
    plan: str = dspy.OutputField(desc="Step-by-step plan")
    commands: str = dspy.OutputField(desc="Suggested commands to execute")
    risks: str = dspy.OutputField(desc="Potential risks and mitigations")

class TaskAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.ChainOfThought(PlanTaskSig)
    
    def forward(self, task: str, context: str = "") -> dspy.Prediction:
        return self.plan(task=task, context=context)
```

**Usage:**
```python
agent = TaskAgent()
result = agent.forward(
    task="Add user authentication to Flask app",
    context="Using SQLAlchemy ORM, need JWT tokens"
)
print(result.plan)
print(result.commands)
print(result.risks)
```

### CodeContext

Analyzes and provides context about code structures.

```python
from dspy_agent.skills import CodeContext, CodeContextSig

class CodeContextSig(dspy.Signature):
    """Analyze code and provide relevant context."""
    code: str = dspy.InputField(desc="Code to analyze")
    query: str = dspy.InputField(desc="Specific question about the code")
    
    analysis: str = dspy.OutputField(desc="Code analysis and insights")
    suggestions: str = dspy.OutputField(desc="Improvement suggestions")

class CodeContext(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(CodeContextSig)
    
    def forward(self, code: str, query: str = "") -> dspy.Prediction:
        return self.analyze(code=code, query=query)
```

### FileLocator

Locates relevant files based on descriptions or requirements.

```python
from dspy_agent.skills import FileLocator, FileLocatorSig

class FileLocatorSig(dspy.Signature):
    """Locate files relevant to a task or query."""
    query: str = dspy.InputField(desc="Description of what files are needed")
    file_list: str = dspy.InputField(desc="Available files in the project")
    
    relevant_files: str = dspy.OutputField(desc="List of relevant files")
    reasoning: str = dspy.OutputField(desc="Why these files are relevant")

class FileLocator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.locate = dspy.ChainOfThought(FileLocatorSig)
    
    def forward(self, query: str, file_list: str) -> dspy.Prediction:
        return self.locate(query=query, file_list=file_list)
```

### PatchVerifier

Verifies and validates code patches for safety and correctness.

```python
from dspy_agent.skills import PatchVerifier, PatchVerifierSig

class PatchVerifierSig(dspy.Signature):
    """Verify a code patch for safety and correctness."""
    patch: str = dspy.InputField(desc="Unified diff patch to verify")
    context: str = dspy.InputField(desc="Surrounding code context")
    
    is_safe: str = dspy.OutputField(desc="Whether the patch is safe to apply")
    issues: str = dspy.OutputField(desc="Potential issues or concerns")
    suggestions: str = dspy.OutputField(desc="Improvement suggestions")

class PatchVerifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.verify = dspy.ChainOfThought(PatchVerifierSig)
    
    def forward(self, patch: str, context: str = "") -> dspy.Prediction:
        return self.verify(patch=patch, context=context)
```

## Code Tools

### Code Search

Advanced code search capabilities with multiple backends.

```python
from dspy_agent.code_tools.code_search import (
    search_text,
    search_file,
    extract_context,
    python_extract_symbol,
    run_ast_grep
)

# Text-based search
results = search_text(
    pattern=r"class.*Agent",
    directory="./src",
    file_extensions=[".py"],
    max_results=50
)

# File-based search
file_results = search_file(
    filename="agent.py",
    directory="./src",
    recursive=True
)

# Extract context around matches
context = extract_context(
    file_path="./src/agent.py",
    line_number=42,
    context_lines=5
)

# Python symbol extraction
symbols = python_extract_symbol(
    file_path="./src/agent.py",
    symbol_name="TaskAgent"
)

# AST-based search (if ast-grep is available)
if ast_grep_available():
    ast_results = run_ast_grep(
        pattern="class $NAME",
        directory="./src"
    )
```

### Code Snapshot

Create comprehensive snapshots of code structure.

```python
from dspy_agent.code_tools.code_snapshot import build_code_snapshot

snapshot = build_code_snapshot(
    directory="./src",
    include_patterns=["*.py", "*.js", "*.ts"],
    exclude_patterns=["__pycache__", "node_modules"],
    max_file_size=1024*1024,  # 1MB limit
    include_git_info=True
)

print(f"Files: {len(snapshot['files'])}")
print(f"Total size: {snapshot['total_size']} bytes")
print(f"Git branch: {snapshot['git_info']['branch']}")
```

### Patch Operations

Apply and manage code patches safely.

```python
from dspy_agent.code_tools.patcher import apply_unified_patch, summarize_patch

# Apply a unified diff patch
patch_content = """
--- a/src/agent.py
+++ b/src/agent.py
@@ -10,6 +10,7 @@ class Agent:
     def __init__(self):
         self.skills = []
+        self.initialized = True
     
     def add_skill(self, skill):
         self.skills.append(skill)
"""

result = apply_unified_patch(
    patch_content=patch_content,
    base_directory="./",
    dry_run=False
)

if result["success"]:
    print("Patch applied successfully")
else:
    print(f"Patch failed: {result['error']}")

# Summarize patch changes
summary = summarize_patch(patch_content)
print(f"Files changed: {summary['files_changed']}")
print(f"Lines added: {summary['lines_added']}")
print(f"Lines removed: {summary['lines_removed']}")
```

## Embedding System

### Index Management

Build and manage code embeddings for semantic search.

```python
from dspy_agent.embedding.indexer import (
    build_index,
    save_index,
    load_index,
    semantic_search,
    tokenize
)

# Build embeddings index
index = build_index(
    directory="./src",
    include_patterns=["*.py", "*.md"],
    chunk_size=512,
    overlap=50,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Save index to disk
save_index(index, "./embeddings_index.pkl")

# Load existing index
loaded_index = load_index("./embeddings_index.pkl")

# Perform semantic search
results = semantic_search(
    index=loaded_index,
    query="authentication and user management",
    top_k=10,
    threshold=0.7
)

for result in results:
    print(f"File: {result['file']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content'][:100]}...")
    print("---")

# Tokenize text for analysis
tokens = tokenize("def authenticate_user(username, password):")
print(f"Tokens: {tokens}")
```

### Embeddings Index

Low-level embeddings index operations.

```python
from dspy_agent.embedding.embeddings_index import EmbeddingsIndex

# Create embeddings index
index = EmbeddingsIndex(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"  # or "cuda" for GPU
)

# Add documents
documents = [
    {"id": "doc1", "content": "User authentication with JWT tokens", "metadata": {"file": "auth.py"}},
    {"id": "doc2", "content": "Database connection and ORM setup", "metadata": {"file": "db.py"}},
]

index.add_documents(documents)

# Search similar documents
query = "login and security"
results = index.search(query, top_k=5)

for result in results:
    print(f"ID: {result['id']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content']}")
    print(f"Metadata: {result['metadata']}")
    print("---")

# Update document
index.update_document("doc1", {
    "content": "Enhanced user authentication with JWT and refresh tokens",
    "metadata": {"file": "auth.py", "updated": True}
})

# Remove document
index.remove_document("doc2")

# Save/load index
index.save("./my_index.pkl")
loaded_index = EmbeddingsIndex.load("./my_index.pkl")
```

## Reinforcement Learning

### RL Environment

The RL system provides environments for training coding agents.

```python
from dspy_agent.rl.rlkit import RLToolEnv, ToolAction, RewardConfig

# Create RL environment
config = RewardConfig(
    weights={"pass_rate": 1.0, "lint_score": 0.5},
    penalty_kinds=["blast_radius"],
    clamp01_kinds=["pass_rate"],
    scales={"blast_radius": (0, 10)}
)

env = RLToolEnv(
    workspace="./my_project",
    reward_config=config,
    timeout_sec=300
)

# Reset environment
obs = env.reset()
print(f"Initial observation: {obs}")

# Take action
action = ToolAction.RUN_TESTS  # or 0 for integer action
obs, reward, done, truncated, info = env.step(action)

print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Info: {info}")

# Available actions
print("Available actions:")
for action in ToolAction:
    print(f"  {action.name}: {action.value}")
```

### Bandit Policies

Various bandit algorithms for action selection.

```python
from dspy_agent.rl.rlkit import EpsilonGreedyBandit, UCB1Bandit, ThompsonSamplingBandit

# Epsilon-greedy bandit
epsilon_bandit = EpsilonGreedyBandit(
    n_actions=3,
    epsilon=0.1,
    initial_values=0.0
)

# UCB1 bandit
ucb_bandit = UCB1Bandit(
    n_actions=3,
    c=2.0,  # exploration parameter
    initial_values=0.0
)

# Thompson sampling bandit
thompson_bandit = ThompsonSamplingBandit(
    n_actions=3,
    alpha_prior=1.0,
    beta_prior=1.0
)

# Select action
action = epsilon_bandit.select_action()
print(f"Selected action: {action}")

# Update with reward
epsilon_bandit.update(action, reward=0.8)

# Get action values
values = epsilon_bandit.get_action_values()
print(f"Action values: {values}")

# Get policy statistics
stats = epsilon_bandit.get_stats()
print(f"Stats: {stats}")
```

### Training Interface

High-level training interface for RL policies.

```python
from dspy_agent.rl.rlkit import train_bandit_policy, train_neural_policy

# Train bandit policy
bandit_results = train_bandit_policy(
    workspace="./my_project",
    policy_type="epsilon-greedy",
    n_steps=1000,
    epsilon=0.1,
    save_path="./bandit_policy.pkl"
)

print(f"Final reward: {bandit_results['final_reward']}")
print(f"Best action: {bandit_results['best_action']}")

# Train neural policy (requires RL extras)
try:
    neural_results = train_neural_policy(
        workspace="./my_project",
        n_steps=5000,
        learning_rate=0.001,
        hidden_size=64,
        save_path="./neural_policy.pt"
    )
    print(f"Final loss: {neural_results['final_loss']}")
except ImportError:
    print("Neural training requires: pip install .[rl]")
```

## Streaming & Kafka

### Kafka Integration

Stream learning events and context through Kafka.

```python
from dspy_agent.streaming.streaming_kafka import KafkaProducer, KafkaConsumer
from dspy_agent.streaming.streaming_config import StreamingConfig

# Configure streaming
config = StreamingConfig(
    kafka_bootstrap_servers="localhost:9092",
    topics={
        "agent_learning": "agent.learning",
        "logs_context": "logs.ctx.backend"
    }
)

# Producer for sending events
producer = KafkaProducer(config)

# Send learning event
learning_event = {
    "timestamp": 1234567890,
    "action": "run_tests",
    "reward": 0.85,
    "context": {"workspace": "./my_project"},
    "metadata": {"session_id": "abc123"}
}

producer.send_event("agent_learning", learning_event)

# Consumer for receiving events
consumer = KafkaConsumer(config, ["agent.learning"])

for message in consumer.consume():
    print(f"Received: {message}")
    # Process message
    if message["action"] == "run_tests":
        print(f"Test reward: {message['reward']}")
```

### Log Processing

Process and analyze application logs.

```python
from dspy_agent.streaming.log_reader import extract_key_events, load_logs

# Load logs from directory
logs = load_logs("./logs", pattern="*.log")
print(f"Loaded {len(logs)} log files")

# Extract key events
events = extract_key_events(
    logs,
    event_types=["error", "warning", "performance"],
    time_window_hours=24,
    max_events=100
)

for event in events:
    print(f"[{event['timestamp']}] {event['level']}: {event['message']}")
    if event.get('stack_trace'):
        print(f"  Stack trace: {event['stack_trace'][:100]}...")

# Filter and process events
error_events = [e for e in events if e['level'] == 'error']
print(f"Found {len(error_events)} error events")

# Group by error type
from collections import defaultdict
error_groups = defaultdict(list)
for event in error_events:
    error_type = event.get('exception_type', 'unknown')
    error_groups[error_type].append(event)

for error_type, events in error_groups.items():
    print(f"{error_type}: {len(events)} occurrences")
```

## Configuration

### Settings Management

Centralized configuration management.

```python
from dspy_agent.config import get_settings, Settings

# Get current settings
settings = get_settings()

print(f"Workspace: {settings.workspace}")
print(f"LLM Model: {settings.model_name}")
print(f"Use Ollama: {settings.use_ollama}")
print(f"RL Steps: {settings.rl_background_steps}")

# Access RL configuration
print(f"RL Weights: {settings.rl_weights}")
print(f"RL Scales: {settings.rl_scales}")

# Override settings via environment variables
import os
os.environ["DSPY_MODEL_NAME"] = "gpt-4o-mini"
os.environ["DSPY_USE_OLLAMA"] = "false"

# Reload settings
settings = get_settings()
print(f"Updated model: {settings.model_name}")
```

### LLM Configuration

Configure language model backends.

```python
from dspy_agent.llm import configure_lm, check_ollama_ready

# Configure Ollama
configure_lm(
    use_ollama=True,
    model_name="qwen3:1.7b",
    base_url="http://localhost:11434",
    timeout=30
)

# Check if Ollama is ready
if check_ollama_ready():
    print("Ollama is ready")
else:
    print("Ollama is not available")

# Configure OpenAI-compatible API
configure_lm(
    use_ollama=False,
    model_name="gpt-4o-mini",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    timeout=60
)

# Test LLM connection
import dspy
lm = dspy.settings.lm
try:
    response = lm("Hello, world!")
    print(f"LLM response: {response}")
except Exception as e:
    print(f"LLM error: {e}")
```

## Error Handling

### Common Exceptions

DSPy Agent defines custom exceptions for different error scenarios.

```python
from dspy_agent.code_tools.code_search import CodeSearchError
from dspy_agent.embedding.indexer import EmbeddingError
from dspy_agent.rl.rlkit import RLEnvironmentError

try:
    # Code search operation
    results = search_text("invalid[regex", "./src")
except CodeSearchError as e:
    print(f"Search error: {e}")

try:
    # Embedding operation
    index = build_index("./nonexistent")
except EmbeddingError as e:
    print(f"Embedding error: {e}")

try:
    # RL environment operation
    env = RLToolEnv(workspace="./invalid")
    obs = env.reset()
except RLEnvironmentError as e:
    print(f"RL error: {e}")
```

### Logging and Debugging

Enable detailed logging for debugging.

```python
import logging
from dspy_agent.config import get_settings

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("dspy_agent")

# Log current configuration
settings = get_settings()
logger.info(f"Workspace: {settings.workspace}")
logger.debug(f"Full settings: {settings}")

# Enable LLM request logging
import dspy
dspy.settings.log_openai_usage = True

# Custom logging handler
class AgentLogHandler(logging.Handler):
    def emit(self, record):
        if record.levelname == "ERROR":
            # Send to monitoring system
            print(f"ALERT: {record.getMessage()}")

handler = AgentLogHandler()
logger.addHandler(handler)
```

This API documentation provides comprehensive coverage of DSPy Agent's core functionality. For more specific use cases and examples, refer to the test files in the `tests/` directory.
