# DSPy Agent Learning Guide

## 🧠 **How the Agent Learns**

The DSPy Agent uses **Reinforcement Learning (RL)** to improve its performance over time. It learns from your interactions and feedback to make better decisions about which tools to use and how to approach tasks.

## 🚀 **Getting Started**

### 1. **Start the Agent**
```bash
# Simple start
./scripts/start-agent.sh

# Or manually
uv run dspy-agent --workspace $(pwd)
```

### 2. **Basic Commands to Get Started**
Once the agent starts, try these commands:

```
help                    # See all available commands
ls                      # List files in your workspace
tree                    # Show directory structure
plan "add a new feature" # Get a task plan
```

## 🎯 **How Learning Works**

### **The Learning Loop**
1. **You give a task** → Agent chooses tools/actions
2. **Agent executes actions** → Gets results and feedback
3. **Agent learns** → Updates its policy for future tasks
4. **Next time** → Agent makes better decisions

### **What the Agent Learns**
- **Tool Selection**: Which tools work best for different tasks
- **Action Sequences**: What order of actions leads to success
- **Context Understanding**: How to interpret your requests
- **Error Recovery**: How to handle and fix problems

## 📊 **Learning Data Storage**

The agent stores learning data in your workspace:

```
.dspy_rl_state.json     # Current learning state
.dspy_rl_events.jsonl   # Learning events and rewards
logs/                   # Detailed execution logs
```

## 🛠️ **Interactive Learning Commands**

### **Planning and Execution**
```
plan "fix the failing tests"     # Get a plan
edit "update the API" --apply    # Propose and apply changes
grep "error"                     # Search for issues
esearch "authentication"         # Semantic search
```

### **Learning-Focused Commands**
```
stats                           # Show learning statistics
ctx                             # Extract key learning events
auto                            # Manage auto-training loop
```

## 🎮 **Making the Agent Learn Effectively**

### **1. Give Clear, Specific Tasks**
```bash
# Good examples
plan "add user authentication to the login endpoint"
edit "fix the memory leak in the data processor"
grep "TODO.*security"

# Less effective
plan "make it better"
edit "fix stuff"
```

### **2. Provide Feedback**
The agent learns from:
- **Success**: When tasks complete successfully
- **Failure**: When things go wrong (it learns what not to do)
- **Your corrections**: When you fix or modify its work

### **3. Use Consistent Patterns**
- Use similar task descriptions for similar work
- Follow consistent naming conventions
- Use the same tools for similar problems

### **4. Let It Explore**
- Don't micromanage every action
- Let it try different approaches
- Allow it to make mistakes and learn

## 🔄 **Learning Modes**

### **Interactive Learning** (Default)
- Agent learns from each command you give
- Immediate feedback and adaptation
- Best for development and exploration

### **Auto-Training** (Optional)
```bash
# Enable auto-training (advanced)
export DSPY_AUTO_TRAIN=true
dspy-agent --workspace $(pwd)
```

## 📈 **Monitoring Learning Progress**

### **Check Learning Stats**
```
stats                           # Show current learning statistics
```

### **View Learning Events**
```bash
# Check learning state
cat .dspy_rl_state.json

# View recent learning events
tail -f .dspy_rl_events.jsonl
```

### **Analyze Learning Data**
```bash
# View learning progress
uv run python -c "
import json
with open('.dspy_rl_state.json') as f:
    state = json.load(f)
    print('Learning State:', state)
"
```

## 🎯 **Practical Learning Examples**

### **Example 1: Learning to Fix Tests**
```bash
# Start with a failing test
grep "test.*fail"

# Let the agent plan a fix
plan "fix the failing test in test_user_auth.py"

# Apply the fix
edit "fix the authentication test" --apply

# The agent learns: 'edit' + 'test fix' = good outcome
```

### **Example 2: Learning Code Patterns**
```bash
# Search for similar patterns
esearch "database connection setup"

# Let the agent understand the pattern
plan "add database connection to the new service"

# Apply the pattern
edit "implement database connection following the existing pattern" --apply
```

### **Example 3: Learning Error Handling**
```bash
# Find error-prone areas
grep "except.*pass"

# Plan improvements
plan "improve error handling in the data processor"

# Apply improvements
edit "add proper error handling and logging" --apply
```

## 🧪 **Testing the Learning**

### **Run Learning Tests**
```bash
# Test RL components
./scripts/test_rl.py

# Test learning behavior
uv run pytest tests/test_rl_tooling.py -v
```

### **Verify Learning Works**
```bash
# Check that the agent prefers successful actions
uv run python -c "
from dspy_agent.rl.rlkit import bandit_trainer, TrainerConfig
# Run a quick learning test
print('Learning test completed')
"
```

## 🎛️ **Advanced Learning Configuration**

### **Environment Variables**
```bash
# Learning parameters
export DSPY_AUTO_TRAIN=true              # Enable auto-training
export DSPY_AUTO_TRAIN_INTERVAL_SEC=1800 # Training interval
export DSPY_AUTO_RL_STEPS=200            # RL training steps
export DSPY_LOG_LEVEL=DEBUG              # Detailed logging
```

### **Custom Learning Settings**
```bash
# Start with custom learning config
dspy-agent --workspace $(pwd) --approval manual
```

## 🚨 **Troubleshooting Learning**

### **If Learning Seems Slow**
1. **Check learning data**: `cat .dspy_rl_state.json`
2. **Verify rewards**: Look for positive rewards in events
3. **Increase interaction**: Use more commands to give it data
4. **Check logs**: `tail -f logs/dspy_agent.log`

### **If Learning Stops**
1. **Reset learning state**: `rm .dspy_rl_state.json .dspy_rl_events.jsonl`
2. **Restart agent**: Exit and restart the agent
3. **Check permissions**: Ensure agent can write learning files

### **If Actions Seem Random**
1. **Give more feedback**: Use `--apply` to show what works
2. **Be consistent**: Use similar task descriptions
3. **Check learning state**: Verify learning data is being saved

## 🎉 **Best Practices for Learning**

### **Do:**
- ✅ Give clear, specific tasks
- ✅ Use consistent language and patterns
- ✅ Let the agent try different approaches
- ✅ Provide feedback through your actions
- ✅ Use the agent regularly to build up learning data

### **Don't:**
- ❌ Give vague or ambiguous tasks
- ❌ Constantly override the agent's decisions
- ❌ Expect perfect performance immediately
- ❌ Ignore the learning data and statistics

## 📊 **Learning Metrics to Watch**

### **Success Indicators**
- **Tool Selection**: Agent chooses appropriate tools for tasks
- **Action Sequences**: Logical progression of actions
- **Error Recovery**: Handles failures gracefully
- **Context Understanding**: Interprets requests correctly

### **Learning Progress**
- **Reward Trends**: Rewards should increase over time
- **Action Diversity**: Agent tries different approaches
- **Success Rate**: More tasks complete successfully
- **Efficiency**: Faster task completion

## 🚀 **Getting the Most Out of Learning**

1. **Start Simple**: Begin with basic tasks to build learning data
2. **Be Patient**: Learning takes time and interaction
3. **Stay Consistent**: Use similar patterns and language
4. **Monitor Progress**: Check learning stats regularly
5. **Provide Feedback**: Let the agent know what works

## 🎯 **Quick Start Learning Session**

```bash
# 1. Start the agent
./scripts/start-agent.sh

# 2. Give it some tasks to learn from
plan "explore the codebase structure"
grep "def main"
esearch "configuration setup"

# 3. Let it work on a real task
plan "add error logging to the main function"
edit "add comprehensive error logging" --apply

# 4. Check learning progress
stats

# 5. Continue with more tasks to build learning data
```

The agent will learn from each interaction and get better at understanding your preferences and the codebase patterns. The more you use it, the smarter it becomes! 🧠✨
