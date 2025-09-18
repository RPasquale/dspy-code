# DSPy Agent Learning Example

## 🎯 **Quick Learning Session**

Here's a practical example of how to use the DSPy Agent and make it learn:

### **Step 1: Start the Agent**
```bash
./scripts/start-agent.sh
```

### **Step 2: Give It Learning Tasks**

Once the agent starts, try these commands in sequence:

```bash
# 1. Explore the codebase
plan "explore the project structure and understand the main components"

# 2. Search for specific patterns
grep "def test_"
esearch "database connection"

# 3. Get context about the codebase
ctx

# 4. Plan a real task
plan "add error logging to the main functions"

# 5. Apply changes (this teaches the agent what works)
edit "add comprehensive error logging with try-catch blocks" --apply

# 6. Check learning progress
stats
```

### **Step 3: Watch It Learn**

The agent learns from each interaction:

- **`plan`** → Learns to break down tasks
- **`grep`** → Learns code search patterns  
- **`esearch`** → Learns semantic search
- **`edit --apply`** → Learns what changes work
- **`stats`** → Shows learning progress

### **Step 4: Continue Learning**

Keep giving it tasks:

```bash
# More learning opportunities
plan "improve the test coverage"
grep "TODO"
esearch "performance optimization"
edit "optimize the slow functions" --apply
```

## 🧠 **What the Agent Learns**

### **Tool Selection**
- Which tools work best for different tasks
- When to use `grep` vs `esearch`
- How to combine multiple tools effectively

### **Action Sequences**
- Logical progression of actions
- How to build context before making changes
- When to search before editing

### **Context Understanding**
- How to interpret your requests
- What patterns to look for
- How to understand code relationships

### **Success Patterns**
- What changes lead to good outcomes
- How to avoid common mistakes
- Which approaches work best

## 📊 **Monitoring Learning**

### **Check Learning State**
```bash
# In the agent session
stats

# Or check files directly
cat .dspy_rl_state.json
tail -f .dspy_rl_events.jsonl
```

### **Learning Indicators**
- **Tool Selection**: Agent chooses appropriate tools
- **Action Sequences**: Logical progression
- **Success Rate**: More tasks complete successfully
- **Efficiency**: Faster task completion

## 🎮 **Interactive Learning Tips**

### **Give Clear Tasks**
```bash
# Good
plan "add input validation to the user registration endpoint"
edit "implement proper error handling for database operations"

# Less effective
plan "fix stuff"
edit "make it better"
```

### **Provide Feedback**
- Use `--apply` when changes are good
- Let it try different approaches
- Don't micromanage every action

### **Be Consistent**
- Use similar language for similar tasks
- Follow consistent patterns
- Use the same tools for similar problems

## 🚀 **Advanced Learning**

### **Enable Auto-Training**
```bash
export DSPY_AUTO_TRAIN=true
dspy-agent --workspace $(pwd)
```

### **Custom Learning Configuration**
```bash
export DSPY_AUTO_TRAIN_INTERVAL_SEC=1800  # Train every 30 minutes
export DSPY_AUTO_RL_STEPS=200             # More training steps
export DSPY_LOG_LEVEL=DEBUG               # Detailed logging
```

## 🎉 **Expected Learning Progression**

### **Week 1: Basic Learning**
- Agent learns your coding style
- Understands project structure
- Gets familiar with common patterns

### **Week 2: Improved Tool Selection**
- Better at choosing appropriate tools
- More efficient action sequences
- Improved context understanding

### **Week 3: Advanced Patterns**
- Recognizes complex patterns
- Suggests better approaches
- Anticipates your needs

### **Week 4+: Expert Assistant**
- Highly efficient task completion
- Proactive suggestions
- Deep understanding of your codebase

## 🔧 **Troubleshooting Learning**

### **If Learning Seems Slow**
1. Use more commands to give it data
2. Be more specific in your requests
3. Check that learning files are being created
4. Ensure you're using `--apply` for good changes

### **If Actions Seem Random**
1. Give more feedback through your actions
2. Use consistent language and patterns
3. Check learning state with `stats`
4. Restart the agent if needed

### **Reset Learning (if needed)**
```bash
rm .dspy_rl_state.json .dspy_rl_events.jsonl
# Restart the agent to begin fresh
```

## 🎯 **Quick Start Learning Session**

```bash
# 1. Start the agent
./scripts/start-agent.sh

# 2. Give it some learning tasks
plan "understand the project structure"
grep "def main"
esearch "configuration"
ctx

# 3. Let it work on a real task
plan "add error handling to the main function"
edit "implement comprehensive error handling" --apply

# 4. Check learning progress
stats

# 5. Continue with more tasks
plan "improve the test coverage"
grep "test_.*fail"
edit "fix the failing tests" --apply

# 6. Watch it get smarter!
stats
```

The more you use the agent, the smarter it becomes! 🧠✨
