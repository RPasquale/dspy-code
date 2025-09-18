# Real-Time Agent Monitoring with Kafka/Spark

## 🎯 **Overview**

Your DSPy Agent now has comprehensive real-time monitoring that leverages your existing Kafka and Spark infrastructure to provide live visibility into the agent's actions, thoughts, and learning progress.

## 🏗️ **Architecture**

```
Agent Actions → Kafka Topics → Spark Processing → Frontend/CLI Display
     ↓              ↓              ↓                    ↓
  Execute        Stream         Process            Real-time
  Actions        Data           Analytics          Monitoring
```

### **Kafka Topics Used**:
- `agent.actions` - Agent actions and results
- `agent.thoughts` - Agent reasoning and decision-making
- `agent.learning` - Learning progress and metrics
- `agent.metrics` - Performance and health metrics
- `agent.monitor.frontend` - Real-time data for frontend display
- `agent.rl.vectorized` - RL training data (existing)

## 🚀 **Getting Started**

### **1. Start the Streaming Infrastructure**
```bash
# Start the local Kafka/Spark stack
uv run dspy-agent lightweight_up

# Or start the full streaming stack
uv run dspy-agent up
```

### **2. Start Agent Streaming Integration**
```bash
# Start the agent streaming integration (in background)
uv run python scripts/integrate_agent_streaming.py --monitor &
```

### **3. Start Real-Time Monitoring**
```bash
# CLI monitoring with rich display
uv run python scripts/realtime_agent_monitor.py --cli

# Or just show current status
uv run python scripts/realtime_agent_monitor.py --status
```

## 📊 **Monitoring Features**

### **Real-Time Action Tracking**
- **Tool Usage**: See which tools the agent is using
- **Action Results**: View success/failure of each action
- **Rewards**: Monitor learning rewards in real-time
- **Context**: Understand what the agent is working on

### **Agent Thoughts Display**
- **Reasoning**: See the agent's decision-making process
- **Planning**: View how the agent plans tasks
- **Problem Solving**: Watch the agent work through problems
- **Learning**: Observe how the agent learns from experience

### **Learning Progress Monitoring**
- **Success Rate**: Track how often the agent succeeds
- **Learning Trend**: See if the agent is improving
- **Tool Preferences**: Monitor which tools the agent prefers
- **Reward Evolution**: Watch rewards change over time

### **Performance Metrics**
- **Action Count**: Total number of actions taken
- **Response Time**: How quickly the agent responds
- **Error Rate**: Frequency of errors and failures
- **Resource Usage**: Memory and CPU utilization

## 🖥️ **CLI Monitoring Interface**

The CLI monitor provides a rich, real-time display:

```
┌─────────────────────────────────────────────────────────────────┐
│ 🤖 DSPy Agent Real-Time Monitor | Events: 45 | Success: 78% │
├─────────────────────────────────────────────────────────────────┤
│ Recent Actions          │ Agent Thoughts                       │
│ 14:32:15 grep (0.8)     │ 💭 I need to understand the codebase │
│ 14:32:18 esearch (0.9)  │ 💭 This function looks complex      │
│ 14:32:21 edit (0.7)     │ 💭 Let me check the tests first     │
├─────────────────────────────────────────────────────────────────┤
│ Learning Stats          │ Learning Progress                    │
│ Total Events: 45        │ Recent Rewards:                      │
│ Success Rate: 78%       │ 0.8 0.9 0.7 0.6 0.8                 │
│ Avg Reward: 0.75        │ Tool Usage:                          │
│ Learning Trend: improving│ grep: 12, esearch: 8, edit: 6      │
└─────────────────────────────────────────────────────────────────┘
```

## 🌐 **Frontend Integration**

The monitoring system publishes data to Kafka topics that your frontend can consume:

### **WebSocket Connection** (if using React dashboard):
```javascript
// Connect to Kafka stream for real-time updates
const ws = new WebSocket('ws://localhost:8080/ws/agent-monitor');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'action_update') {
        updateActionDisplay(data.data);
    } else if (data.type === 'thought_update') {
        updateThoughtDisplay(data.data);
    } else if (data.type === 'learning_update') {
        updateLearningDisplay(data.data);
    }
};
```

### **REST API Endpoints**:
```bash
# Get current agent status
curl http://localhost:8080/api/agent/status

# Get recent actions
curl http://localhost:8080/api/agent/actions?limit=20

# Get learning metrics
curl http://localhost:8080/api/agent/learning
```

## 🔧 **Configuration**

### **Environment Variables**:
```bash
# Kafka configuration
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export KAFKA_GROUP_ID=dspy-agent-monitor

# Monitoring configuration
export AGENT_MONITOR_INTERVAL=1.0
export AGENT_METRICS_INTERVAL=30.0
export AGENT_LOG_LEVEL=INFO

# Frontend configuration
export FRONTEND_WS_PORT=8080
export FRONTEND_API_PORT=8080
```

### **Kafka Topic Configuration**:
```json
{
  "topics": [
    {
      "name": "agent.actions",
      "partitions": 3,
      "replication_factor": 1
    },
    {
      "name": "agent.thoughts", 
      "partitions": 3,
      "replication_factor": 1
    },
    {
      "name": "agent.learning",
      "partitions": 1,
      "replication_factor": 1
    },
    {
      "name": "agent.monitor.frontend",
      "partitions": 1,
      "replication_factor": 1
    }
  ]
}
```

## 📈 **Advanced Monitoring**

### **Custom Metrics**
You can add custom metrics by publishing to the metrics topic:

```python
from scripts.integrate_agent_streaming import AgentStreamingPublisher

publisher = AgentStreamingPublisher(project_root)

# Publish custom metrics
publisher.publish_metrics({
    "custom_metric": 42,
    "performance_score": 0.95,
    "user_satisfaction": 0.88
})
```

### **Alerting**
Set up alerts based on agent performance:

```python
# Example: Alert if success rate drops below 50%
if learning_metrics["success_rate"] < 0.5:
    publisher.publish_metrics({
        "alert": "low_success_rate",
        "value": learning_metrics["success_rate"],
        "threshold": 0.5
    })
```

### **Historical Analysis**
Access historical data for analysis:

```bash
# Get learning events from the last hour
uv run python -c "
import json
from pathlib import Path
from datetime import datetime, timedelta

events_file = Path('.dspy_rl_events.jsonl')
if events_file.exists():
    with open(events_file) as f:
        events = [json.loads(line) for line in f if line.strip()]
    
    # Filter recent events
    cutoff = datetime.now() - timedelta(hours=1)
    recent_events = [
        e for e in events 
        if datetime.fromtimestamp(e.get('timestamp', 0)) > cutoff
    ]
    
    print(f'Recent events: {len(recent_events)}')
"
```

## 🛠️ **Troubleshooting**

### **Common Issues**:

1. **Kafka not available**:
   ```bash
   # Check if Kafka is running
   docker ps | grep kafka
   
   # Start Kafka if needed
   uv run dspy-agent lightweight_up
   ```

2. **No data in topics**:
   ```bash
   # Check if agent streaming integration is running
   ps aux | grep integrate_agent_streaming
   
   # Start it if needed
   uv run python scripts/integrate_agent_streaming.py --monitor &
   ```

3. **Frontend not receiving data**:
   ```bash
   # Check if frontend is connected to Kafka
   curl http://localhost:8080/api/agent/status
   
   # Check Kafka topic
   docker exec -it kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic agent.monitor.frontend
   ```

### **Debug Mode**:
```bash
# Enable debug logging
export AGENT_LOG_LEVEL=DEBUG
export KAFKA_DEBUG=true

# Run with verbose output
uv run python scripts/realtime_agent_monitor.py --cli --debug
```

## 🎯 **Best Practices**

### **Monitoring Setup**:
1. **Start infrastructure first**: Always start Kafka/Spark before monitoring
2. **Use background processes**: Run streaming integration in background
3. **Monitor resources**: Keep an eye on memory and CPU usage
4. **Set up alerts**: Configure alerts for critical metrics

### **Performance Optimization**:
1. **Batch updates**: Don't publish every single action
2. **Filter data**: Only publish relevant information
3. **Use compression**: Enable Kafka compression for large payloads
4. **Monitor throughput**: Watch for backpressure in Kafka topics

### **Data Management**:
1. **Retention policies**: Set appropriate retention for Kafka topics
2. **Cleanup old data**: Regularly clean up old log files
3. **Backup metrics**: Store important metrics in persistent storage
4. **Archive data**: Archive historical data for long-term analysis

## 🚀 **Quick Start Commands**

```bash
# 1. Start the full monitoring stack
uv run dspy-agent lightweight_up &
uv run python scripts/integrate_agent_streaming.py --monitor &
uv run python scripts/realtime_agent_monitor.py --cli

# 2. Test the system
uv run python scripts/integrate_agent_streaming.py --test

# 3. Check status
uv run python scripts/realtime_agent_monitor.py --status

# 4. View in frontend (if available)
open http://localhost:8080
```

## 🎉 **Benefits**

With real-time monitoring, you can:

- ✅ **See what the agent is thinking** in real-time
- ✅ **Monitor learning progress** as it happens
- ✅ **Identify issues** before they become problems
- ✅ **Optimize performance** based on live data
- ✅ **Provide feedback** to improve the agent
- ✅ **Scale monitoring** to multiple agents
- ✅ **Integrate with existing** Kafka/Spark infrastructure

The agent now provides complete transparency into its decision-making process, making it a true collaborative coding partner! 🤖✨
