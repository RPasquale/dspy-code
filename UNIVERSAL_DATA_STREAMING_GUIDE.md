# Universal Data Streaming Guide

## 🎯 **Overview**

The DSPy Agent now has a **universal data streaming engine** that can ingest data from ANY source and automatically:
- ✅ **Stream data** through Kafka
- ✅ **Vectorize data** for RL training  
- ✅ **Store in RedDB** for persistence
- ✅ **Create training data** automatically
- ✅ **Learn from streams** in real-time

## 🏗️ **Architecture**

```
Data Sources → Connectors → Kafka → Vectorization → RedDB → RL Training
     ↓             ↓         ↓          ↓           ↓         ↓
  APIs, DBs,    Enhanced   Streaming  Existing   Enhanced   Agent
  Files, etc.   Connectors  Pipeline  Pipeline   Storage   Learning
```

## 🚀 **Quick Start**

### **1. Start the Streaming Infrastructure**
```bash
# Start the existing streaming stack
uv run dspy-agent lightweight_up &
```

### **2. Stream from Any Data Source**
```bash
# Stream from GitHub commits
uv run python scripts/stream_any_data_source.py --github-repo microsoft/vscode --commits

# Stream from a database
uv run python scripts/stream_any_data_source.py --database postgresql --db-host localhost --db-user myuser --db-password mypass --db-name mydb --db-query "SELECT * FROM users"

# Stream from a file
uv run python scripts/stream_any_data_source.py --file /path/to/data.json

# Stream from a webhook
uv run python scripts/stream_any_data_source.py --webhook --webhook-port 8080
```

### **3. Monitor the Streaming**
```bash
# Monitor real-time data ingestion
uv run python scripts/realtime_agent_monitor.py --cli
```

## 📊 **Supported Data Sources**

### **APIs**
- **REST APIs**: Any HTTP endpoint
- **GraphQL APIs**: GraphQL endpoints
- **GitHub API**: Commits, issues, pull requests
- **Social Media APIs**: Twitter, LinkedIn, etc.
- **IoT APIs**: Sensor data, device APIs

### **Databases**
- **PostgreSQL**: SQL queries
- **MySQL**: SQL queries  
- **MongoDB**: Document queries
- **Redis**: Key-value data
- **Any SQL database**: Via connection strings

### **Files**
- **JSON files**: Structured data
- **CSV files**: Tabular data
- **Log files**: Application logs
- **Text files**: Any text content
- **Binary files**: With custom parsers

### **Webhooks**
- **Incoming webhooks**: Real-time data
- **Outgoing webhooks**: Event notifications
- **Custom endpoints**: Any HTTP endpoint

### **Message Queues**
- **RabbitMQ**: Message queues
- **Apache Kafka**: Event streams
- **Redis Pub/Sub**: Real-time messaging
- **AWS SQS**: Cloud messaging

## 🔧 **Configuration Examples**

### **GitHub Repository Streaming**
```bash
# Stream commits from a repository
uv run python scripts/stream_any_data_source.py \
  --github-repo microsoft/vscode \
  --github-data commits \
  --github-interval 30

# Stream issues from a repository  
uv run python scripts/stream_any_data_source.py \
  --github-repo facebook/react \
  --github-data issues \
  --github-interval 60
```

### **Database Streaming**
```bash
# PostgreSQL streaming
uv run python scripts/stream_any_data_source.py \
  --database postgresql \
  --db-host localhost \
  --db-port 5432 \
  --db-user myuser \
  --db-password mypass \
  --db-name mydb \
  --db-query "SELECT * FROM users WHERE created_at > NOW() - INTERVAL '1 hour'"

# MongoDB streaming
uv run python scripts/stream_any_data_source.py \
  --database mongodb \
  --db-host localhost \
  --db-port 27017 \
  --db-user myuser \
  --db-password mypass \
  --db-name mydb \
  --db-collection events
```

### **File Streaming**
```bash
# JSON file streaming
uv run python scripts/stream_any_data_source.py \
  --file /path/to/data.json \
  --file-interval 5

# Log file streaming
uv run python scripts/stream_any_data_source.py \
  --file /var/log/application.log \
  --file-interval 1
```

### **Webhook Streaming**
```bash
# Start webhook receiver
uv run python scripts/stream_any_data_source.py \
  --webhook \
  --webhook-port 8080 \
  --webhook-path /webhook

# Send data to webhook
curl -X POST http://localhost:8080/webhook \
  -H "Content-Type: application/json" \
  -d '{"event": "user_action", "data": {"user_id": 123, "action": "login"}}'
```

### **Generic API Streaming**
```bash
# Stream from any API
uv run python scripts/stream_any_data_source.py \
  --api-url "https://api.example.com/data" \
  --api-method GET \
  --api-headers '{"Authorization": "Bearer token"}' \
  --api-params '{"limit": 100}'
```

## 📁 **Configuration Files**

### **JSON Configuration**
Create a `streaming_config.json` file:

```json
{
  "sources": [
    {
      "name": "github_vscode_commits",
      "type": "api",
      "config": {
        "url": "https://api.github.com/repos/microsoft/vscode/commits",
        "method": "GET",
        "headers": {"Accept": "application/vnd.github.v3+json"},
        "params": {"per_page": 10}
      },
      "enabled": true,
      "poll_interval": 30.0,
      "batch_size": 100
    },
    {
      "name": "postgres_users",
      "type": "database",
      "config": {
        "type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "user": "myuser",
        "password": "mypass",
        "database": "mydb",
        "query": "SELECT * FROM users WHERE active = true"
      },
      "enabled": true,
      "poll_interval": 60.0
    },
    {
      "name": "webhook_receiver",
      "type": "webhook",
      "config": {
        "host": "localhost",
        "port": 8080,
        "path": "/webhook"
      },
      "enabled": true
    }
  ]
}
```

Then use it:
```bash
uv run python scripts/stream_any_data_source.py --config streaming_config.json
```

## 🧠 **How It Works**

### **1. Data Ingestion**
- **Connectors** poll data sources at configured intervals
- **Data** is immediately published to Kafka topics
- **RedDB** stores metadata and action records

### **2. Vectorization**
- **Existing vectorization pipeline** processes the data
- **RLVectorizer** converts data to embeddings
- **Feature store** maintains vector representations

### **3. Training Data Generation**
- **Automatic training examples** are created from streamed data
- **Action records** are generated for RL training
- **Rewards** are assigned based on data quality and relevance

### **4. Agent Learning**
- **Existing RL training** processes the new data
- **Agent** learns from patterns in the streamed data
- **Real-time adaptation** to new data sources

## 📈 **Monitoring and Analytics**

### **Real-Time Monitoring**
```bash
# Monitor streaming in real-time
uv run python scripts/realtime_agent_monitor.py --cli

# Check streaming status
uv run python scripts/realtime_agent_monitor.py --status
```

### **Data Analytics**
```bash
# View recent actions from streaming
uv run python -c "
from dspy_agent.db import get_enhanced_data_manager
dm = get_enhanced_data_manager()
actions = dm.get_recent_actions(limit=50)
print(f'Recent streaming actions: {len(actions)}')
for action in actions[-10:]:
    print(f'  {action.tool} -> {action.action} (reward: {action.reward:.2f})')
"
```

### **RedDB Queries**
```bash
# Query streaming data in RedDB
uv run python -c "
from dspy_agent.db import get_enhanced_data_manager
dm = get_enhanced_data_manager()

# Get recent logs from streaming
logs = dm.get_recent_logs(limit=20)
print(f'Recent streaming logs: {len(logs)}')

# Get top performing actions
from dspy_agent.db.enhanced_storage import get_recent_high_reward_actions
high_reward_actions = get_recent_high_reward_actions(dm, min_reward=0.7, limit=10)
print(f'High reward streaming actions: {len(high_reward_actions)}')
"
```

## 🎯 **Use Cases**

### **1. Code Repository Monitoring**
```bash
# Monitor multiple repositories
uv run python scripts/stream_any_data_source.py \
  --github-repo microsoft/vscode --commits &
uv run python scripts/stream_any_data_source.py \
  --github-repo facebook/react --commits &
uv run python scripts/stream_any_data_source.py \
  --github-repo tensorflow/tensorflow --commits &
```

### **2. Database Change Monitoring**
```bash
# Monitor database changes
uv run python scripts/stream_any_data_source.py \
  --database postgresql \
  --db-query "SELECT * FROM audit_log WHERE created_at > NOW() - INTERVAL '1 hour'"
```

### **3. Log File Analysis**
```bash
# Stream application logs
uv run python scripts/stream_any_data_source.py \
  --file /var/log/nginx/access.log \
  --file-interval 1
```

### **4. IoT Sensor Data**
```bash
# Stream sensor data via API
uv run python scripts/stream_any_data_source.py \
  --api-url "https://api.iot-platform.com/sensors" \
  --api-headers '{"Authorization": "Bearer sensor_token"}' \
  --api-params '{"device_id": "sensor_001"}'
```

### **5. Social Media Monitoring**
```bash
# Stream social media data
uv run python scripts/stream_any_data_source.py \
  --api-url "https://api.twitter.com/2/tweets/search/recent" \
  --api-headers '{"Authorization": "Bearer twitter_token"}' \
  --api-params '{"query": "machine learning", "max_results": 100}'
```

## 🔧 **Advanced Configuration**

### **Custom Connectors**
You can create custom connectors by extending the base classes:

```python
from scripts.enhanced_streaming_connectors import StreamingDataConnector

class CustomConnector(StreamingDataConnector):
    async def connect(self) -> bool:
        # Custom connection logic
        pass
    
    async def poll(self) -> List[Dict[str, Any]]:
        # Custom polling logic
        pass
    
    async def disconnect(self):
        # Custom disconnection logic
        pass
```

### **Custom Data Processing**
```python
# Override data processing in connectors
async def process_data(self, data_list: List[Dict[str, Any]]):
    # Custom processing before storing in RedDB
    processed_data = self.custom_transform(data_list)
    await super().process_data(processed_data)
```

### **Custom Topics**
```python
# Use custom Kafka topics
config = DataSourceConfig(
    name="custom_source",
    type="api",
    config={"url": "https://api.example.com/data"},
    kafka_topic="custom.data.topic"  # Custom topic
)
```

## 🚀 **Performance Optimization**

### **Batch Processing**
```json
{
  "batch_size": 1000,
  "poll_interval": 5.0
}
```

### **Connection Pooling**
```json
{
  "max_connections": 10,
  "connection_timeout": 30.0
}
```

### **Caching**
```json
{
  "cache_enabled": true,
  "cache_ttl": 3600,
  "cache_size": 10000
}
```

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **Connection failures**:
   ```bash
   # Check connectivity
   curl -I https://api.github.com/repos/microsoft/vscode/commits
   ```

2. **Kafka not available**:
   ```bash
   # Check Kafka status
   docker ps | grep kafka
   ```

3. **RedDB errors**:
   ```bash
   # Check RedDB status
   uv run python -c "from dspy_agent.db import get_storage; print(get_storage().health_check())"
   ```

### **Debug Mode**
```bash
# Enable debug logging
export DSPY_LOG_LEVEL=DEBUG
uv run python scripts/stream_any_data_source.py --github-repo microsoft/vscode --commits
```

## 🎉 **Benefits**

With universal data streaming, the agent can:

- ✅ **Learn from any data source** in real-time
- ✅ **Adapt to new environments** automatically  
- ✅ **Scale to multiple data sources** simultaneously
- ✅ **Integrate with existing infrastructure** seamlessly
- ✅ **Provide real-time insights** into data patterns
- ✅ **Generate training data** automatically
- ✅ **Improve continuously** from streaming data

The agent becomes a **universal learning system** that can ingest and learn from any data source! 🤖✨

## 🚀 **Next Steps**

1. **Start streaming**: Choose a data source and start streaming
2. **Monitor progress**: Use the real-time monitoring interface
3. **Scale up**: Add more data sources as needed
4. **Customize**: Create custom connectors for specific needs
5. **Analyze**: Use RedDB queries to analyze streaming data

**Start streaming now**: `uv run python scripts/stream_any_data_source.py --github-repo microsoft/vscode --commits`
