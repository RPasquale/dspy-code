# Multi-Policy Streaming Training System

## Overview

The enhanced DSPy agent now supports fully functioning multi-step multi-policy training with real-time streaming data integration. This system leverages your existing Go + Rust + DSPy + InferMesh + RedDB + Kafka + Spark infrastructure for continuous policy improvement.

## Key Features

### 🚀 Multi-Policy Training
- **Policy A**: Tool selection at each step (discrete action)
- **Policy B**: Next tool selection conditioned on updates (multi-step sequences)
- **Policy C**: Tool usage quality optimization (arguments/parameters)

### 📊 Streaming Integration
- Real-time Kafka data ingestion
- Spark processing for large-scale analytics
- RedDB persistence for policy state
- InferMesh distributed computing

### 🔄 Continuous Learning
- Auto-generates GRPO datasets from streaming data
- Multi-step episode sequences for sequential policies
- Real-time reward shaping and policy updates

## Usage Commands

### Basic Multi-Policy Training

```bash
# Enhanced train-easy with multi-policy support
python -m dspy_agent.cli train-easy \
  --workspace . \
  --signature CodeContextSig \
  --rl-steps 150 \
  --episode-len 5 \
  --multi-policy \
  --streaming \
  --kafka-bootstrap localhost:9092 \
  --reddb-url http://localhost:8080
```

### Advanced Streaming Training

```bash
# Full streaming multi-policy training
python -m dspy_agent.cli train-streaming \
  --workspace . \
  --signature CodeContextSig \
  --rl-steps 500 \
  --grpo-steps 1000 \
  --episode-len 5 \
  --n-envs 4 \
  --continuous \
  --max-iterations 10 \
  --kafka-bootstrap localhost:9092 \
  --reddb-url http://localhost:8080 \
  --spark-master local[*]
```

### Continuous Training Loop

```bash
# Run continuous training with automatic dataset generation
python -m dspy_agent.cli train-streaming \
  --continuous \
  --max-iterations 20 \
  --episode-len 10 \
  --actions "ls,pwd,cat,cd,git,make" \
  --environment production
```

## Architecture Components

### 1. Streaming Data Flow
```
Kafka Topics → Spark Processing → RedDB Storage → Policy Training
     ↓              ↓                ↓              ↓
agent.actions  Real-time Analytics  Policy State  GRPO Updates
agent.rewards  Context Building     Action History  Tool Optimization
agent.tool_usage  Reward Shaping    Performance Metrics  Sequential Learning
```

### 2. Multi-Policy Coordination
- **Tool Selection Policy**: Learns which tool to use at each step
- **Sequential Policy**: Learns optimal tool sequences based on context
- **Usage Policy**: Optimizes tool parameters and arguments

### 3. Real-time Integration
- **Event Bus**: Publishes training metrics to Kafka
- **Context Provider**: Uses recent streaming data for policy context
- **Reward Shaping**: Real-time reward computation with streaming feedback

## Configuration Options

### Environment Variables
```bash
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export REDDB_URL=http://localhost:8080
export REDDB_NAMESPACE=dspy
export SPARK_MASTER=local[*]
```

### Docker Stack Integration
```bash
# Start the full stack
docker compose -f docker/lightweight/docker-compose.yml up -d

# Ensure all services are running
make health-check-complete

# Start InferMesh nodes and Rust runner
docker compose -f docker/lightweight/docker-compose.yml up -d \
  infermesh-node-a infermesh-node-b infermesh-router rust-env-runner
```

## Training Phases

### Phase 1: RL Tool Selection
- Multi-step PPO training for sequential tool selection
- Streaming context integration for better decisions
- Real-time reward computation and policy updates

### Phase 2: GRPO Policy Optimization
- Auto-generates datasets from streaming data
- Group Relative Policy Optimization for tool usage
- Continuous policy improvement with real-time feedback

### Phase 3: Multi-Policy Coordination
- Coordinates between tool selection and usage policies
- Learns optimal tool sequences for code generation tasks
- Adapts to changing environment conditions

## Performance Optimization

### Streaming Performance
- Batch processing for high-throughput data ingestion
- Parallel environment training (4+ environments)
- Real-time context updates for policy decisions

### Memory Management
- Automatic dataset cleanup after training iterations
- Efficient context window management
- Streaming data compression and storage

### Scalability
- Horizontal scaling with multiple Kafka partitions
- Spark cluster integration for large-scale processing
- RedDB distributed storage for policy persistence

## Monitoring and Analytics

### Real-time Metrics
```bash
# Check training progress
python -m dspy_agent.cli rl report

# View recent actions
python -m dspy_agent.cli rl recent --limit 20

# Verify storage health
python -m dspy_agent.cli rl verify-storage
```

### Streaming Analytics
- Kafka topic monitoring for data flow
- Spark job metrics for processing performance
- RedDB analytics for policy state tracking

## Troubleshooting

### Common Issues
1. **Streaming Setup Failed**: Check Kafka and RedDB connectivity
2. **Spark Integration Issues**: Verify PySpark installation
3. **Policy Training Failures**: Check environment variables and dependencies

### Debug Commands
```bash
# Test streaming connectivity
python -c "from dspy_agent.streaming.event_bus import get_event_bus; print('Streaming OK')"

# Test RedDB connection
python -c "from dspy_agent.db.enhanced_storage import get_enhanced_data_manager; print('RedDB OK')"

# Test Spark integration
python -c "from pyspark.sql import SparkSession; print('Spark OK')"
```

## Advanced Usage

### Custom Action Filtering
```bash
python -m dspy_agent.cli train-streaming \
  --actions "git,make,docker,curl" \
  --episode-len 8 \
  --continuous
```

### Production Deployment
```bash
# Production environment with full stack
python -m dspy_agent.cli train-streaming \
  --environment production \
  --kafka-bootstrap kafka-cluster:9092 \
  --reddb-url http://reddb-cluster:8080 \
  --spark-master spark://spark-master:7077 \
  --continuous \
  --max-iterations 100
```

## Integration with Existing Infrastructure

### Go + Rust Integration
- Rust env-runner for high-performance tool execution
- Go orchestrator for distributed training coordination
- InferMesh for distributed computing across nodes

### Database Integration
- RedDB for persistent policy state and analytics
- SQLite fallback for local development
- Real-time data synchronization across components

### Streaming Pipeline
- Kafka for high-throughput message streaming
- Spark for real-time data processing
- Event-driven architecture for responsive training

This system provides a complete end-to-end solution for multi-policy training with streaming data integration, optimized for your existing infrastructure stack.
