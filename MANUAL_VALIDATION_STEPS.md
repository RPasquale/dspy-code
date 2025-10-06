# Manual Streaming Pipeline Validation Steps

This guide provides step-by-step manual validation of the DSPy streaming pipeline.

## Prerequisites
- Docker Compose services running: `cd docker/lightweight && docker compose up -d`
- DSPy agent running in another terminal

## Step 1: Inventory Kafka Topics

```bash
cd docker/lightweight
docker compose exec kafka kafka-topics.sh --bootstrap-server localhost:9092 --list
```

**Expected Output:**
- `code.fs.events`
- `agent.results.backend`
- `agent.results.frontend` 
- `agent.rl.vectorized`
- `embedding_input`
- `embeddings`

## Step 2: Test Topic Message Flow

### Test code.fs.events (File System Events)
```bash
# In one terminal - start consumer
docker compose exec kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic code.fs.events \
  --from-beginning \
  --timeout-ms 5000

# In another terminal - trigger file change
echo '# Test change' >> some_file.py
```

**Expected:** JSON with `{"path": "...", "event": "modified", ...}`

### Test agent.results.backend (Agent Logs)
```bash
docker compose exec kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic agent.results.backend \
  --from-beginning \
  --timeout-ms 5000
```

**Expected:** Log snippets from agent operations

### Test embedding_input (Spark Output)
```bash
docker compose exec kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic embedding_input \
  --from-beginning \
  --timeout-ms 5000
```

**Expected:** JSON with `{"topic": "...", "text": "...", "doc_id": ...}`

### Test embeddings (InferMesh Output)
```bash
docker compose exec kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic embeddings \
  --from-beginning \
  --timeout-ms 5000
```

**Expected:** Records with `"vector": [...]` and `embedded_ts`

### Test agent.rl.vectorized (RL Features)
```bash
docker compose exec kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic agent.rl.vectorized \
  --from-beginning \
  --timeout-ms 5000
```

**Expected:** Feature arrays with metadata like `{"topic": "code.fs.events", "features": [...], "feature_names": [...]}`

## Step 3: Monitor Container Logs

### Spark Vectorizer
```bash
docker compose logs spark-vectorizer --tail 40
```
**Look for:** `Progress: ... numInputRows=...` messages

### Embed Worker
```bash
docker compose logs embed-worker --tail 40
```
**Look for:** `records_in/out` and InferMesh call reports

### Code Watcher
```bash
docker compose logs dspy-code-watch --tail 40
```
**Look for:** File change notifications

## Step 4: Validate Parquet Files and Checkpoints

```bash
# Check vectorized directories
ls -lt vectorized/embeddings
ls -lt vectorized/embeddings_imesh

# Check checkpoints
ls -lt .dspy_checkpoints/vectorizer/sources
```

**Expected:** Recent timestamps showing continuous updates

## Step 5: Verify Agent Ingestion

### Check Agent Log Files
```bash
tail -f logs/agent_actions.jsonl
tail -f logs/agent_tool_usage.jsonl  
tail -f logs/agent_learning.jsonl
```

**Expected:** New entries as agent processes streamed data

### Test RL Training
```bash
uv run dspy-agent rl neural --workspace "$(pwd)" --steps 100 --n-envs 1 --log-jsonl logs/rl_trace.jsonl
tail logs/rl_trace.jsonl
```

**Expected:** RL training entries matching Kafka `agent.rl.vectorized` stream

## Step 6: Check Consumer Groups

```bash
# List all consumer groups
docker compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 --all-groups --list

# Check specific groups
docker compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 --group spark-vectorizer --describe

docker compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 --group dspy-code-indexer --describe
```

**Expected:** Active consumer groups with advancing offsets

## Troubleshooting

### If no messages in topics:
1. Trigger activity (edit files, run agent commands)
2. Check if services are running: `docker compose ps`
3. Check service logs for errors

### If missing topics:
1. Verify stream configuration in `.dspy_stream.json`
2. Check `dspy_agent/streaming/streamkit.py:1097-1132`
3. Restart services: `docker compose restart`

### If no parquet files:
1. Check Spark vectorizer logs
2. Verify Kafka connectivity
3. Check `.dspy_checkpoints/vectorizer` for checkpoint errors

### If agent not ingesting:
1. Verify agent is running: `ps aux | grep dspy-agent`
2. Check agent logs for errors
3. Verify Kafka topic connectivity from agent

## Quick Health Check Commands

```bash
# Overall system status
docker compose ps

# Kafka health
docker compose exec kafka kafka-topics.sh --bootstrap-server localhost:9092 --list

# Recent activity
tail -20 logs/agent_actions.jsonl
ls -lt vectorized/embeddings/ | head -5

# Consumer activity
docker compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 --all-groups --list
```
