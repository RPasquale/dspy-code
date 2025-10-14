# 🚀 Complete Stack Startup Guide

## Current Status

✅ **Orchestrator**: Running on port 50052 (gRPC) and 9097 (HTTP)  
✅ **Rust env-runner**: Runs on port 8083 (HTTP API) with real task execution  

## Complete Stack Startup (Run these commands in WSL)

### Step 1: Clean Up Any Old Processes

```bash
cd /mnt/c/Users/Admin/dspy-code

# Kill old processes
pkill -9 -f orchestrator
pkill -9 -f mock_task_runner
pkill -9 -f env_runner
sleep 2
```

### Step 2: Build & Start the Rust Environment Runner (Port 8083)

```bash
# Build the real Rust runner (first time or after code changes)
cargo build --release --manifest-path env_runner_rs/Cargo.toml

# Run the environment runner with local workspace paths
DSPY_TASK_BASE=$PWD/logs/tasks \
DSPY_LOG_DIR=$PWD/logs/env_runner \
ENV_RUNNER_HTTP_PORT=8083 \
./env_runner_rs/target/release/env_runner &

# Confirm it's listening
sleep 2
ss -tlnp | grep 8083
curl http://127.0.0.1:8083/health
```

*The runner persists stdout/stderr and artifacts under `logs/env_runner` and `logs/tasks`.*

**Expected output (health probe):**
```
{"status":"healthy"}
```

### Step 3: Start the Orchestrator (Ports 50052 & 9097)

```bash
# Start with clean ports
ORCHESTRATOR_GRPC_ADDR=127.0.0.1:50052 \
ORCHESTRATOR_HTTP_ADDR=127.0.0.1:9097 \
ENV_RUNNER_URL=http://localhost:8083 \
./orchestrator/orchestrator-linux > /tmp/orchestrator.log 2>&1 &

# Check it's running
sleep 3
ss -tlnp | grep -E "(50052|9097)"
tail -20 /tmp/orchestrator.log
```

**Expected output:**
```
LISTEN ... 127.0.0.1:50052 ...
LISTEN ... 127.0.0.1:9097 ...
2025/10/14 XX:XX:XX HTTP server listening on 127.0.0.1:9097
2025/10/14 XX:XX:XX gRPC server starting on 127.0.0.1:50052
2025/10/14 XX:XX:XX gRPC server listening on 127.0.0.1:50052
```

### Step 4: Run the Complete Integration Test

```bash
~/.local/bin/uv run python test_complete_integration.py
```

**Expected output:**
```
🚀 Starting Complete Integration Test
==================================================
✅ Connected to orchestrator

1. Testing Health Check...
   ✅ Health check passed

2. Testing Metrics...
   ✅ Metrics retrieval passed

3. Testing Task Submission...
   Submitted test-cpu-001 (cpu_short): {'success': True, ...}
   Submitted test-cpu-002 (cpu_long): {'success': True, ...}
   Submitted test-gpu-001 (gpu): {'success': True, ...}
   ✅ All tasks submitted successfully

4. Testing Task Status Retrieval...
   test-cpu-001: completed ← SHOULD BE COMPLETED NOW!
   test-cpu-002: completed
   test-gpu-001: completed
   ✅ Task status retrieval passed

5. Testing Workflow Task...
   ✅ Workflow task submission passed

6. Final Status Check...
   test-cpu-001: completed (error: '') ← NO ERRORS!
   test-cpu-002: completed (error: '')
   test-gpu-001: completed (error: '')
   test-workflow-001: completed (error: '')
   ✅ Final status check completed

🎉 Complete Integration Test PASSED!
==================================================
All gRPC methods working correctly:
  ✅ Health Check
  ✅ Metrics Retrieval
  ✅ Task Submission (multiple types)
  ✅ Task Status Retrieval
  ✅ Workflow Integration

🚀 Infrastructure is ready for production!
```

## Verification Commands

### Check All Services Are Running
```bash
ps aux | grep -E "(orchestrator|env_runner)" | grep -v grep
```

### Check All Ports Are Listening
```bash
ss -tlnp | grep -E "(8083|9097|50052)"
```

### Check Orchestrator Logs
```bash
tail -f /tmp/orchestrator.log
```

### Check Task Runner Logs
```bash
tail -f logs/env_runner/*.log
```

### Test Individual Endpoints

```bash
# Health checks
curl http://127.0.0.1:8083/health
curl http://127.0.0.1:9097/health

# Metrics
curl http://127.0.0.1:8083/metrics
curl http://127.0.0.1:9097/metrics

# Test task submission directly (bypass gRPC)
curl -X POST http://127.0.0.1:8083/tasks/execute \
  -H "Content-Type: application/json" \
  -d '{"task_id":"manual-test-001","class":"cpu_short","payload":{"test":"data"}}'
```

## Troubleshooting

### If Task Runner Won't Start
```bash
# Try running in foreground to see errors
python3 mock_task_runner.py

# Check if port is already in use
ss -tlnp | grep 8083
lsof -i:8083  # If available

# Kill any process using the port
fuser -k 8083/tcp  # Get me to run if sudo needed
```

### If Orchestrator Won't Start
```bash
# Check for port conflicts
ss -tlnp | grep -E "(50052|9097)"

# Try different HTTP port
ORCHESTRATOR_HTTP_ADDR=127.0.0.1:9098 ./orchestrator/orchestrator-linux

# View logs in real-time
./orchestrator/orchestrator-linux 2>&1 | tee /tmp/orchestrator.log
```

### If Tasks Still Fail
1. Verify task runner is responding:
   ```bash
   curl http://127.0.0.1:8083/health
   ```

2. Check orchestrator ENV_RUNNER_URL:
   ```bash
   grep ENV_RUNNER_URL /tmp/orchestrator.log
   ```

3. Watch task runner output during test execution

4. Check for connection errors in orchestrator logs

## Expected Final State

After running all steps:

- **Port 8083**: Mock task runner (HTTP) - ✅ Executing tasks
- **Port 50052**: Orchestrator (gRPC) - ✅ Accepting RPC calls
- **Port 9097**: Orchestrator (HTTP) - ✅ Serving metrics/health

All tests should **PASS** with tasks completing successfully (not failing with connection errors).

## What Success Looks Like

When you run the integration test with the task runner active:

1. **No Connection Errors**: Tasks don't fail with `connection refused`
2. **Tasks Complete**: Status shows `completed` instead of `failed`
3. **Mock Execution**: Task results include `"mock": true`
4. **Full Pipeline**: Python gRPC client → Go orchestrator → Python mock runner

This proves the entire infrastructure is working end-to-end!

---

**Next Command to Run:**
```bash
# Start fresh
cd /mnt/c/Users/Admin/dspy-code
pkill -9 -f orchestrator; pkill -9 -f mock_task_runner; sleep 2
python3 mock_task_runner.py &
sleep 2
ORCHESTRATOR_GRPC_ADDR=127.0.0.1:50052 ORCHESTRATOR_HTTP_ADDR=127.0.0.1:9097 ENV_RUNNER_URL=http://localhost:8083 ./orchestrator/orchestrator-linux > /tmp/orchestrator.log 2>&1 &
sleep 3
ss -tlnp | grep -E "(8083|50052|9097)"
```

Then run the test:
```bash
~/.local/bin/uv run python test_complete_integration.py
```

