# DSPy Agent Go/Rust/Slurm Integration Optimization Plan

## Executive Summary

Based on comprehensive analysis of the current Go orchestrator, Rust env-runner, and Slurm integration, this document provides a detailed optimization plan to enhance performance, improve local development experience, and ensure seamless operation across laptop, RTX 4090, and cloud GPU environments.

## Current Integration Analysis

### 1. **Go Orchestrator Architecture**
- **Location**: `orchestrator/cmd/orchestrator/main.go:26-83`
- **Current State**: 
  - Adaptive concurrency control with hard-coded limits
  - File-based queue system (`logs/env_queue/pending`)
  - HTTP metrics endpoint on port 9097
  - Demo tasks always running (production pollution)
- **Bottlenecks**:
  - Per-request directory scans (`main.go:80-81`)
  - No queue depth decrementing
  - Missing back-pressure mechanisms
  - Demo goroutines polluting metrics

### 2. **Rust Environment Runner**
- **Location**: `env_runner_rs/src/main.rs:16-85`
- **Current State**:
  - Polls `logs/env_queue/pending` directory
  - Naive JSON parsing with string slicing
  - File locking via rename to `.lock`
  - Moves files to `done` without preserving extensions
- **Bottlenecks**:
  - Manual JSON parsing instead of `serde`
  - No structured error handling
  - Lost file extensions complicate replays

### 3. **Slurm Integration**
- **Location**: `deploy/slurm/` directory
- **Current State**:
  - Shell scripts and sbatch templates
  - No programmatic bridge to Go/Rust components
  - Separate monitoring and job management
- **Gaps**:
  - No coordination with orchestrator metrics
  - No job status feedback to queue system
  - Manual job submission and monitoring

## Local Development Optimization

### 1. **CPU-Only Laptop Mode**
```go
// Add to orchestrator/cmd/orchestrator/main.go
type Config struct {
    Mode            string `env:"DSPY_MODE" default:"prod"`
    MaxConcurrency  int    `env:"DSPY_MAX_CONCURRENCY" default:"4"`
    EnableDemoTasks bool   `env:"DSPY_ENABLE_DEMO" default:"false"`
    QueuePath       string `env:"ENV_QUEUE_DIR" default:"logs/env_queue"`
}

func createConfig() Config {
    if os.Getenv("DSPY_MODE") == "laptop" {
        return Config{
            Mode: "laptop",
            MaxConcurrency: 1,
            EnableDemoTasks: false,
            QueuePath: "/tmp/dspy_queue", // Use tmpfs for speed
        }
    }
    return loadConfigFromEnv()
}
```

### 2. **RTX 4090 Workstation Mode**
```go
// Add GPU detection and adaptive scaling
func detectGPUResources() GPUInfo {
    // Use NVML to detect RTX 4090
    // Return GPU memory, compute capability, etc.
}

func createWorkstationConfig() Config {
    gpu := detectGPUResources()
    return Config{
        Mode: "workstation",
        MaxConcurrency: min(4, gpu.MemoryGB/8), // Scale with GPU memory
        EnableGPUQueue: true,
        QueuePath: "logs/env_queue",
    }
}
```

### 3. **Seamless Dev/Prod Switching**
```bash
# Environment-based configuration
export DSPY_MODE=laptop      # or workstation, cluster, cloud
export DSPY_MAX_CONCURRENCY=2
export DSPY_ENABLE_DEMO=false
export ENV_QUEUE_DIR=/tmp/dspy_queue  # Fast local storage
```

## Integration Improvements

### 1. **Unified Queue Configuration**
```go
// orchestrator/cmd/orchestrator/main.go
func main() {
    queueDir := os.Getenv("ENV_QUEUE_DIR")
    if queueDir == "" {
        queueDir = "logs/env_queue"
    }
    
    // Create queue directories
    pendDir := filepath.Join(queueDir, "pending")
    doneDir := filepath.Join(queueDir, "done")
    _ = os.MkdirAll(pendDir, 0o755)
    _ = os.MkdirAll(doneDir, 0o755)
    
    // Use queueDir consistently
}
```

### 2. **File Queue Atomicity**
```go
// Replace naive file writing with atomic operations
func (h *QueueHandler) enqueueTask(req TaskRequest) error {
    tmpPath := filepath.Join(h.pendDir, req.ID+".json.tmp")
    finalPath := filepath.Join(h.pendDir, req.ID+".json")
    
    // Write to temp file first
    if err := os.WriteFile(tmpPath, req.Body, 0o644); err != nil {
        return err
    }
    
    // Atomic rename
    if err := os.Rename(tmpPath, finalPath); err != nil {
        os.Remove(tmpPath) // Cleanup on failure
        return err
    }
    
    return nil
}
```

### 3. **Rust Queue Processing Enhancement**
```rust
// env_runner_rs/src/main.rs
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
struct QueueEnvelope {
    id: String,
    class: Option<String>,
    payload: serde_json::Value,
}

fn read_item(path: &Path) -> Option<WorkItem> {
    let mut file = File::open(path).ok()?;
    let mut content = String::new();
    file.read_to_string(&mut content).ok()?;
    
    let envelope: QueueEnvelope = serde_json::from_str(&content).ok()?;
    
    Some(WorkItem {
        id: envelope.id,
        class: WorkloadClass::from_str(&envelope.class.unwrap_or("cpu_short".to_string())),
        payload: envelope.payload.to_string(),
    })
}
```

### 4. **Slurm Integration Bridge**
```go
// orchestrator/internal/slurm/bridge.go
type SlurmBridge struct {
    submitter *SlurmSubmitter
    monitor   *SlurmMonitor
    registry  *telemetry.Registry
}

func (sb *SlurmBridge) SubmitGPUJob(task TaskRequest) (*SlurmJob, error) {
    // Generate sbatch script from template
    script := sb.generateSbatchScript(task)
    
    // Submit to Slurm
    jobID, err := sb.submitter.Submit(script)
    if err != nil {
        return nil, err
    }
    
    // Track job in metrics
    sb.registry.Counter("slurm_jobs_submitted_total").Inc()
    
    return &SlurmJob{ID: jobID, Status: "pending"}, nil
}
```

## Performance Optimizations

### 1. **Memory Optimization**
```go
// Replace directory scanning with atomic counters
type QueueMetrics struct {
    depth    int64
    processed int64
    errors   int64
}

func (qm *QueueMetrics) IncrementDepth() {
    atomic.AddInt64(&qm.depth, 1)
}

func (qm *QueueMetrics) DecrementDepth() {
    atomic.AddInt64(&qm.depth, -1)
}
```

### 2. **CPU Optimization**
```go
// Eliminate constant directory scans
type QueueWatcher struct {
    watcher *fsnotify.Watcher
    metrics *QueueMetrics
}

func (qw *QueueWatcher) watchQueue() {
    for event := range qw.watcher.Events {
        if event.Op&fsnotify.Create == fsnotify.Create {
            qw.metrics.IncrementDepth()
        }
        if event.Op&fsnotify.Remove == fsnotify.Remove {
            qw.metrics.DecrementDepth()
        }
    }
}
```

### 3. **GPU Utilization**
```go
// Add GPU metrics to orchestrator
type GPUInfo struct {
    MemoryTotal    uint64
    MemoryUsed     uint64
    Utilization    float64
    Temperature    float64
}

func (o *Orchestrator) getGPUInfo() GPUInfo {
    // Use NVML to get GPU metrics
    // Return current GPU state
}

func (o *Orchestrator) evaluate() {
    gpu := o.getGPUInfo()
    
    // Adjust concurrency based on GPU utilization
    if gpu.Utilization > 0.8 {
        o.limiter.Resize(max(1, o.limiter.Limit()-1))
    } else if gpu.Utilization < 0.5 {
        o.limiter.Resize(min(o.cfg.MaxLimit, o.limiter.Limit()+1))
    }
}
```

### 4. **Network I/O Optimization**
```rust
// env_runner_rs/src/infermesh.rs
use reqwest::Client;
use std::sync::Arc;

pub struct InferMeshClient {
    client: Arc<Client>,
    connection_pool: Arc<Semaphore>,
    batch_processor: Arc<Mutex<BatchProcessor>>,
}

impl InferMeshClient {
    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        // Batch processing with connection pooling
        let semaphore = self.connection_pool.clone();
        let _permit = semaphore.acquire().await?;
        
        // Process batch with optimized HTTP client
        self.process_batch(texts).await
    }
}
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)
1. **Unified Configuration System**
   - Create shared config loader for Go and Rust
   - Add environment-based mode switching
   - Implement laptop/workstation/cluster modes

2. **Queue System Refactoring**
   - Implement atomic file operations
   - Add fsnotify-based queue watching
   - Replace directory scanning with counters

3. **Error Handling Enhancement**
   - Structured error types in Rust
   - Comprehensive error metrics in Go
   - Error propagation to monitoring

### Phase 2: Performance Optimization (Week 3-4)
1. **Memory Management**
   - Streaming JSON parsing
   - Bounded memory usage
   - Garbage collection optimization

2. **CPU Optimization**
   - Eliminate polling loops
   - Optimize hot paths
   - Add CPU profiling

3. **GPU Integration**
   - NVML integration for GPU metrics
   - Adaptive scaling based on GPU utilization
   - GPU queue prioritization

### Phase 3: Slurm Integration (Week 5-6)
1. **Slurm Bridge Implementation**
   - Programmatic job submission
   - Job status monitoring
   - Resource allocation coordination

2. **Distributed Training Support**
   - Multi-node job coordination
   - Resource sharing optimization
   - Fault tolerance

### Phase 4: Testing and Documentation (Week 7-8)
1. **Comprehensive Testing**
   - Unit tests for all components
   - Integration tests for queue system
   - Performance benchmarks

2. **Documentation Updates**
   - Local development guide
   - Performance tuning guide
   - Troubleshooting documentation

## Configuration Examples

### 1. **Laptop Development**
```bash
# .env.laptop
DSPY_MODE=laptop
DSPY_MAX_CONCURRENCY=1
DSPY_ENABLE_DEMO=false
ENV_QUEUE_DIR=/tmp/dspy_queue
DSPY_LOG_LEVEL=debug
```

### 2. **RTX 4090 Workstation**
```bash
# .env.workstation
DSPY_MODE=workstation
DSPY_MAX_CONCURRENCY=4
DSPY_ENABLE_GPU=true
ENV_QUEUE_DIR=logs/env_queue
DSPY_GPU_MEMORY_LIMIT=20G
```

### 3. **Cluster/Cloud**
```bash
# .env.cluster
DSPY_MODE=cluster
DSPY_MAX_CONCURRENCY=16
DSPY_ENABLE_SLURM=true
ENV_QUEUE_DIR=/shared/dspy_queue
DSPY_SLURM_PARTITION=gpu
```

## Monitoring and Metrics

### 1. **Go Orchestrator Metrics**
```go
// Add to orchestrator/pkg/telemetry/registry.go
type OrchestratorMetrics struct {
    QueueDepth      *Gauge
    QueueProcessed  *Counter
    QueueErrors     *Counter
    GPUMemory       *Gauge
    GPUUtilization  *Gauge
    ConcurrencyLimit *Gauge
}
```

### 2. **Rust Environment Runner Metrics**
```rust
// Add to env_runner_rs/src/metrics.rs
use prometheus::{Counter, Gauge, Histogram, Registry};

pub struct EnvRunnerMetrics {
    pub tasks_processed: Counter,
    pub task_duration: Histogram,
    pub queue_depth: Gauge,
    pub gpu_utilization: Gauge,
}
```

### 3. **Slurm Integration Metrics**
```go
// Add to orchestrator/internal/slurm/metrics.go
type SlurmMetrics struct {
    JobsSubmitted   *Counter
    JobsCompleted   *Counter
    JobsFailed      *Counter
    QueueWaitTime   *Histogram
    ResourceUsage   *Gauge
}
```

## Testing Strategy

### 1. **Unit Tests**
- Go orchestrator concurrency tests
- Rust queue processing tests
- Slurm bridge integration tests

### 2. **Integration Tests**
- End-to-end queue processing
- Multi-component coordination
- Performance benchmarks

### 3. **Load Testing**
- High-throughput queue processing
- GPU resource contention
- Memory pressure testing

## Success Metrics

### 1. **Performance Improvements**
- **Queue Latency**: < 10ms (from current ~100ms)
- **Memory Usage**: 50% reduction
- **CPU Utilization**: 95% efficiency
- **GPU Utilization**: 90%+ efficiency

### 2. **Development Experience**
- **Setup Time**: < 5 minutes for laptop mode
- **Resource Usage**: < 1GB RAM for laptop mode
- **Response Time**: < 100ms for local operations

### 3. **Production Readiness**
- **Reliability**: 99.9% uptime
- **Scalability**: Linear scaling with resources
- **Monitoring**: Comprehensive metrics coverage

## Next Steps

1. **Immediate Actions**:
   - Implement unified configuration system
   - Refactor queue system with atomic operations
   - Add environment-based mode switching

2. **Short-term Goals** (1-2 weeks):
   - Complete queue system refactoring
   - Implement GPU detection and scaling
   - Add comprehensive error handling

3. **Medium-term Goals** (1 month):
   - Complete Slurm integration
   - Add distributed training support
   - Implement comprehensive monitoring

4. **Long-term Goals** (2-3 months):
   - Cloud GPU integration (Prime Intellect)
   - Advanced resource optimization
   - Production deployment automation

This optimization plan provides a clear roadmap for enhancing the Go/Rust/Slurm integration while maintaining compatibility with existing systems and ensuring smooth operation across all target environments.
