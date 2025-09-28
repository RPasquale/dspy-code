package infermesh

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/dspy/orchestrator/internal/limiter"
	"github.com/dspy/orchestrator/internal/metrics"
	"github.com/dspy/orchestrator/pkg/telemetry"
)

// InferMeshOrchestrator coordinates InferMesh embedding requests with adaptive concurrency
type InferMeshOrchestrator struct {
	client      *InferMeshClient
	limiter     *limiter.AdaptiveLimiter
	metrics     *InferMeshMetrics
	registry    *telemetry.Registry
	mu          sync.RWMutex
	closed      bool
}

// InferMeshMetrics tracks InferMesh-specific performance metrics
type InferMeshMetrics struct {
	TotalRequests      int64
	SuccessfulRequests int64
	FailedRequests     int64
	AverageLatency     time.Duration
	QueueDepth         int64
	ConcurrentRequests int64
	BatchEfficiency    float64
	ErrorRate          float64
}

// NewInferMeshOrchestrator creates a new InferMesh orchestrator with adaptive concurrency
func NewInferMeshOrchestrator(
	baseURL, apiKey, model string,
	registry *telemetry.Registry,
	initialLimit int,
) *InferMeshOrchestrator {
	client := NewInferMeshClient(baseURL, apiKey, model)
	limiter := limiter.NewAdaptiveLimiter(initialLimit)

	// Create InferMesh-specific metrics
	queueDepthGauge := telemetry.NewGauge(registry, "infermesh_queue_depth", "Number of queued InferMesh requests")
	latencyGauge := telemetry.NewGauge(registry, "infermesh_latency_seconds", "Average InferMesh request latency")
	errorRateGauge := telemetry.NewGauge(registry, "infermesh_error_rate", "InferMesh error rate")
	throughputGauge := telemetry.NewGauge(registry, "infermesh_throughput_requests_per_second", "InferMesh requests per second")

	orchestrator := &InferMeshOrchestrator{
		client:   client,
		limiter: limiter,
		metrics:  &InferMeshMetrics{},
		registry: registry,
	}

	// Start metrics update goroutine
	go orchestrator.updateMetricsLoop(queueDepthGauge, latencyGauge, errorRateGauge, throughputGauge)

	return orchestrator
}

// Embed processes embedding requests with adaptive concurrency control
func (imo *InferMeshOrchestrator) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	imo.mu.RLock()
	if imo.closed {
		imo.mu.RUnlock()
		return nil, fmt.Errorf("orchestrator is closed")
	}
	imo.mu.RUnlock()

	// Acquire concurrency limit
	if err := imo.limiter.Acquire(ctx); err != nil {
		imo.updateErrorMetrics()
		return nil, fmt.Errorf("failed to acquire concurrency limit: %w", err)
	}
	defer imo.limiter.Release()

	// Update metrics
	imo.mu.Lock()
	imo.metrics.TotalRequests++
	imo.metrics.ConcurrentRequests++
	imo.mu.Unlock()

	defer func() {
		imo.mu.Lock()
		imo.metrics.ConcurrentRequests--
		imo.mu.Unlock()
	}()

	// Process embedding request
	startTime := time.Now()
	embeddings, err := imo.client.Embed(ctx, texts)
	latency := time.Since(startTime)

	// Update metrics
	imo.mu.Lock()
	if err != nil {
		imo.metrics.FailedRequests++
		imo.metrics.ErrorRate = float64(imo.metrics.FailedRequests) / float64(imo.metrics.TotalRequests)
	} else {
		imo.metrics.SuccessfulRequests++
		imo.metrics.AverageLatency = (imo.metrics.AverageLatency + latency) / 2
	}
	imo.mu.Unlock()

	return embeddings, err
}

// EmbedBatch processes large batches with optimized concurrency
func (imo *InferMeshOrchestrator) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	imo.mu.RLock()
	if imo.closed {
		imo.mu.RUnlock()
		return nil, fmt.Errorf("orchestrator is closed")
	}
	imo.mu.RUnlock()

	// For large batches, we might need multiple concurrency slots
	batchSize := len(texts)
	requiredSlots := (batchSize + 511) / 512 // Assume 512 texts per slot

	for i := 0; i < requiredSlots; i++ {
		if err := imo.limiter.Acquire(ctx); err != nil {
			imo.updateErrorMetrics()
			return nil, fmt.Errorf("failed to acquire concurrency limit for batch: %w", err)
		}
		defer imo.limiter.Release()
	}

	// Update metrics
	imo.mu.Lock()
	imo.metrics.TotalRequests++
	imo.metrics.ConcurrentRequests++
	imo.mu.Unlock()

	defer func() {
		imo.mu.Lock()
		imo.metrics.ConcurrentRequests--
		imo.mu.Unlock()
	}()

	// Process batch
	startTime := time.Now()
	embeddings, err := imo.client.EmbedBatch(ctx, texts)
	latency := time.Since(startTime)

	// Update metrics
	imo.mu.Lock()
	if err != nil {
		imo.metrics.FailedRequests++
		imo.metrics.ErrorRate = float64(imo.metrics.FailedRequests) / float64(imo.metrics.TotalRequests)
	} else {
		imo.metrics.SuccessfulRequests++
		imo.metrics.AverageLatency = (imo.metrics.AverageLatency + latency) / 2
		imo.metrics.BatchEfficiency = float64(len(texts)) / float64(batchSize)
	}
	imo.mu.Unlock()

	return embeddings, err
}

// AdaptConcurrency adjusts concurrency limits based on InferMesh performance
func (imo *InferMeshOrchestrator) AdaptConcurrency() {
	imo.mu.RLock()
	metrics := *imo.metrics
	imo.mu.RUnlock()

	currentLimit := imo.limiter.Limit()
	newLimit := currentLimit

	// Increase concurrency if:
	// - Error rate is low (< 5%)
	// - Latency is reasonable (< 2 seconds)
	// - We have capacity
	if metrics.ErrorRate < 0.05 && metrics.AverageLatency < 2*time.Second && metrics.ConcurrentRequests < int64(currentLimit) {
		newLimit = currentLimit + 2
		if newLimit > 50 { // Cap at 50 concurrent requests
			newLimit = 50
		}
	}

	// Decrease concurrency if:
	// - Error rate is high (> 10%)
	// - Latency is high (> 5 seconds)
	// - We're hitting limits
	if metrics.ErrorRate > 0.10 || metrics.AverageLatency > 5*time.Second || metrics.ConcurrentRequests >= int64(currentLimit) {
		newLimit = currentLimit - 2
		if newLimit < 1 {
			newLimit = 1
		}
	}

	if newLimit != currentLimit {
		imo.limiter.Resize(newLimit)
	}
}

// GetMetrics returns current InferMesh metrics
func (imo *InferMeshOrchestrator) GetMetrics() InferMeshMetrics {
	imo.mu.RLock()
	defer imo.mu.RUnlock()
	return *imo.metrics
}

// GetConcurrencyLimit returns the current concurrency limit
func (imo *InferMeshOrchestrator) GetConcurrencyLimit() int {
	return imo.limiter.Limit()
}

// SetConcurrencyLimit manually sets the concurrency limit
func (imo *InferMeshOrchestrator) SetConcurrencyLimit(limit int) {
	imo.limiter.Resize(limit)
}

// updateMetricsLoop continuously updates metrics for monitoring
func (imo *InferMeshOrchestrator) updateMetricsLoop(
	queueDepthGauge, latencyGauge, errorRateGauge, throughputGauge *telemetry.Gauge,
) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		imo.mu.RLock()
		metrics := *imo.metrics
		imo.mu.RUnlock()

		// Update Prometheus metrics
		queueDepthGauge.Set(float64(metrics.QueueDepth))
		latencyGauge.Set(metrics.AverageLatency.Seconds())
		errorRateGauge.Set(metrics.ErrorRate)
		
		// Calculate throughput (requests per second)
		throughput := float64(metrics.SuccessfulRequests) / time.Since(time.Now().Add(-time.Minute)).Seconds()
		throughputGauge.Set(throughput)

		// Adapt concurrency based on metrics
		imo.AdaptConcurrency()
	}
}

// updateErrorMetrics updates error-related metrics
func (imo *InferMeshOrchestrator) updateErrorMetrics() {
	imo.mu.Lock()
	defer imo.mu.Unlock()
	imo.metrics.FailedRequests++
	imo.metrics.ErrorRate = float64(imo.metrics.FailedRequests) / float64(imo.metrics.TotalRequests)
}

// Close shuts down the orchestrator
func (imo *InferMeshOrchestrator) Close() error {
	imo.mu.Lock()
	defer imo.mu.Unlock()
	
	if imo.closed {
		return nil
	}
	
	imo.closed = true
	imo.limiter.Close()
	return imo.client.Close()
}

// CreateInferMeshMetricsSource creates a metrics source for the main orchestrator
func CreateInferMeshMetricsSource(registry *telemetry.Registry) metrics.Source {
	return &InferMeshMetricsSource{
		Registry: registry,
		QueueMetric: "infermesh_queue_depth",
		GPUMetric:   "infermesh_latency_seconds",
		ErrorMetric: "infermesh_error_rate",
	}
}

// InferMeshMetricsSource implements metrics.Source for InferMesh
type InferMeshMetricsSource struct {
	Registry    *telemetry.Registry
	QueueMetric string
	GPUMetric   string
	ErrorMetric string
}

// Sample implements the metrics.Source interface
func (ims *InferMeshMetricsSource) Sample(ctx context.Context) (metrics.Snapshot, error) {
	if ims.Registry == nil {
		return metrics.Snapshot{}, fmt.Errorf("nil registry")
	}

	var snap metrics.Snapshot
	if v, ok := ims.Registry.Value(ims.QueueMetric); ok {
		snap.QueueDepth = v
	}
	if v, ok := ims.Registry.Value(ims.GPUMetric); ok {
		snap.GPUWaitSeconds = v
	}
	if v, ok := ims.Registry.Value(ims.ErrorMetric); ok {
		snap.ErrorRate = v
	}
	return snap, nil
}
