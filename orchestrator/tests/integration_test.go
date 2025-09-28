package tests

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/dspy/orchestrator/internal/metrics"
	"github.com/dspy/orchestrator/internal/workflow"
	"github.com/dspy/orchestrator/pkg/telemetry"
)

// TestOrchestratorIntegration tests the complete orchestrator functionality
func TestOrchestratorIntegration(t *testing.T) {
	// Create test registry
	registry := telemetry.NewRegistry()

	// Setup metrics
	queueGauge := telemetry.NewGauge(registry, "test_queue_depth", "Test queue depth")
	gpuWaitGauge := telemetry.NewGauge(registry, "test_gpu_wait_seconds", "Test GPU wait time")
	errorGauge := telemetry.NewGauge(registry, "test_error_rate", "Test error rate")

	// Create metrics source
	source := &metrics.RegistrySource{
		Registry:    registry,
		QueueMetric: "test_queue_depth",
		GPUMetric:   "test_gpu_wait_seconds",
		ErrorMetric: "test_error_rate",
	}

	// Configure orchestrator
	cfg := workflow.Config{
		BaseLimit:          2,
		MinLimit:           1,
		MaxLimit:           8,
		IncreaseStep:       1,
		DecreaseStep:       1,
		QueueHighWatermark: 10,
		GPUWaitHigh:        2,
		ErrorRateHigh:      0.1,
		AdaptationInterval: 100 * time.Millisecond,
	}

	// Create orchestrator
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	orchestrator, err := workflow.New(ctx, cfg, source, registry)
	if err != nil {
		t.Fatalf("Failed to create orchestrator: %v", err)
	}
	defer orchestrator.Shutdown()

	// Test basic task execution
	t.Run("BasicTaskExecution", func(t *testing.T) {
		var wg sync.WaitGroup
		results := make(chan int, 10)

		for i := 0; i < 10; i++ {
			wg.Add(1)
			orchestrator.Go("test_task", func(ctx context.Context) error {
				defer wg.Done()
				select {
				case results <- 1:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			})
		}

		wg.Wait()
		close(results)

		count := 0
		for range results {
			count++
		}

		if count != 10 {
			t.Errorf("Expected 10 results, got %d", count)
		}
	})

	// Test adaptive scaling
	t.Run("AdaptiveScaling", func(t *testing.T) {
		// Simulate high queue depth to trigger scaling up
		queueGauge.Set(20)
		time.Sleep(200 * time.Millisecond) // Wait for adaptation

		// Simulate high GPU wait to trigger scaling down
		gpuWaitGauge.Set(5)
		time.Sleep(200 * time.Millisecond) // Wait for adaptation

		// Simulate high error rate to trigger scaling down
		errorGauge.Set(0.5)
		time.Sleep(200 * time.Millisecond) // Wait for adaptation
	})

	// Test context cancellation
	t.Run("ContextCancellation", func(t *testing.T) {
		subCtx, subCancel := context.WithCancel(context.Background())
		defer subCancel()

		subOrchestrator, err := workflow.New(subCtx, cfg, source, registry)
		if err != nil {
			t.Fatalf("failed to create orchestrator for cancellation test: %v", err)
		}
		defer subOrchestrator.Shutdown()

		var wg sync.WaitGroup
		errors := make(chan error, 5)

		for i := 0; i < 5; i++ {
			wg.Add(1)
			subOrchestrator.Go("cancelled_task", func(ctx context.Context) error {
				defer wg.Done()
				select {
				case <-time.After(1 * time.Second):
					return nil
				case <-ctx.Done():
					errors <- ctx.Err()
					return ctx.Err()
				}
			})
		}

		// Cancel after a short delay
		go func() {
			time.Sleep(100 * time.Millisecond)
			subCancel()
		}()

		wg.Wait()
		close(errors)

		// Check that tasks were cancelled
		cancelledCount := 0
		for err := range errors {
			if err == context.Canceled {
				cancelledCount++
			}
		}

		if cancelledCount == 0 {
			t.Error("Expected some tasks to be cancelled")
		}
	})

	// Test error handling
	t.Run("ErrorHandling", func(t *testing.T) {
		var wg sync.WaitGroup
		errors := make(chan error, 3)

		for i := 0; i < 3; i++ {
			wg.Add(1)
			orchestrator.Go("error_task", func(ctx context.Context) error {
				defer wg.Done()
				errors <- fmt.Errorf("test error")
				return fmt.Errorf("test error")
			})
		}

		wg.Wait()
		close(errors)

		errorCount := 0
		for range errors {
			errorCount++
		}

		if errorCount != 3 {
			t.Errorf("Expected 3 errors, got %d", errorCount)
		}
	})
}

// TestMetricsExposition tests the metrics HTTP endpoint
func TestMetricsExposition(t *testing.T) {
	registry := telemetry.NewRegistry()

	// Add some test metrics
	gauge := telemetry.NewGauge(registry, "test_gauge", "Test gauge metric")
	gauge.Set(42.5)

	counter := telemetry.NewCounterVec(registry, "test_counter", "Test counter metric", []string{"label"})
	counter.WithLabelValues("value1").Add(10)
	counter.WithLabelValues("value2").Add(20)

	// Create HTTP handler
	handler := registry.Handler()

	// Test HTTP request
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	// Check response
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	body := w.Body.String()
	if body == "" {
		t.Error("Expected non-empty response body")
	}

	// Check for gauge metric
	if !contains(body, "test_gauge 42.5") {
		t.Error("Expected gauge metric in response")
	}

	// Check for counter metrics
	if !contains(body, "test_counter{label=\"value1\"} 10") {
		t.Error("Expected counter metric with value1 in response")
	}

	if !contains(body, "test_counter{label=\"value2\"} 20") {
		t.Error("Expected counter metric with value2 in response")
	}
}

// TestConcurrentAccess tests concurrent access to the orchestrator
func TestConcurrentAccess(t *testing.T) {
	registry := telemetry.NewRegistry()
	source := &metrics.RegistrySource{
		Registry:    registry,
		QueueMetric: "test_queue_depth",
		GPUMetric:   "test_gpu_wait_seconds",
		ErrorMetric: "test_error_rate",
	}

	cfg := workflow.Config{
		BaseLimit:          4,
		MinLimit:           1,
		MaxLimit:           16,
		IncreaseStep:       2,
		DecreaseStep:       2,
		QueueHighWatermark: 50,
		GPUWaitHigh:        5,
		ErrorRateHigh:      0.3,
		AdaptationInterval: 50 * time.Millisecond,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	orchestrator, err := workflow.New(ctx, cfg, source, registry)
	if err != nil {
		t.Fatalf("Failed to create orchestrator: %v", err)
	}
	defer orchestrator.Shutdown()

	// Test concurrent task submission
	var wg sync.WaitGroup
	results := make(chan int, 100)

	for i := 0; i < 100; i++ {
		wg.Add(1)
		orchestrator.Go("concurrent_task", func(ctx context.Context) error {
			defer wg.Done()
			select {
			case results <- 1:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
	}

	wg.Wait()
	close(results)

	count := 0
	for range results {
		count++
	}

	if count != 100 {
		t.Errorf("Expected 100 results, got %d", count)
	}
}

// TestResourceLimits tests resource limit enforcement
func TestResourceLimits(t *testing.T) {
	registry := telemetry.NewRegistry()
	source := &metrics.RegistrySource{
		Registry:    registry,
		QueueMetric: "test_queue_depth",
		GPUMetric:   "test_gpu_wait_seconds",
		ErrorMetric: "test_error_rate",
	}

	cfg := workflow.Config{
		BaseLimit:          2,
		MinLimit:           1,
		MaxLimit:           4,
		IncreaseStep:       1,
		DecreaseStep:       1,
		QueueHighWatermark: 10,
		GPUWaitHigh:        2,
		ErrorRateHigh:      0.1,
		AdaptationInterval: 100 * time.Millisecond,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	orchestrator, err := workflow.New(ctx, cfg, source, registry)
	if err != nil {
		t.Fatalf("Failed to create orchestrator: %v", err)
	}
	defer orchestrator.Shutdown()

	// Test that we can't exceed the limit
	var wg sync.WaitGroup
	results := make(chan int, 10)

	for i := 0; i < 10; i++ {
		wg.Add(1)
		orchestrator.Go("limited_task", func(ctx context.Context) error {
			defer wg.Done()
			select {
			case results <- 1:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
	}

	wg.Wait()
	close(results)

	count := 0
	for range results {
		count++
	}

	// Should have executed all tasks (orchestrator handles queuing internally)
	if count != 10 {
		t.Errorf("Expected 10 results, got %d", count)
	}
}

// TestMetricsSampling tests metrics sampling and adaptation
func TestMetricsSampling(t *testing.T) {
	registry := telemetry.NewRegistry()

	// Setup metrics
	queueGauge := telemetry.NewGauge(registry, "test_queue_depth", "Test queue depth")
	gpuWaitGauge := telemetry.NewGauge(registry, "test_gpu_wait_seconds", "Test GPU wait time")
	errorGauge := telemetry.NewGauge(registry, "test_error_rate", "Test error rate")

	source := &metrics.RegistrySource{
		Registry:    registry,
		QueueMetric: "test_queue_depth",
		GPUMetric:   "test_gpu_wait_seconds",
		ErrorMetric: "test_error_rate",
	}

	cfg := workflow.Config{
		BaseLimit:          2,
		MinLimit:           1,
		MaxLimit:           8,
		IncreaseStep:       1,
		DecreaseStep:       1,
		QueueHighWatermark: 5,
		GPUWaitHigh:        2,
		ErrorRateHigh:      0.1,
		AdaptationInterval: 100 * time.Millisecond,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	orchestrator, err := workflow.New(ctx, cfg, source, registry)
	if err != nil {
		t.Fatalf("Failed to create orchestrator: %v", err)
	}
	defer orchestrator.Shutdown()

	// Test scaling up
	queueGauge.Set(10) // Above threshold
	time.Sleep(200 * time.Millisecond)

	// Test scaling down due to GPU wait
	gpuWaitGauge.Set(5) // Above threshold
	time.Sleep(200 * time.Millisecond)

	// Test scaling down due to error rate
	errorGauge.Set(0.5) // Above threshold
	time.Sleep(200 * time.Millisecond)

	// The orchestrator should have adapted to these conditions
	// (We can't easily test the internal state, but we can verify it doesn't crash)
}

// TestGracefulShutdown tests graceful shutdown behavior
func TestGracefulShutdown(t *testing.T) {
	registry := telemetry.NewRegistry()
	source := &metrics.RegistrySource{
		Registry:    registry,
		QueueMetric: "test_queue_depth",
		GPUMetric:   "test_gpu_wait_seconds",
		ErrorMetric: "test_error_rate",
	}

	cfg := workflow.Config{
		BaseLimit:          2,
		MinLimit:           1,
		MaxLimit:           4,
		IncreaseStep:       1,
		DecreaseStep:       1,
		QueueHighWatermark: 10,
		GPUWaitHigh:        2,
		ErrorRateHigh:      0.1,
		AdaptationInterval: 100 * time.Millisecond,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	orchestrator, err := workflow.New(ctx, cfg, source, registry)
	if err != nil {
		t.Fatalf("Failed to create orchestrator: %v", err)
	}

	// Start some tasks
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		orchestrator.Go("shutdown_task", func(ctx context.Context) error {
			defer wg.Done()
			select {
			case <-time.After(1 * time.Second):
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
	}

	// Shutdown after a short delay
	go func() {
		time.Sleep(100 * time.Millisecond)
		orchestrator.Shutdown()
	}()

	// Wait for tasks to complete or be cancelled
	wg.Wait()
}

// Helper function to check if a string contains a substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || contains(s[1:], substr)))
}
