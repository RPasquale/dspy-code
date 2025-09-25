package metrics

import (
	"context"
	"testing"

	"github.com/dspy/orchestrator/pkg/telemetry"
)

func TestRegistrySourceSample(t *testing.T) {
	reg := telemetry.NewRegistry()
	queue := telemetry.NewGauge(reg, "env_queue_depth", "")
	gpu := telemetry.NewGauge(reg, "gpu_wait_seconds", "")
	errg := telemetry.NewGauge(reg, "env_error_rate", "")
	queue.Set(42)
	gpu.Set(1.5)
	errg.Set(0.02)

	source := &RegistrySource{Registry: reg, QueueMetric: "env_queue_depth", GPUMetric: "gpu_wait_seconds", ErrorMetric: "env_error_rate"}
	snap, sampleErr := source.Sample(context.Background())
	if sampleErr != nil {
		t.Fatalf("sample failed: %v", sampleErr)
	}
	if snap.QueueDepth != 42 {
		t.Fatalf("expected queue depth 42, got %v", snap.QueueDepth)
	}
	if snap.GPUWaitSeconds != 1.5 {
		t.Fatalf("expected gpu wait 1.5, got %v", snap.GPUWaitSeconds)
	}
	if snap.ErrorRate != 0.02 {
		t.Fatalf("expected error rate 0.02, got %v", snap.ErrorRate)
	}
}
