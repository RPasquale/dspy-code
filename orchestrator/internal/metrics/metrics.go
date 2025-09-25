package metrics

import (
	"context"
	"errors"

	"github.com/dspy/orchestrator/pkg/telemetry"
)

// Snapshot captures orchestrator-relevant measurements derived from telemetry metrics.
type Snapshot struct {
	QueueDepth     float64
	GPUWaitSeconds float64
	ErrorRate      float64
}

// Source defines a type capable of producing metric snapshots.
type Source interface {
	Sample(ctx context.Context) (Snapshot, error)
}

// RegistrySource reads metrics from a telemetry registry using well-known metric names.
type RegistrySource struct {
	Registry    *telemetry.Registry
	QueueMetric string
	GPUMetric   string
	ErrorMetric string
}

// Sample implements the Source interface.
func (r *RegistrySource) Sample(_ context.Context) (Snapshot, error) {
	if r.Registry == nil {
		return Snapshot{}, errors.New("nil registry")
	}
	var snap Snapshot
	if v, ok := r.Registry.Value(r.QueueMetric); ok {
		snap.QueueDepth = v
	}
	if v, ok := r.Registry.Value(r.GPUMetric); ok {
		snap.GPUWaitSeconds = v
	}
	if v, ok := r.Registry.Value(r.ErrorMetric); ok {
		snap.ErrorRate = v
	}
	return snap, nil
}
