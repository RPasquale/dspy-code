package workflow

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/dspy/orchestrator/internal/metrics"
	"github.com/dspy/orchestrator/pkg/telemetry"
)

type stubSource struct{ value atomic.Value }

func newStubSource(initial metrics.Snapshot) *stubSource {
	s := &stubSource{}
	s.value.Store(initial)
	return s
}
func (s *stubSource) Sample(_ context.Context) (metrics.Snapshot, error) {
	v := s.value.Load()
	if v == nil {
		return metrics.Snapshot{}, errors.New("no snapshot")
	}
	return v.(metrics.Snapshot), nil
}
func (s *stubSource) set(v metrics.Snapshot) { s.value.Store(v) }

func TestOrchestratorAdaptiveResizes(t *testing.T) {
	reg := telemetry.NewRegistry()
	source := newStubSource(metrics.Snapshot{})
	cfg := Config{BaseLimit: 2, MinLimit: 1, MaxLimit: 5, IncreaseStep: 1, DecreaseStep: 2, QueueHighWatermark: 10, GPUWaitHigh: 4, ErrorRateHigh: 0.2, AdaptationInterval: 10 * time.Second}
	o, err := New(context.Background(), cfg, source, reg)
	if err != nil {
		t.Fatalf("new orchestrator err: %v", err)
	}
	t.Cleanup(func() { _ = o.Shutdown() })
	if got := o.limiter.Limit(); got != 2 {
		t.Fatalf("expected initial limit 2, got %d", got)
	}
	source.set(metrics.Snapshot{QueueDepth: 20})
	o.evaluate()
	if got := o.limiter.Limit(); got != 3 {
		t.Fatalf("expected limit 3, got %d", got)
	}
	source.set(metrics.Snapshot{GPUWaitSeconds: 6})
	o.evaluate()
	if got := o.limiter.Limit(); got != 1 {
		t.Fatalf("expected limit 1, got %d", got)
	}
	source.set(metrics.Snapshot{QueueDepth: 30})
	o.evaluate()
	if got := o.limiter.Limit(); got != 2 {
		t.Fatalf("expected limit 2, got %d", got)
	}
}

func TestOrchestratorStructuredCancellation(t *testing.T) {
	reg := telemetry.NewRegistry()
	source := newStubSource(metrics.Snapshot{})
    cfg := Config{BaseLimit: 3, MinLimit: 1, MaxLimit: 4, IncreaseStep: 1, DecreaseStep: 1, QueueHighWatermark: 10, AdaptationInterval: time.Second}
	o, err := New(context.Background(), cfg, source, reg)
	if err != nil {
		t.Fatalf("create orchestrator: %v", err)
	}
    done := make(chan struct{})
    startedA := make(chan struct{})
    startedC := make(chan struct{})
    o.Go("task-a", func(ctx context.Context) error { close(startedA); <-ctx.Done(); close(done); return ctx.Err() })
    o.Go("task-b", func(ctx context.Context) error { <-ctx.Done(); return ctx.Err() })
    o.Go("task-c", func(ctx context.Context) error { close(startedC); <-ctx.Done(); return ctx.Err() })
    select { case <-startedA: case <-time.After(500 * time.Millisecond): t.Fatalf("task-a did not start") }
    select { case <-startedC: case <-time.After(500 * time.Millisecond): t.Fatalf("task-c did not start") }
    err = o.Shutdown(); if err != nil && !errors.Is(err, context.Canceled) { t.Fatalf("expected context cancelled error, got %v", err) }
    select { case <-done: case <-time.After(1000 * time.Millisecond): t.Fatalf("task not cancelled by shutdown") }
}

func TestOrchestratorWaitReturnsTaskError(t *testing.T) {
	reg := telemetry.NewRegistry()
	source := newStubSource(metrics.Snapshot{})
	cfg := Config{BaseLimit: 1, MinLimit: 1, MaxLimit: 2, AdaptationInterval: time.Second}
	o, err := New(context.Background(), cfg, source, reg)
	if err != nil {
		t.Fatalf("new orchestrator: %v", err)
	}
	o.Go("fail", func(ctx context.Context) error { return errors.New("boom") })
	err = o.Wait()
	if err == nil {
		t.Fatalf("expected errgroup to return error")
	}
	_ = o.Shutdown()
}
