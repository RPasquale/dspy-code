package workflow

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/dspy/orchestrator/internal/limiter"
	"github.com/dspy/orchestrator/internal/metrics"
	"github.com/dspy/orchestrator/pkg/errgroup"
	"github.com/dspy/orchestrator/pkg/telemetry"
)

// TaskFunc represents a unit of orchestration work.
type TaskFunc func(ctx context.Context) error

// Config defines tuning parameters for the orchestrator.
type Config struct {
	BaseLimit          int
	MinLimit           int
	MaxLimit           int
	IncreaseStep       int
	DecreaseStep       int
	QueueHighWatermark float64
	GPUWaitHigh        float64
	ErrorRateHigh      float64
	AdaptationInterval time.Duration
}

// Orchestrator coordinates task execution with structured concurrency and adaptive throttling.
type Orchestrator struct {
	cfg    Config
	source metrics.Source

	ctx    context.Context
	cancel context.CancelFunc

	group    *errgroup.Group
	groupCtx context.Context

	limiter *limiter.AdaptiveLimiter

	registry      *telemetry.Registry
	limitGauge    *telemetry.Gauge
	inflightGauge *telemetry.Gauge
	errorCounter  *telemetry.CounterVec
	mu            sync.Mutex
	closed        bool
}

// New creates a new orchestrator instance.
func New(parent context.Context, cfg Config, source metrics.Source, registry *telemetry.Registry) (*Orchestrator, error) {
	if cfg.MaxLimit > 0 && cfg.MinLimit > cfg.MaxLimit {
		return nil, fmt.Errorf("min limit %d greater than max limit %d", cfg.MinLimit, cfg.MaxLimit)
	}
	if cfg.BaseLimit < cfg.MinLimit && cfg.MinLimit > 0 {
		return nil, fmt.Errorf("base limit %d lower than min limit %d", cfg.BaseLimit, cfg.MinLimit)
	}
	if cfg.BaseLimit > cfg.MaxLimit && cfg.MaxLimit > 0 {
		return nil, fmt.Errorf("base limit %d higher than max limit %d", cfg.BaseLimit, cfg.MaxLimit)
	}
	if cfg.AdaptationInterval <= 0 {
		cfg.AdaptationInterval = 2 * time.Second
	}

	ctx := parent
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)
	group, groupCtx := errgroup.WithContext(ctx)
	initial := cfg.BaseLimit
	if initial == 0 {
		initial = cfg.MinLimit
	}
	lim := limiter.NewAdaptiveLimiter(initial)

	reg := registry
	if reg == nil {
		reg = telemetry.NewRegistry()
	}
	limitGauge := telemetry.NewGauge(reg, "orchestrator_concurrency_limit", "Current orchestration concurrency limit.")
	limitGauge.Set(float64(initial))
	inflightGauge := telemetry.NewGauge(reg, "orchestrator_inflight_tasks", "Number of in-flight orchestrated tasks.")
	errorCounter := telemetry.NewCounterVec(reg, "orchestrator_task_errors_total", "Count of orchestrated tasks that returned an error.", []string{"task"})

	o := &Orchestrator{cfg: cfg, source: source, ctx: ctx, cancel: cancel, group: group, groupCtx: groupCtx, limiter: lim, registry: reg, limitGauge: limitGauge, inflightGauge: inflightGauge, errorCounter: errorCounter}
	o.group.Go(func() error { return o.adaptationLoop() })
	return o, nil
}

// Registry exposes the underlying telemetry registry used by the orchestrator.
func (o *Orchestrator) Registry() *telemetry.Registry { return o.registry }

// Go schedules a task respecting the adaptive limiter and structured concurrency semantics.
func (o *Orchestrator) Go(name string, fn TaskFunc) {
	o.mu.Lock()
	if o.closed {
		o.mu.Unlock()
		return
	}
	o.mu.Unlock()
	o.group.Go(func() error {
		if err := o.limiter.Acquire(o.groupCtx); err != nil {
			return err
		}
		o.inflightGauge.Inc()
		defer func() { o.inflightGauge.Dec(); o.limiter.Release() }()
		taskCtx, cancel := context.WithCancel(o.groupCtx)
		defer cancel()
		if err := fn(taskCtx); err != nil {
			o.errorCounter.WithLabelValues(name).Inc()
			return err
		}
		return nil
	})
}

// adaptationLoop periodically samples metrics and resizes the limiter.
func (o *Orchestrator) adaptationLoop() error {
	ticker := time.NewTicker(o.cfg.AdaptationInterval)
	defer ticker.Stop()
	for {
		select {
		case <-o.groupCtx.Done():
			return o.groupCtx.Err()
		case <-ticker.C:
			o.evaluate()
		}
	}
}

func (o *Orchestrator) evaluate() {
	if o.source == nil {
		return
	}
	snap, err := o.source.Sample(o.groupCtx)
	if err != nil {
		return
	}
	current := o.limiter.Limit()
	newLimit := current
	if snap.QueueDepth >= o.cfg.QueueHighWatermark && (o.cfg.GPUWaitHigh == 0 || snap.GPUWaitSeconds <= o.cfg.GPUWaitHigh) && (o.cfg.ErrorRateHigh == 0 || snap.ErrorRate <= o.cfg.ErrorRateHigh) {
		next := current + o.cfg.IncreaseStep
		if o.cfg.MaxLimit > 0 && next > o.cfg.MaxLimit {
			next = o.cfg.MaxLimit
		}
		newLimit = next
	}
	if (o.cfg.GPUWaitHigh > 0 && snap.GPUWaitSeconds > o.cfg.GPUWaitHigh) || (o.cfg.ErrorRateHigh > 0 && snap.ErrorRate > o.cfg.ErrorRateHigh) {
		next := current - o.cfg.DecreaseStep
		if o.cfg.MinLimit > 0 && next < o.cfg.MinLimit {
			next = o.cfg.MinLimit
		}
		if next < 0 {
			next = 0
		}
		newLimit = next
	}
	if o.cfg.MinLimit > 0 && newLimit < o.cfg.MinLimit {
		newLimit = o.cfg.MinLimit
	}
	if o.cfg.MaxLimit > 0 && newLimit > o.cfg.MaxLimit {
		newLimit = o.cfg.MaxLimit
	}
	if newLimit != current {
		o.limiter.Resize(newLimit)
		o.limitGauge.Set(float64(newLimit))
	}
}

// Shutdown cancels all running tasks and waits for completion.
func (o *Orchestrator) Shutdown() error {
	o.mu.Lock()
	if o.closed {
		o.mu.Unlock()
		return nil
	}
	o.closed = true
	o.mu.Unlock()
	o.cancel()
	o.limiter.Close()
	return o.group.Wait()
}

// Wait blocks until the orchestrator has no more tasks or the context is cancelled.
func (o *Orchestrator) Wait() error { return o.group.Wait() }
