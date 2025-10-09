package queue

import (
	"context"
	"log"
	"path/filepath"
	"sync"
	"time"

	"github.com/dspy/orchestrator/pkg/telemetry"
)

// QueueWatcher monitors queue directory changes using fsnotify
type QueueWatcher struct {
	pendDir    string
	doneDir    string
	registry   *telemetry.Registry
	queueDepth *telemetry.Gauge
	submitted  *telemetry.Counter
	processed  *telemetry.Counter
	completed  *telemetry.Counter
	mu         sync.RWMutex
	stopCh     chan struct{}
	pollEvery  time.Duration
	lastPend   map[string]struct{}
	lastDone   map[string]struct{}
}

// NewQueueWatcher creates a new queue watcher
func NewQueueWatcher(pendDir, doneDir string, registry *telemetry.Registry) (*QueueWatcher, error) {
	return &QueueWatcher{
		pendDir:    pendDir,
		doneDir:    doneDir,
		registry:   registry,
		queueDepth: telemetry.NewGauge(registry, "queue_depth", "Number of pending tasks in queue"),
		submitted:  registry.Counter("queue_tasks_submitted_total"),
		processed:  registry.Counter("queue_tasks_processed_total"),
		completed:  registry.Counter("queue_tasks_completed_total"),
		stopCh:     make(chan struct{}),
		pollEvery:  2 * time.Second,
		lastPend:   make(map[string]struct{}),
		lastDone:   make(map[string]struct{}),
	}, nil
}

// Start begins watching for queue changes
func (qw *QueueWatcher) Start(ctx context.Context) error {
	go qw.pollLoop(ctx)
	return nil
}

// Stop stops the watcher
func (qw *QueueWatcher) Stop() {
	close(qw.stopCh)
}

func (qw *QueueWatcher) pollLoop(ctx context.Context) {
	ticker := time.NewTicker(qw.pollEvery)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-qw.stopCh:
			return
		case <-ticker.C:
			qw.pollOnce()
		}
	}
}

func (qw *QueueWatcher) pollOnce() {
	pending, err := qw.listFiles(qw.pendDir)
	if err != nil {
		log.Printf("queue watcher pending scan failed: %v", err)
		return
	}
	done, err := qw.listFiles(qw.doneDir)
	if err != nil {
		log.Printf("queue watcher done scan failed: %v", err)
		return
	}

	for name := range pending {
		if _, seen := qw.lastPend[name]; !seen {
			if qw.submitted != nil {
				qw.submitted.Inc()
			}
		}
	}
	for name := range qw.lastPend {
		if _, still := pending[name]; !still {
			if qw.processed != nil {
				qw.processed.Inc()
			}
		}
	}
	for name := range done {
		if _, seen := qw.lastDone[name]; !seen {
			if qw.completed != nil {
				qw.completed.Inc()
			}
		}
	}

	qw.queueDepth.Set(float64(len(pending)))

	qw.lastPend = pending
	qw.lastDone = done
}

func (qw *QueueWatcher) listFiles(dir string) (map[string]struct{}, error) {
	glob, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
	}
	set := make(map[string]struct{}, len(glob))
	for _, path := range glob {
		set[path] = struct{}{}
	}
	return set, nil
}

// countFiles counts files in a directory
func (qw *QueueWatcher) countFiles(dir string) (int, error) {
	entries, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return 0, err
	}

	return len(entries), nil
}

// GetQueueDepth returns the current queue depth
func (qw *QueueWatcher) GetQueueDepth() int {
	qw.mu.RLock()
	defer qw.mu.RUnlock()

	count, err := qw.countFiles(qw.pendDir)
	if err != nil {
		return 0
	}

	return count
}

// GetQueueStats returns queue statistics
func (qw *QueueWatcher) GetQueueStats() QueueStats {
	return QueueStats{
		Pending:   qw.GetQueueDepth(),
		Done:      qw.getDoneCount(),
		Submitted: qw.getSubmittedCount(),
		Processed: qw.getProcessedCount(),
		Completed: qw.getCompletedCount(),
	}
}

// getDoneCount counts files in done directory
func (qw *QueueWatcher) getDoneCount() int {
	count, err := qw.countFiles(qw.doneDir)
	if err != nil {
		return 0
	}
	return count
}

// getSubmittedCount gets the submitted counter value
func (qw *QueueWatcher) getSubmittedCount() float64 {
	if qw.submitted == nil {
		return 0
	}
	if v, ok := qw.registry.Value("queue_tasks_submitted_total"); ok {
		return v
	}
	return 0
}

// getProcessedCount gets the processed counter value
func (qw *QueueWatcher) getProcessedCount() float64 {
	if qw.processed == nil {
		return 0
	}
	if v, ok := qw.registry.Value("queue_tasks_processed_total"); ok {
		return v
	}
	return 0
}

// getCompletedCount gets the completed counter value
func (qw *QueueWatcher) getCompletedCount() float64 {
	if qw.completed == nil {
		return 0
	}
	if v, ok := qw.registry.Value("queue_tasks_completed_total"); ok {
		return v
	}
	return 0
}

// QueueStats represents queue statistics
type QueueStats struct {
	Pending   int     `json:"pending"`
	Done      int     `json:"done"`
	Submitted float64 `json:"submitted"`
	Processed float64 `json:"processed"`
	Completed float64 `json:"completed"`
}
