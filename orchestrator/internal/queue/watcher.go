package queue

import (
	"context"
	"log"
	"path/filepath"
	"sync"
	"time"

	"github.com/dspy/orchestrator/pkg/telemetry"
	"github.com/fsnotify/fsnotify"
)

// QueueWatcher monitors queue directory changes using fsnotify
type QueueWatcher struct {
	watcher    *fsnotify.Watcher
	pendDir    string
	doneDir    string
	registry   *telemetry.Registry
	queueDepth *telemetry.Gauge
	mu         sync.RWMutex
	stopCh     chan struct{}
}

// NewQueueWatcher creates a new queue watcher
func NewQueueWatcher(pendDir, doneDir string, registry *telemetry.Registry) (*QueueWatcher, error) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}
	
	// Watch the pending directory
	if err := watcher.Add(pendDir); err != nil {
		watcher.Close()
		return nil, err
	}
	
	// Watch the done directory
	if err := watcher.Add(doneDir); err != nil {
		watcher.Close()
		return nil, err
	}
	
	return &QueueWatcher{
		watcher:    watcher,
		pendDir:    pendDir,
		doneDir:    doneDir,
		registry:   registry,
		queueDepth: telemetry.NewGauge(registry, "queue_depth", "Number of pending tasks in queue"),
		stopCh:     make(chan struct{}),
	}, nil
}

// Start begins watching for queue changes
func (qw *QueueWatcher) Start(ctx context.Context) error {
	go qw.watchLoop(ctx)
	go qw.updateMetricsLoop(ctx)
	return nil
}

// Stop stops the watcher
func (qw *QueueWatcher) Stop() {
	close(qw.stopCh)
	qw.watcher.Close()
}

// watchLoop monitors file system events
func (qw *QueueWatcher) watchLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-qw.stopCh:
			return
		case event := <-qw.watcher.Events:
			qw.handleEvent(event)
		case err := <-qw.watcher.Errors:
			if err != nil {
				log.Printf("Queue watcher error: %v", err)
			}
		}
	}
}

// handleEvent handles file system events
func (qw *QueueWatcher) handleEvent(event fsnotify.Event) {
	// Only process files, not directories
	if event.Op&fsnotify.Create == fsnotify.Create {
		if filepath.Dir(event.Name) == qw.pendDir {
			// New task added to pending queue
			qw.queueDepth.Inc()
			qw.registry.Counter("queue_tasks_submitted_total").Inc()
		}
	}
	
	if event.Op&fsnotify.Remove == fsnotify.Remove {
		if filepath.Dir(event.Name) == qw.pendDir {
			// Task removed from pending queue
			qw.queueDepth.Dec()
			qw.registry.Counter("queue_tasks_processed_total").Inc()
		}
	}
	
	if event.Op&fsnotify.Write == fsnotify.Write {
		if filepath.Dir(event.Name) == qw.doneDir {
			// Task completed
			qw.registry.Counter("queue_tasks_completed_total").Inc()
		}
	}
}

// updateMetricsLoop periodically updates queue metrics
func (qw *QueueWatcher) updateMetricsLoop(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-qw.stopCh:
			return
		case <-ticker.C:
			qw.updateQueueDepth()
		}
	}
}

// updateQueueDepth updates the queue depth metric
func (qw *QueueWatcher) updateQueueDepth() {
	// Count files in pending directory
	count, err := qw.countFiles(qw.pendDir)
	if err != nil {
		log.Printf("Failed to count pending files: %v", err)
		return
	}
	
	// Update gauge
	qw.queueDepth.Set(float64(count))
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
		Pending:  qw.GetQueueDepth(),
		Done:     qw.getDoneCount(),
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
	// This would need to be implemented in the telemetry package
	// For now, return 0
	return 0
}

// getProcessedCount gets the processed counter value
func (qw *QueueWatcher) getProcessedCount() float64 {
	// This would need to be implemented in the telemetry package
	// For now, return 0
	return 0
}

// getCompletedCount gets the completed counter value
func (qw *QueueWatcher) getCompletedCount() float64 {
	// This would need to be implemented in the telemetry package
	// For now, return 0
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
