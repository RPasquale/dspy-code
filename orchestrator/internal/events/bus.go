package events

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/dspy/orchestrator/pkg/telemetry"
)

// EventBus handles event publishing to various backends including RedDB and Kafka spools.
type EventBus struct {
	registry     *telemetry.Registry
	kafkaEnabled bool
	reddbEnabled bool
	logFile      *os.File
	kafkaTopic   string
	vectorTopic  string
	kafkaWriter  *spoolWriter
	vectorWriter *spoolWriter
	reddbClient  *http.Client
	reddbURL     string
	subMux       sync.RWMutex
	subscribers  map[int64]*eventSubscriber
	nextSubID    int64
}

// Event represents a system event.
type Event struct {
	Type      string                 `json:"type"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
	Source    string                 `json:"source"`
}

type eventSubscriber struct {
	ch      chan Event
	filters map[string]struct{}
}

// NewEventBus creates a new event bus instance.
func NewEventBus(registry *telemetry.Registry, kafkaEnabled, reddbEnabled bool) (*EventBus, error) {
	logsDir := "logs"
	if err := os.MkdirAll(logsDir, 0o755); err != nil {
		return nil, err
	}

	logFile, err := os.OpenFile(filepath.Join(logsDir, "agent_action.jsonl"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}

	eb := &EventBus{
		registry:     registry,
		kafkaEnabled: kafkaEnabled,
		reddbEnabled: reddbEnabled,
		logFile:      logFile,
	}

	if kafkaEnabled {
		spoolDir := os.Getenv("KAFKA_SPOOL_DIR")
		if spoolDir == "" {
			spoolDir = filepath.Join("logs", "kafka")
		}
		if err := os.MkdirAll(spoolDir, 0o755); err != nil {
			return nil, fmt.Errorf("create kafka spool dir: %w", err)
		}
		eb.kafkaTopic = envOrDefault("KAFKA_EVENT_TOPIC", "agent.events")
		eb.vectorTopic = envOrDefault("KAFKA_VECTOR_TOPIC", "agent.vectorized")
		eb.kafkaWriter = newSpoolWriter(filepath.Join(spoolDir, eb.kafkaTopic+".jsonl"))
		eb.vectorWriter = newSpoolWriter(filepath.Join(spoolDir, eb.vectorTopic+".jsonl"))
	}

	if reddbEnabled {
		eb.reddbURL = strings.TrimRight(os.Getenv("REDDB_URL"), "/")
		if eb.reddbURL != "" {
			eb.reddbClient = &http.Client{Timeout: 10 * time.Second}
		}
	}

	return eb, nil
}

// Subscribe registers a subscriber for the given event types. If types is empty,
// all events are delivered. The returned cancel function must be invoked to
// release resources.
func (eb *EventBus) Subscribe(types []string) (<-chan Event, func()) {
	eb.subMux.Lock()
	defer eb.subMux.Unlock()
	if eb.subscribers == nil {
		eb.subscribers = make(map[int64]*eventSubscriber)
	}

	filters := make(map[string]struct{})
	for _, t := range types {
		if t == "" {
			continue
		}
		filters[t] = struct{}{}
	}

	id := eb.nextSubID
	eb.nextSubID++

	ch := make(chan Event, 64)
	eb.subscribers[id] = &eventSubscriber{ch: ch, filters: filters}

	cancel := func() {
		eb.subMux.Lock()
		if sub, ok := eb.subscribers[id]; ok {
			delete(eb.subscribers, id)
			close(sub.ch)
		}
		eb.subMux.Unlock()
	}

	return ch, cancel
}

// PublishTaskSubmitted publishes a task submitted event.
func (eb *EventBus) PublishTaskSubmitted(ctx context.Context, taskID string, taskData map[string]interface{}) error {
	data := map[string]interface{}{
		"task_id": taskID,
	}
	for k, v := range taskData {
		data[k] = v
	}
	return eb.publishEvent(ctx, Event{
		Type:      "task_submitted",
		Timestamp: time.Now(),
		Data:      data,
		Source:    "orchestrator",
	})
}

// PublishTaskCompleted publishes task completion events.
func (eb *EventBus) PublishTaskCompleted(ctx context.Context, taskID string, result map[string]interface{}) error {
	return eb.publishEvent(ctx, Event{
		Type:      "task_completed",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"task_id": taskID,
			"result":  result,
		},
		Source: "orchestrator",
	})
}

// PublishTaskFailed publishes failure events.
func (eb *EventBus) PublishTaskFailed(ctx context.Context, taskID string, failure string) error {
	return eb.publishEvent(ctx, Event{
		Type:      "task_failed",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"task_id": taskID,
			"error":   failure,
		},
		Source: "orchestrator",
	})
}

// PublishTaskStarted emits a task_started event when execution begins.
func (eb *EventBus) PublishTaskStarted(ctx context.Context, taskID string, payload map[string]interface{}) error {
	data := map[string]interface{}{
		"task_id": taskID,
	}
	for k, v := range payload {
		data[k] = v
	}
	return eb.publishEvent(ctx, Event{
		Type:      "task_started",
		Timestamp: time.Now(),
		Data:      data,
		Source:    "runner",
	})
}

// PublishSlurmJobSubmitted emits slurm submission events.
func (eb *EventBus) PublishSlurmJobSubmitted(ctx context.Context, taskID, jobID string) error {
	return eb.publishEvent(ctx, Event{
		Type:      "slurm_job_submitted",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"task_id": taskID,
			"job_id":  jobID,
		},
		Source: "slurm_bridge",
	})
}

// PublishSlurmJobCompleted emits slurm completion events.
func (eb *EventBus) PublishSlurmJobCompleted(ctx context.Context, taskID, jobID string, result map[string]interface{}) error {
	event := Event{
		Type:      "slurm_job_completed",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"task_id": taskID,
			"job_id":  jobID,
			"result":  result,
		},
		Source: "slurm_bridge",
	}
	return eb.publishEvent(ctx, event)
}

// PublishVectorizedMessage writes vectorized payloads for Spark/Kafka workers.
func (eb *EventBus) PublishVectorizedMessage(ctx context.Context, key string, payload any) error {
	if eb.vectorWriter == nil {
		return fmt.Errorf("vector writer not configured")
	}
	record := map[string]interface{}{
		"topic":     eb.vectorTopic,
		"key":       key,
		"value":     payload,
		"timestamp": time.Now().UnixMilli(),
	}
	return eb.vectorWriter.Append(record)
}

func (eb *EventBus) publishEvent(ctx context.Context, event Event) error {
	ensureResourceID(&event)
	if err := eb.logToFile(event); err != nil {
		return fmt.Errorf("log event: %w", err)
	}
	if eb.kafkaWriter != nil {
		record := map[string]interface{}{
			"topic":     eb.kafkaTopic,
			"key":       event.Type,
			"value":     event,
			"timestamp": time.Now().UnixMilli(),
		}
		if err := eb.kafkaWriter.Append(record); err != nil {
			fmt.Printf("failed to append kafka spool: %v\n", err)
		}
	}
	if eb.reddbClient != nil {
		if err := eb.publishToRedDB(ctx, event); err != nil {
			fmt.Printf("failed to publish to RedDB: %v\n", err)
		}
	}
	eb.registry.Counter("events_published_total").Inc()
	eb.broadcast(event)
	return nil
}

func ensureResourceID(event *Event) {
	if event.Data == nil {
		return
	}
	if _, ok := event.Data["resource_id"]; ok {
		return
	}
	for _, key := range []string{"task_id", "run_id", "workflow_id", "job_id"} {
		if val, ok := event.Data[key]; ok {
			if v := fmt.Sprint(val); v != "" {
				event.Data["resource_id"] = v
				return
			}
		}
	}
}

func (eb *EventBus) broadcast(event Event) {
	eb.subMux.RLock()
	defer eb.subMux.RUnlock()
	for _, sub := range eb.subscribers {
		if len(sub.filters) > 0 {
			if _, ok := sub.filters[event.Type]; !ok {
				continue
			}
		}
		select {
		case sub.ch <- event:
		default:
		}
	}
}

func (eb *EventBus) logToFile(event Event) error {
	payload, err := json.Marshal(event)
	if err != nil {
		return err
	}
	_, err = eb.logFile.Write(append(payload, '\n'))
	return err
}

func (eb *EventBus) publishToRedDB(ctx context.Context, event Event) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, eb.reddbURL+"/api/v1/events", bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := eb.reddbClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("reddb status %d", resp.StatusCode)
	}
	return nil
}

// Close releases event bus resources.
func (eb *EventBus) Close() error {
	if eb.logFile != nil {
		if err := eb.logFile.Close(); err != nil {
			return err
		}
	}
	eb.subMux.Lock()
	for id, sub := range eb.subscribers {
		close(sub.ch)
		delete(eb.subscribers, id)
	}
	eb.subMux.Unlock()
	return nil
}

// spoolWriter persists JSON records for downstream ingestion.
type spoolWriter struct {
	mu   sync.Mutex
	path string
}

func newSpoolWriter(path string) *spoolWriter {
	return &spoolWriter{path: path}
}

func (w *spoolWriter) Append(record map[string]interface{}) error {
	payload, err := json.Marshal(record)
	if err != nil {
		return err
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := os.MkdirAll(filepath.Dir(w.path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(w.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(payload, '\n'))
	return err
}

func envOrDefault(key, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return fallback
}
