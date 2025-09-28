package events

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/dspy/orchestrator/pkg/telemetry"
)

// EventBus handles event publishing to various backends
type EventBus struct {
	registry    *telemetry.Registry
	kafkaEnabled bool
	reddbEnabled bool
	logFile     *os.File
}

// Event represents a system event
type Event struct {
	Type      string                 `json:"type"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
	Source    string                 `json:"source"`
}

// NewEventBus creates a new event bus
func NewEventBus(registry *telemetry.Registry, kafkaEnabled, reddbEnabled bool) (*EventBus, error) {
	// Ensure logs directory exists
	logsDir := "logs"
	if err := os.MkdirAll(logsDir, 0755); err != nil {
		return nil, err
	}

	// Open log file for agent actions
	logFile, err := os.OpenFile(
		filepath.Join(logsDir, "agent_action.jsonl"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND,
		0644,
	)
	if err != nil {
		return nil, err
	}

	return &EventBus{
		registry:     registry,
		kafkaEnabled: kafkaEnabled,
		reddbEnabled: reddbEnabled,
		logFile:     logFile,
	}, nil
}

// PublishTaskSubmitted publishes a task submitted event
func (eb *EventBus) PublishTaskSubmitted(ctx context.Context, taskID string, taskData map[string]interface{}) error {
	event := Event{
		Type:      "task_submitted",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"task_id":   taskID,
			"task_data": taskData,
		},
		Source: "orchestrator",
	}

	return eb.publishEvent(ctx, event)
}

// PublishTaskCompleted publishes a task completed event
func (eb *EventBus) PublishTaskCompleted(ctx context.Context, taskID string, result map[string]interface{}) error {
	event := Event{
		Type:      "task_completed",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"task_id": taskID,
			"result":  result,
		},
		Source: "orchestrator",
	}

	return eb.publishEvent(ctx, event)
}

// PublishTaskFailed publishes a task failed event
func (eb *EventBus) PublishTaskFailed(ctx context.Context, taskID string, error string) error {
	event := Event{
		Type:      "task_failed",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"task_id": taskID,
			"error":   error,
		},
		Source: "orchestrator",
	}

	return eb.publishEvent(ctx, event)
}

// PublishSlurmJobSubmitted publishes a Slurm job submitted event
func (eb *EventBus) PublishSlurmJobSubmitted(ctx context.Context, taskID, jobID string) error {
	event := Event{
		Type:      "slurm_job_submitted",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"task_id": taskID,
			"job_id":  jobID,
		},
		Source: "slurm_bridge",
	}

	return eb.publishEvent(ctx, event)
}

// PublishSlurmJobCompleted publishes a Slurm job completed event
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

// publishEvent publishes an event to all enabled backends
func (eb *EventBus) publishEvent(ctx context.Context, event Event) error {
	// Always log to file
	if err := eb.logToFile(event); err != nil {
		return fmt.Errorf("failed to log event to file: %w", err)
	}

	// Publish to Kafka if enabled
	if eb.kafkaEnabled {
		if err := eb.publishToKafka(ctx, event); err != nil {
			// Log error but don't fail the operation
			fmt.Printf("Failed to publish to Kafka: %v\n", err)
		}
	}

	// Publish to RedDB if enabled
	if eb.reddbEnabled {
		if err := eb.publishToRedDB(ctx, event); err != nil {
			// Log error but don't fail the operation
			fmt.Printf("Failed to publish to RedDB: %v\n", err)
		}
	}

	// Update metrics
	eb.registry.Counter("events_published_total").Inc()

	return nil
}

// logToFile logs the event to the agent action log file
func (eb *EventBus) logToFile(event Event) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}

	_, err = eb.logFile.Write(append(data, '\n'))
	return err
}

// publishToKafka publishes the event to Kafka
func (eb *EventBus) publishToKafka(ctx context.Context, event Event) error {
	// This would integrate with the existing Kafka infrastructure
	// For now, just log that we would publish to Kafka
	fmt.Printf("Would publish to Kafka: %s\n", event.Type)
	return nil
}

// publishToRedDB publishes the event to RedDB
func (eb *EventBus) publishToRedDB(ctx context.Context, event Event) error {
	// This would integrate with the existing RedDB infrastructure
	// For now, just log that we would publish to RedDB
	fmt.Printf("Would publish to RedDB: %s\n", event.Type)
	return nil
}

// Close closes the event bus
func (eb *EventBus) Close() error {
	if eb.logFile != nil {
		return eb.logFile.Close()
	}
	return nil
}
