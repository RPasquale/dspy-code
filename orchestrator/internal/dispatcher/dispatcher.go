package dispatcher

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/dspy/orchestrator/internal/events"
	"github.com/dspy/orchestrator/internal/queue"
	"github.com/dspy/orchestrator/internal/runner"
	"github.com/dspy/orchestrator/internal/workflow"
	"github.com/dspy/orchestrator/internal/workflowspec"
	"github.com/dspy/orchestrator/pkg/telemetry"
)

// HardwareProvider exposes runner hardware snapshots suitable for payload enrichment.
type HardwareProvider func() map[string]any

// TaskDispatcher coordinates task execution across the file queue, workflow orchestrator, and event bus.
type TaskDispatcher struct {
	orchestrator  *workflow.Orchestrator
	runnerClient  *runner.HTTPClient
	eventBus      *events.EventBus
	queueWatcher  *queue.QueueWatcher
	queueGauge    *telemetry.Gauge
	hardware      HardwareProvider
	workflowStore workflowspec.Store
	pendingDir    string
	doneDir       string
}

// NewTaskDispatcher constructs a dispatcher with the dependencies required to execute tasks.
func NewTaskDispatcher(
	orchestrator *workflow.Orchestrator,
	runnerClient *runner.HTTPClient,
	eventBus *events.EventBus,
	queueWatcher *queue.QueueWatcher,
	queueGauge *telemetry.Gauge,
	hardware HardwareProvider,
	workflowStore workflowspec.Store,
	pendingDir string,
	doneDir string,
) *TaskDispatcher {
	return &TaskDispatcher{
		orchestrator:  orchestrator,
		runnerClient:  runnerClient,
		eventBus:      eventBus,
		queueWatcher:  queueWatcher,
		queueGauge:    queueGauge,
		hardware:      hardware,
		workflowStore: workflowStore,
		pendingDir:    pendingDir,
		doneDir:       doneDir,
	}
}

// Dispatch schedules the provided task envelope for execution.
func (d *TaskDispatcher) Dispatch(parent context.Context, envelope map[string]interface{}) {
	if d.orchestrator == nil {
		return
	}
	id, _ := envelope["id"].(string)
	class, _ := envelope["class"].(string)
	rawPayload, _ := envelope["payload"].(map[string]interface{})

	payloadCopy := make(map[string]interface{})
	for k, v := range rawPayload {
		payloadCopy[k] = v
	}
	attachWorkflowContext(payloadCopy, d.workflowStore)
	d.attachRunnerHardwarePayload(payloadCopy)

	d.orchestrator.Go("env_task_"+id, func(ctx context.Context) error {
		sourcePath := filepath.Join(d.pendingDir, id+".json")
		lockPath := sourcePath + ".lock"

		if err := os.Rename(sourcePath, lockPath); err != nil {
			if !os.IsNotExist(err) {
				return err
			}
			if _, statErr := os.Stat(lockPath); statErr != nil {
				return fmt.Errorf("task file missing for %s", id)
			}
		}

		start := time.Now()
		if d.eventBus != nil {
			started := map[string]interface{}{
				"class":      class,
				"started_at": start.Format(time.RFC3339Nano),
			}
			if tenant, ok := payloadCopy["tenant"]; ok {
				started["tenant"] = tenant
			}
			if wf, ok := payloadCopy["workflow_id"]; ok {
				started["workflow_id"] = wf
			}
			if err := d.eventBus.PublishTaskStarted(ctx, id, started); err != nil {
				log.Printf("publish started event: %v", err)
			}
		}
		req := runner.TaskRequest{ID: id, Class: class, Payload: payloadCopy}
		resp, err := d.runnerClient.ExecuteTask(ctx, req)
		duration := time.Since(start)

		if err != nil {
			_ = os.Rename(lockPath, sourcePath)
			if publishErr := d.eventBus.PublishTaskFailed(ctx, id, err.Error()); publishErr != nil {
				log.Printf("publish failure event: %v", publishErr)
			}
			RefreshQueueGauge(d.queueWatcher, d.queueGauge)
			return err
		}

		doneRecord := map[string]interface{}{
			"id":            id,
			"class":         class,
			"payload":       payloadCopy,
			"runner_result": resp,
			"completed_at":  time.Now().Format(time.RFC3339Nano),
			"duration_ms":   duration.Seconds() * 1000,
		}
		doneBytes, err := json.MarshalIndent(doneRecord, "", "  ")
		if err != nil {
			log.Printf("marshal done record: %v", err)
		} else {
			_ = os.MkdirAll(d.doneDir, 0o755)
			if writeErr := os.WriteFile(filepath.Join(d.doneDir, id+".json"), doneBytes, 0o644); writeErr != nil {
				log.Printf("write done file: %v", writeErr)
			}
		}

		_ = os.Remove(lockPath)
		RefreshQueueGauge(d.queueWatcher, d.queueGauge)

		completionPayload := map[string]interface{}{
			"latency_ms": resp.LatencyMs,
			"class":      class,
		}
		if wfCtx, ok := payloadCopy["workflow_context"]; ok {
			completionPayload["workflow_context"] = wfCtx
		}
		if wfID, ok := payloadCopy["workflow_id"]; ok {
			completionPayload["workflow_id"] = wfID
		}
		if resp.Metadata != nil {
			completionPayload["metadata"] = resp.Metadata
		}
		if publishErr := d.eventBus.PublishTaskCompleted(ctx, id, completionPayload); publishErr != nil {
			log.Printf("publish completion event: %v", publishErr)
		}

		return nil
	})
}

// RefreshQueueGauge sets the queue depth gauge using the watcher.
func (d *TaskDispatcher) RefreshQueueGauge() {
	RefreshQueueGauge(d.queueWatcher, d.queueGauge)
}

// RefreshQueueGauge is a helper that updates the provided gauge based on the queue watcher.
func RefreshQueueGauge(queueWatcher *queue.QueueWatcher, gauge *telemetry.Gauge) {
	if queueWatcher == nil || gauge == nil {
		return
	}
	gauge.Set(float64(queueWatcher.GetQueueDepth()))
}

func (d *TaskDispatcher) attachRunnerHardwarePayload(payload map[string]interface{}) {
	if payload == nil {
		return
	}
	if _, exists := payload["runner_hardware"]; exists {
		return
	}
	if d.hardware == nil {
		return
	}
	if snapshot := d.hardware(); snapshot != nil {
		payload["runner_hardware"] = snapshot
	}
}

func attachWorkflowContext(payload map[string]interface{}, store workflowspec.Store) {
	if store == nil || payload == nil {
		return
	}
	idRaw, ok := payload["workflow_id"]
	if !ok {
		return
	}
	wfID, ok := idRaw.(string)
	if !ok || strings.TrimSpace(wfID) == "" {
		return
	}
	wf, err := store.Get(wfID)
	if err != nil {
		log.Printf("workflow lookup failed for %s: %v", wfID, err)
		return
	}
	ctxPayload := workflow.BuildWorkflowContext(wf)
	if ctxPayload == nil {
		return
	}
	payload["workflow_context"] = ctxPayload
	if _, exists := payload["tenant"]; !exists {
		if tenant, ok := ctxPayload["tenant"].(string); ok && tenant != "" {
			payload["tenant"] = tenant
		}
	}
}
