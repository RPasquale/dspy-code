package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/dspy/orchestrator/internal/events"
	"github.com/dspy/orchestrator/internal/metrics"
	"github.com/dspy/orchestrator/internal/queue"
	"github.com/dspy/orchestrator/internal/runner"
	"github.com/dspy/orchestrator/internal/slurm"
	"github.com/dspy/orchestrator/internal/workflow"
	"github.com/dspy/orchestrator/internal/workflowspec"
	"github.com/dspy/orchestrator/pkg/telemetry"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	registry := telemetry.NewRegistry()

	eventBus, err := events.NewEventBus(registry, true, true)
	if err != nil {
		log.Fatalf("failed to create event bus: %v", err)
	}
	defer eventBus.Close()

	workflowDir := os.Getenv("WORKFLOW_STORE_DIR")
	if workflowDir == "" {
		workflowDir = "data/workflows"
	}
	workflowStore, err := workflowspec.NewFileStore(workflowDir)
	if err != nil {
		log.Fatalf("failed to create workflow store: %v", err)
	}
	workflowGauge := telemetry.NewGauge(registry, "workflows_total", "Number of registered workflows.")
	refreshWorkflowGauge(workflowStore, workflowGauge)

	runDir := os.Getenv("WORKFLOW_RUN_DIR")
	if runDir == "" {
		runDir = "data/workflow_runs"
	}
	runStore, err := workflow.NewRunStore(runDir)
	if err != nil {
		log.Fatalf("failed to create workflow run store: %v", err)
	}

	queueDir := os.Getenv("ENV_QUEUE_DIR")
	if queueDir == "" {
		queueDir = "logs/env_queue"
	}
	pendDir := filepath.Join(queueDir, "pending")
	doneDir := filepath.Join(queueDir, "done")
	_ = os.MkdirAll(pendDir, 0o755)
	_ = os.MkdirAll(doneDir, 0o755)

	queueWatcher, err := queue.NewQueueWatcher(pendDir, doneDir, registry)
	if err != nil {
		log.Fatalf("failed to create queue watcher: %v", err)
	}
	if err := queueWatcher.Start(ctx); err != nil {
		log.Fatalf("failed to start queue watcher: %v", err)
	}

	slurmBridge := slurm.NewSlurmBridge(registry, queueDir, eventBus)
	if err := slurmBridge.Start(ctx); err != nil {
		log.Fatalf("failed to start Slurm bridge: %v", err)
	}
	defer slurmBridge.Stop()

	queueGauge := telemetry.NewGauge(registry, "env_queue_depth", "Number of queued environment evaluations awaiting execution.")
	gpuWaitGauge := telemetry.NewGauge(registry, "gpu_wait_seconds", "Average seconds tasks waited for GPU resources.")
	errorGauge := telemetry.NewGauge(registry, "env_error_rate", "Rolling error rate observed across tasks.")
	runnerGpuGauge := telemetry.NewGauge(registry, "runner_gpu_total", "Detected GPU devices on the environment runner.")

	source := &metrics.RegistrySource{
		Registry:    registry,
		QueueMetric: "env_queue_depth",
		GPUMetric:   "gpu_wait_seconds",
		ErrorMetric: "env_error_rate",
	}

	cfg := workflow.Config{
		BaseLimit:          4,
		MinLimit:           1,
		MaxLimit:           32,
		IncreaseStep:       2,
		DecreaseStep:       2,
		QueueHighWatermark: 40,
		GPUWaitHigh:        2,
		ErrorRateHigh:      0.2,
		AdaptationInterval: 3 * time.Second,
	}

	orchestrator, err := workflow.New(ctx, cfg, source, registry)
	if err != nil {
		log.Fatalf("create orchestrator: %v", err)
	}

	runnerClient := runner.NewHTTPClient(os.Getenv("ENV_RUNNER_URL"))
	runnerHardware := newRunnerHardwareCache()
	go pollRunnerMetrics(ctx, runnerClient, queueWatcher, runnerHardware, queueGauge, gpuWaitGauge, errorGauge, runnerGpuGauge)

	executor := workflow.NewExecutor(orchestrator, runStore, workflowStore, runnerClient, slurmBridge, eventBus, workflow.WithHardwareProvider(runnerHardware.SnapshotAsMap))

	mux := http.NewServeMux()
	mux.Handle("/metrics", registry.Handler())
	mux.HandleFunc("/workflows", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			workflows, err := workflowStore.List()
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			respondJSON(w, map[string]interface{}{"items": workflows})
		case http.MethodPost:
			defer r.Body.Close()
			var wf workflowspec.Workflow
			if err := json.NewDecoder(r.Body).Decode(&wf); err != nil {
				http.Error(w, fmt.Sprintf("invalid workflow payload: %v", err), http.StatusBadRequest)
				return
			}
			saved, err := workflowStore.Save(&wf)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			w.WriteHeader(http.StatusCreated)
			respondJSON(w, saved)
			refreshWorkflowGauge(workflowStore, workflowGauge)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/workflows/", func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/workflows/")
		if path == "" {
			http.Error(w, "workflow id is required", http.StatusBadRequest)
			return
		}
		parts := strings.Split(path, "/")
		id := parts[0]
		subresource := ""
		if len(parts) > 1 {
			subresource = strings.Join(parts[1:], "/")
		}
		switch {
		case subresource == "runs":
			switch r.Method {
			case http.MethodPost:
				defer r.Body.Close()
				var req struct {
					IdempotencyKey string                 `json:"idempotency_key"`
					Input          map[string]interface{} `json:"input"`
				}
				if r.Body != nil && r.Body != http.NoBody {
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err != io.EOF {
						http.Error(w, fmt.Sprintf("invalid workflow run payload: %v", err), http.StatusBadRequest)
						return
					}
				}
				runRecord, err := executor.StartRun(r.Context(), id, req.IdempotencyKey, req.Input)
				if err != nil {
					http.Error(w, fmt.Sprintf("start run failed: %v", err), http.StatusInternalServerError)
					return
				}
				respondJSON(w, runRecord)
			case http.MethodGet:
				limit := 0
				if raw := r.URL.Query().Get("limit"); raw != "" {
					if n, err := strconv.Atoi(raw); err == nil {
						limit = n
					}
				}
				runs, err := runStore.ListRuns(id, limit)
				if err != nil {
					http.Error(w, fmt.Sprintf("list runs failed: %v", err), http.StatusInternalServerError)
					return
				}
				respondJSON(w, map[string]interface{}{"items": runs})
			default:
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			}
			return
		case subresource == "history" && r.Method == http.MethodGet:
			revisions, err := workflowStore.History(id, 0)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			respondJSON(w, map[string]interface{}{"items": revisions})
			return
		case subresource != "" && subresource != "history":
			http.NotFound(w, r)
			return
		}
		switch r.Method {
		case http.MethodGet:
			wf, err := workflowStore.Get(id)
			if err != nil {
				http.Error(w, err.Error(), http.StatusNotFound)
				return
			}
			respondJSON(w, wf)
		case http.MethodPut:
			existing, err := workflowStore.Get(id)
			if err != nil {
				http.Error(w, err.Error(), http.StatusNotFound)
				return
			}
			defer r.Body.Close()
			var wf workflowspec.Workflow
			if err := json.NewDecoder(r.Body).Decode(&wf); err != nil {
				http.Error(w, fmt.Sprintf("invalid workflow payload: %v", err), http.StatusBadRequest)
				return
			}
			wf.ID = id
			if wf.CreatedAt.IsZero() {
				wf.CreatedAt = existing.CreatedAt
			}
			saved, err := workflowStore.Save(&wf)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			respondJSON(w, saved)
			refreshWorkflowGauge(workflowStore, workflowGauge)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})
	mux.HandleFunc("/workflow-runs/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/workflow-runs/")
		if id == "" {
			http.Error(w, "run id is required", http.StatusBadRequest)
			return
		}
		if strings.Contains(id, "/") {
			http.NotFound(w, r)
			return
		}
		switch r.Method {
		case http.MethodGet:
			runRecord, err := runStore.GetRun(id)
			if err != nil {
				http.Error(w, err.Error(), http.StatusNotFound)
				return
			}
			respondJSON(w, runRecord)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/queue/submit", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var envelope map[string]interface{}
		if err := json.Unmarshal(body, &envelope); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}

		id, _ := envelope["id"].(string)
		if id == "" {
			id = time.Now().Format("20060102T150405.000000000")
			envelope["id"] = id
		}
		class, _ := envelope["class"].(string)
		if class == "" {
			class = "cpu_short"
			envelope["class"] = class
		}
		payload, _ := envelope["payload"].(map[string]interface{})
		payloadCopy := make(map[string]interface{})
		if payload != nil {
			for k, v := range payload {
				payloadCopy[k] = v
			}
		} else {
			payloadCopy = make(map[string]interface{})
		}
		envelope["payload"] = payloadCopy

		if class == "gpu_slurm" {
			job, err := slurmBridge.SubmitGPUJob(r.Context(), id, payloadCopy)
			if err != nil {
				http.Error(w, fmt.Sprintf("Slurm submission failed: %v", err), http.StatusInternalServerError)
				return
			}
			respondJSON(w, map[string]interface{}{"ok": true, "slurm_job_id": job.ID, "status": "submitted"})
			return
		}

		fpath := filepath.Join(pendDir, id+".json")
		serialized, err := json.Marshal(envelope)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := os.WriteFile(fpath, serialized, 0o644); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		if err := eventBus.PublishTaskSubmitted(ctx, id, map[string]interface{}{"class": class, "payload": payloadCopy}); err != nil {
			log.Printf("failed to publish submission event: %v", err)
		}

		refreshQueueGauge(queueWatcher, queueGauge)
		scheduleEnvTask(ctx, orchestrator, runnerClient, eventBus, queueWatcher, queueGauge, runnerHardware, workflowStore, pendDir, doneDir, envelope)

		respondJSON(w, map[string]interface{}{"ok": true, "id": id})
	})

	mux.HandleFunc("/slurm/status/", func(w http.ResponseWriter, r *http.Request) {
		taskID := strings.TrimPrefix(r.URL.Path, "/slurm/status/")
		job, exists := slurmBridge.GetJobStatus(taskID)
		if !exists {
			http.Error(w, "Job not found", http.StatusNotFound)
			return
		}
		respondJSON(w, job)
	})

	mux.HandleFunc("/queue/status", func(w http.ResponseWriter, r *http.Request) {
		respondJSON(w, queueWatcher.GetQueueStats())
	})

	server := &http.Server{Addr: ":9097", Handler: mux}
	go func() {
		log.Printf("orchestrator listening on %s", server.Addr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("server error: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("shutting down orchestrator")
	_ = server.Shutdown(context.Background())
	if err := orchestrator.Shutdown(); err != nil {
		log.Printf("shutdown error: %v", err)
	}
}

func scheduleEnvTask(parent context.Context, orchestrator *workflow.Orchestrator, runnerClient *runner.HTTPClient, eventBus *events.EventBus, queueWatcher *queue.QueueWatcher, queueGauge *telemetry.Gauge, hardwareCache *runnerHardwareCache, workflowStore workflowspec.Store, pendDir, doneDir string, envelope map[string]interface{}) {
	id, _ := envelope["id"].(string)
	class, _ := envelope["class"].(string)
	rawPayload, _ := envelope["payload"].(map[string]interface{})

	payloadCopy := make(map[string]interface{})
	for k, v := range rawPayload {
		payloadCopy[k] = v
	}
	attachWorkflowContext(payloadCopy, workflowStore)
	attachRunnerHardwarePayload(payloadCopy, hardwareCache)

	orchestrator.Go("env_task_"+id, func(ctx context.Context) error {
		sourcePath := filepath.Join(pendDir, id+".json")
		lockPath := sourcePath + ".lock"

		// Acquire exclusive ownership of the file for execution
		if err := os.Rename(sourcePath, lockPath); err != nil {
			if !os.IsNotExist(err) {
				return err
			}
			// If the lock already exists we reuse it
			if _, statErr := os.Stat(lockPath); statErr != nil {
				return fmt.Errorf("task file missing for %s", id)
			}
		}

		start := time.Now()
		req := runner.TaskRequest{ID: id, Class: class, Payload: payloadCopy}
		resp, err := runnerClient.ExecuteTask(ctx, req)
		duration := time.Since(start)

		if err != nil {
			_ = os.Rename(lockPath, sourcePath)
			if publishErr := eventBus.PublishTaskFailed(ctx, id, err.Error()); publishErr != nil {
				log.Printf("publish failure event: %v", publishErr)
			}
			refreshQueueGauge(queueWatcher, queueGauge)
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
			_ = os.MkdirAll(doneDir, 0o755)
			if writeErr := os.WriteFile(filepath.Join(doneDir, id+".json"), doneBytes, 0o644); writeErr != nil {
				log.Printf("write done file: %v", writeErr)
			}
		}

		_ = os.Remove(lockPath)
		refreshQueueGauge(queueWatcher, queueGauge)

		completionPayload := map[string]interface{}{"latency_ms": resp.LatencyMs, "class": class}
		if wfCtx, ok := payloadCopy["workflow_context"]; ok {
			completionPayload["workflow_context"] = wfCtx
		}
		if wfID, ok := payloadCopy["workflow_id"]; ok {
			completionPayload["workflow_id"] = wfID
		}
		if publishErr := eventBus.PublishTaskCompleted(ctx, id, completionPayload); publishErr != nil {
			log.Printf("publish completion event: %v", publishErr)
		}

		inputs := extractInputs(payloadCopy)
		if len(inputs) == len(resp.Embeddings) && len(inputs) > 0 {
			for idx, input := range inputs {
				tokens := tokenize(input)
				vectorPayload := map[string]interface{}{
					"task_id":   id,
					"index":     idx,
					"input":     input,
					"tokens":    tokens,
					"embedding": resp.Embeddings[idx],
					"timestamp": time.Now().Unix(),
				}
				if err := eventBus.PublishVectorizedMessage(ctx, id, vectorPayload); err != nil {
					if wfCtx, ok := payloadCopy["workflow_context"]; ok {
						vectorPayload["workflow_context"] = wfCtx
					}
					if wfID, ok := payloadCopy["workflow_id"].(string); ok && wfID != "" {
						vectorPayload["workflow_id"] = wfID
					}
					log.Printf("publish vectorized message: %v", err)
				}
			}
		}

		return nil
	})
}

func pollRunnerMetrics(ctx context.Context, client *runner.HTTPClient, queueWatcher *queue.QueueWatcher, hardware *runnerHardwareCache, queueGauge, gpuWaitGauge, errorGauge, gpuTotalGauge *telemetry.Gauge) {
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			metrics, err := client.FetchMetrics(ctx)
			if err != nil {
				continue
			}
			queueGauge.Set(float64(metrics.QueueDepth))
			gpuWaitGauge.Set(float64(metrics.LatencyP95Ms) / 1000.0)
			errorRate := 0.0
			if metrics.TasksProcessed > 0 {
				errorRate = float64(metrics.TotalErrors) / float64(metrics.TasksProcessed)
			}
			errorGauge.Set(math.Min(errorRate, 1.0))
			if metrics.Hardware != nil {
				hardware.Set(metrics.Hardware)
				gpuTotal := 0.0
				for _, accel := range metrics.Hardware.Accelerators {
					gpuTotal += float64(accel.Count)
				}
				gpuTotalGauge.Set(gpuTotal)
			}

			// Fallback to watcher for instantaneous updates
			refreshQueueGauge(queueWatcher, queueGauge)
		}
	}
}

func refreshQueueGauge(queueWatcher *queue.QueueWatcher, gauge *telemetry.Gauge) {
	if queueWatcher == nil || gauge == nil {
		return
	}
	gauge.Set(float64(queueWatcher.GetQueueDepth()))
}

type runnerHardwareCache struct {
	mu          sync.RWMutex
	snapshot    *runner.HardwareSnapshot
	snapshotMap map[string]any
}

func newRunnerHardwareCache() *runnerHardwareCache {
	return &runnerHardwareCache{}
}

func (c *runnerHardwareCache) Set(hw *runner.HardwareSnapshot) {
	if hw == nil {
		return
	}
	clone := *hw
	clone.Accelerators = append([]runner.Accelerator(nil), hw.Accelerators...)
	var snapshotMap map[string]any
	if payload, err := json.Marshal(hw); err == nil {
		_ = json.Unmarshal(payload, &snapshotMap)
	}
	c.mu.Lock()
	c.snapshot = &clone
	c.snapshotMap = snapshotMap
	c.mu.Unlock()
}

func (c *runnerHardwareCache) Snapshot() *runner.HardwareSnapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.snapshot == nil {
		return nil
	}
	clone := *c.snapshot
	clone.Accelerators = append([]runner.Accelerator(nil), c.snapshot.Accelerators...)
	return &clone
}

func (c *runnerHardwareCache) SnapshotAsMap() map[string]any {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.snapshotMap == nil {
		return nil
	}
	clone := make(map[string]any, len(c.snapshotMap))
	for k, v := range c.snapshotMap {
		clone[k] = v
	}
	return clone
}

func tokenize(input string) []string {
	lower := strings.ToLower(input)
	fields := strings.Fields(lower)
	tokens := make([]string, 0, len(fields))
	for _, field := range fields {
		trimmed := strings.Trim(field, ".,;:!?")
		if trimmed != "" {
			tokens = append(tokens, trimmed)
		}
	}
	return tokens
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

func attachRunnerHardwarePayload(payload map[string]interface{}, cache *runnerHardwareCache) {
	if cache == nil || payload == nil {
		return
	}
	if _, exists := payload["runner_hardware"]; exists {
		return
	}
	if snapshot := cache.SnapshotAsMap(); snapshot != nil {
		payload["runner_hardware"] = snapshot
	}
}

func extractInputs(payload map[string]interface{}) []string {
	inputsVal, ok := payload["inputs"]
	if !ok {
		return nil
	}
	arr, ok := inputsVal.([]interface{})
	if !ok {
		return nil
	}
	out := make([]string, 0, len(arr))
	for _, item := range arr {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func refreshWorkflowGauge(store workflowspec.Store, gauge *telemetry.Gauge) {
	if store == nil || gauge == nil {
		return
	}
	workflows, err := store.List()
	if err != nil {
		log.Printf("workflow gauge refresh failed: %v", err)
		return
	}
	gauge.Set(float64(len(workflows)))
}

func respondJSON(w http.ResponseWriter, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		log.Printf("respond json: %v", err)
	}
}
