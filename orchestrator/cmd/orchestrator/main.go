package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/dspy/orchestrator/internal/events"
	"github.com/dspy/orchestrator/internal/metrics"
	"github.com/dspy/orchestrator/internal/queue"
	"github.com/dspy/orchestrator/internal/slurm"
	"github.com/dspy/orchestrator/internal/workflow"
	"github.com/dspy/orchestrator/pkg/telemetry"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	registry := telemetry.NewRegistry()

	// Create event bus
	eventBus, err := events.NewEventBus(registry, true, true) // Enable Kafka and RedDB
	if err != nil {
		log.Fatalf("Failed to create event bus: %v", err)
	}
	defer eventBus.Close()

	// Create queue watcher
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
		log.Fatalf("Failed to create queue watcher: %v", err)
	}

	// Start queue watcher
	if err := queueWatcher.Start(ctx); err != nil {
		log.Fatalf("Failed to start queue watcher: %v", err)
	}

	// Create Slurm bridge
	slurmBridge := slurm.NewSlurmBridge(registry, queueDir, eventBus)
	if err := slurmBridge.Start(ctx); err != nil {
		log.Fatalf("Failed to start Slurm bridge: %v", err)
	}
	defer slurmBridge.Stop()

	// Create metrics
	queueGauge := telemetry.NewGauge(registry, "env_queue_depth", "Number of queued environment evaluations awaiting execution.")
	gpuWaitGauge := telemetry.NewGauge(registry, "gpu_wait_seconds", "Average seconds tasks waited for GPU resources.")
	errorGauge := telemetry.NewGauge(registry, "env_error_rate", "Rolling error rate observed across tasks.")

	source := &metrics.RegistrySource{
		Registry:    registry,
		QueueMetric: "env_queue_depth",
		GPUMetric:   "gpu_wait_seconds",
		ErrorMetric: "env_error_rate",
	}

	// Create orchestrator
	cfg := workflow.Config{
		BaseLimit:          4,
		MinLimit:           1,
		MaxLimit:           16,
		IncreaseStep:       2,
		DecreaseStep:       2,
		QueueHighWatermark: 50,
		GPUWaitHigh:        5,
		ErrorRateHigh:      0.3,
		AdaptationInterval: 5 * time.Second,
	}

	o, err := workflow.New(ctx, cfg, source, registry)
	if err != nil {
		log.Fatalf("create orchestrator: %v", err)
	}

	// HTTP mux: metrics + queue submit + Slurm endpoints
	mux := http.NewServeMux()
	mux.Handle("/metrics", registry.Handler())
	mux.HandleFunc("/queue/submit", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), 400)
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
		if payload == nil {
			payload = make(map[string]interface{})
			envelope["payload"] = payload
		}

		// Check if this is a Slurm job
		if class == "gpu_slurm" {
			// Submit to Slurm
			job, err := slurmBridge.SubmitGPUJob(r.Context(), id, payload)
			if err != nil {
				http.Error(w, fmt.Sprintf("Slurm submission failed: %v", err), 500)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"ok":           true,
				"slurm_job_id": job.ID,
				"status":       "submitted",
			})
			return
		}

		// Regular queue processing
		fpath := filepath.Join(pendDir, id+".json")
		serialized, err := json.Marshal(envelope)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		if err := os.WriteFile(fpath, serialized, 0o644); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}

		// Publish event
		if err := eventBus.PublishTaskSubmitted(ctx, id, map[string]interface{}{
			"class":   class,
			"payload": payload,
		}); err != nil {
			log.Printf("Failed to publish task submission event: %v", err)
		}

		// Update queue depth
		queueDepth := queueWatcher.GetQueueDepth()
		queueGauge.Set(float64(queueDepth))

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
	})

	// Slurm job status endpoint
	mux.HandleFunc("/slurm/status/", func(w http.ResponseWriter, r *http.Request) {
		taskID := r.URL.Path[len("/slurm/status/"):]
		job, exists := slurmBridge.GetJobStatus(taskID)
		if !exists {
			http.Error(w, "Job not found", 404)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(job)
	})

	// Queue status endpoint
	mux.HandleFunc("/queue/status", func(w http.ResponseWriter, r *http.Request) {
		stats := queueWatcher.GetQueueStats()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(stats)
	})

	server := &http.Server{Addr: ":9097", Handler: mux}

	go func() {
		log.Printf("orchestrator listening on %s", server.Addr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("server error: %v", err)
		}
	}()

	// Demo tasks (only if ORCHESTRATOR_DEMO is not set to 0)
	if os.Getenv("ORCHESTRATOR_DEMO") != "0" {
		demoTasks(o, queueGauge, gpuWaitGauge, errorGauge)
	}

	<-ctx.Done()
	log.Println("shutting down orchestrator")
	_ = server.Shutdown(context.Background())
	if err := o.Shutdown(); err != nil {
		log.Printf("shutdown error: %v", err)
	}
}

func demoTasks(o *workflow.Orchestrator, queue *telemetry.Gauge, gpu *telemetry.Gauge, errGauge *telemetry.Gauge) {
	// Simulate metrics oscillation in the background
	go func() {
		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()
		peak := false
		for range ticker.C {
			if peak {
				queue.Set(15)
				gpu.Set(1.2)
				errGauge.Set(0.05)
			} else {
				queue.Set(80)
				gpu.Set(3.5)
				errGauge.Set(0.1)
			}
			peak = !peak
		}
	}()

	for i := 0; i < 32; i++ {
		i := i
		o.Go("demo", func(ctx context.Context) error {
			sleep := time.Duration(500+20*i) * time.Millisecond
			select {
			case <-time.After(sleep):
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
	}
}
