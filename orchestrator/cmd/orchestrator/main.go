package main

import (
    "context"
    "encoding/json"
    "io"
    "log"
    "net/http"
    "os"
    "path/filepath"
    "os/signal"
    "syscall"
    "time"

    "github.com/dspy/orchestrator/internal/metrics"
    "github.com/dspy/orchestrator/internal/workflow"
    "github.com/dspy/orchestrator/pkg/telemetry"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	registry := telemetry.NewRegistry()

	queueGauge := telemetry.NewGauge(registry, "env_queue_depth", "Number of queued environment evaluations awaiting execution.")
	gpuWaitGauge := telemetry.NewGauge(registry, "gpu_wait_seconds", "Average seconds tasks waited for GPU resources.")
	errorGauge := telemetry.NewGauge(registry, "env_error_rate", "Rolling error rate observed across tasks.")

	source := &metrics.RegistrySource{
		Registry:    registry,
		QueueMetric: "env_queue_depth",
		GPUMetric:   "gpu_wait_seconds",
		ErrorMetric: "env_error_rate",
	}

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

    // Queue directories (file-queue integration)
    baseQueue := filepath.Join("logs", "env_queue")
    pendDir := filepath.Join(baseQueue, "pending")
    doneDir := filepath.Join(baseQueue, "done")
    _ = os.MkdirAll(pendDir, 0o755)
    _ = os.MkdirAll(doneDir, 0o755)

    // HTTP mux: metrics + queue submit
    mux := http.NewServeMux()
    mux.Handle("/metrics", registry.Handler())
    mux.HandleFunc("/queue/submit", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost {
            http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
            return
        }
        body, err := io.ReadAll(r.Body)
        if err != nil { http.Error(w, err.Error(), 400); return }
        var req struct{ ID, Class, Payload string }
        _ = json.Unmarshal(body, &req)
        if req.ID == "" { req.ID = time.Now().Format("20060102T150405.000000000") }
        if req.Class == "" { req.Class = "cpu_short" }
        // write file to pending
        fpath := filepath.Join(pendDir, req.ID+".json")
        if err := os.WriteFile(fpath, body, 0o644); err != nil {
            http.Error(w, err.Error(), 500); return
        }
        // bump queue depth gauge by counting pending files
        if n, _ := dirCount(pendDir); n >= 0 { queueGauge.Set(float64(n)) }
        w.Header().Set("Content-Type", "application/json")
        _, _ = w.Write([]byte(`{"ok":true}`))
    })

    server := &http.Server{Addr: ":9097", Handler: mux}

	go func() {
		log.Printf("metrics listening on %s", server.Addr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("metrics server error: %v", err)
		}
	}()

    demoTasks(o, queueGauge, gpuWaitGauge, errorGauge)

	<-ctx.Done()
	log.Println("shutting down orchestrator")
	_ = server.Shutdown(context.Background())
	if err := o.Shutdown(); err != nil {
		log.Printf("shutdown error: %v", err)
	}
}

func demoTasks(o *workflow.Orchestrator, queue *telemetry.Gauge, gpu *telemetry.Gauge, errGauge *telemetry.Gauge) {
	// Simulate metrics oscillation in the background so the adaptive loop has signals.
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

func dirCount(path string) (int, error) {
    f, err := os.Open(path)
    if err != nil { return -1, err }
    defer f.Close()
    names, err := f.Readdirnames(-1)
    if err != nil { return -1, err }
    return len(names), nil
}
