package slurm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/dspy/orchestrator/internal/events"
	"github.com/dspy/orchestrator/pkg/telemetry"
)

// SlurmJob represents a submitted Slurm job
type SlurmJob struct {
	ID         string                 `json:"id"`
	TaskID     string                 `json:"task_id"`
	Status     string                 `json:"status"`
	SubmitTime time.Time              `json:"submit_time"`
	StartTime  *time.Time             `json:"start_time,omitempty"`
	EndTime    *time.Time             `json:"end_time,omitempty"`
	ExitCode   *int                   `json:"exit_code,omitempty"`
	Error      string                 `json:"error,omitempty"`
	Payload    map[string]interface{} `json:"payload,omitempty"`
}

// SlurmBridge handles Slurm job submission and monitoring
type SlurmBridge struct {
	submitter   *SlurmSubmitter
	monitor     *SlurmMonitor
	registry    *telemetry.Registry
	eventBus    *events.EventBus
	jobStore    map[string]*SlurmJob
	mu          sync.RWMutex
	reconcileCh chan string
	stopCh      chan struct{}
	queueDir    string
	pendDir     string
	doneDir     string
	scancelPath string
	ctx         context.Context
}

// SlurmSubmitter handles job submission to Slurm
type SlurmSubmitter struct {
	sbatchPath  string
	templateDir string
	workingDir  string
}

// SlurmMonitor handles job status monitoring
type SlurmMonitor struct {
	squeuePath   string
	sacctPath    string
	pollInterval time.Duration
}

// NewSlurmBridge creates a new Slurm bridge
func NewSlurmBridge(registry *telemetry.Registry, queueDir string, eventBus *events.EventBus) *SlurmBridge {
	pendDir := filepath.Join(queueDir, "pending")
	doneDir := filepath.Join(queueDir, "done")
	return &SlurmBridge{
		submitter: &SlurmSubmitter{
			sbatchPath:  "sbatch",
			templateDir: "deploy/slurm",
			workingDir:  ".",
		},
		monitor: &SlurmMonitor{
			squeuePath:   "squeue",
			sacctPath:    "sacct",
			pollInterval: 30 * time.Second,
		},
		registry:    registry,
		eventBus:    eventBus,
		jobStore:    make(map[string]*SlurmJob),
		reconcileCh: make(chan string, 100),
		stopCh:      make(chan struct{}),
		queueDir:    queueDir,
		pendDir:     pendDir,
		doneDir:     doneDir,
		scancelPath: "scancel",
	}
}

// Start begins the Slurm bridge monitoring
func (sb *SlurmBridge) Start(ctx context.Context) error {
	sb.mu.Lock()
	sb.ctx = ctx
	sb.mu.Unlock()
	// Start reconciliation loop
	go sb.reconciliationLoop(ctx)

	// Start monitoring loop
	go sb.monitoringLoop(ctx)

	return nil
}

// Stop stops the Slurm bridge
func (sb *SlurmBridge) Stop() {
	close(sb.stopCh)
}

func (sb *SlurmBridge) publishContext() context.Context {
	sb.mu.RLock()
	defer sb.mu.RUnlock()
	if sb.ctx != nil {
		return sb.ctx
	}
	return context.Background()
}

// SubmitGPUJob submits a GPU job to Slurm
func (sb *SlurmBridge) SubmitGPUJob(ctx context.Context, taskID string, payload map[string]interface{}) (*SlurmJob, error) {
	// Generate sbatch script from template
	script, err := sb.generateSbatchScript(taskID, payload)
	if err != nil {
		return nil, fmt.Errorf("failed to generate sbatch script: %w", err)
	}
	defer os.Remove(script)

	// Submit to Slurm
	jobID, err := sb.submitToSlurm(script)
	if err != nil {
		return nil, fmt.Errorf("failed to submit to Slurm: %w", err)
	}

	// Create job record
	job := &SlurmJob{
		ID:         jobID,
		TaskID:     taskID,
		Status:     "pending",
		SubmitTime: time.Now(),
		Payload:    clonePayload(payload),
	}

	// Store job
	sb.mu.Lock()
	sb.jobStore[taskID] = job
	sb.mu.Unlock()

	// Update metrics
	sb.registry.Counter("slurm_jobs_submitted_total").Inc()

	// Publish event
	if sb.eventBus != nil {
		publishCtx := ctx
		if publishCtx == nil {
			publishCtx = sb.publishContext()
		}
		if err := sb.eventBus.PublishSlurmJobSubmitted(publishCtx, taskID, jobID); err != nil {
			fmt.Printf("slurm: failed to publish submission event: %v\n", err)
		}
	}

	// Trigger reconciliation
	select {
	case sb.reconcileCh <- taskID:
	default:
	}

	return job, nil
}

// GetJobStatus returns the current status of a job
func (sb *SlurmBridge) GetJobStatus(taskID string) (*SlurmJob, bool) {
	sb.mu.RLock()
	defer sb.mu.RUnlock()

	job, exists := sb.jobStore[taskID]
	return job, exists
}

func (sb *SlurmBridge) CancelJob(taskID string) error {
	sb.mu.Lock()
	job, exists := sb.jobStore[taskID]
	sb.mu.Unlock()
	if !exists {
		return fmt.Errorf("slurm job for task %s not found", taskID)
	}
	if job == nil || job.ID == "" {
		return fmt.Errorf("slurm job for task %s missing identifier", taskID)
	}
	cmd := exec.Command(sb.scancelPath, job.ID)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("scancel %s: %v (%s)", job.ID, err, string(output))
	}
	now := time.Now()
	sb.mu.Lock()
	defer sb.mu.Unlock()
	if existing := sb.jobStore[taskID]; existing != nil {
		existing.Status = "cancelled"
		existing.EndTime = &now
		zero := 0
		existing.ExitCode = &zero
	}
	return nil
}

// generateSbatchScript creates an sbatch script from template
func (sb *SlurmBridge) generateSbatchScript(taskID string, payload map[string]interface{}) (string, error) {
	method := strings.ToLower(getString(payload, "method", ""))
	templateName := getString(payload, "template", "")
	if templateName == "" {
		switch method {
		case "puffer_rl", "rl":
			templateName = "train_puffer_rl.sbatch"
		case "ddp":
			templateName = "train_ddp.sbatch"
		default:
			templateName = "train_agent_methodologies.sbatch"
		}
	}

	// Read template
	templatePath := filepath.Join(sb.submitter.templateDir, templateName)
	template, err := os.ReadFile(templatePath)
	if err != nil {
		return "", err
	}

	// Replace placeholders
	script := string(template)
	script = strings.ReplaceAll(script, "${TASK_ID}", taskID)
	script = strings.ReplaceAll(script, "${TRAINING_METHOD}", getString(payload, "method", "grpo"))
	script = strings.ReplaceAll(script, "${MODULE_NAME}", getString(payload, "module", "orchestrator"))
	script = strings.ReplaceAll(script, "${MODEL_NAME}", getString(payload, "model", "gpt2"))
	script = strings.ReplaceAll(script, "${RL_SKILL_OVERRIDE}", getString(payload, "skill", ""))

	// Common resource overrides
	script = strings.ReplaceAll(script, "${NODES:-1}", strconv.Itoa(getInt(payload, "nodes", 1)))
	script = strings.ReplaceAll(script, "${GPUS:-1}", strconv.Itoa(getInt(payload, "gpus", 1)))
	script = strings.ReplaceAll(script, "${CPUS_PER_TASK:-8}", strconv.Itoa(getInt(payload, "cpus_per_task", 8)))
	script = strings.ReplaceAll(script, "${MEMORY_GB:-48}", strconv.Itoa(getInt(payload, "memory_gb", 48)))
	script = strings.ReplaceAll(script, "${TIME_LIMIT:-04:00:00}", getString(payload, "time_limit", "04:00:00"))
	script = strings.ReplaceAll(script, "${LOG_DIR:-/workspace/logs}", getString(payload, "log_dir", "/workspace/logs"))

	if templateName == "train_puffer_rl.sbatch" {
		workspace := getString(payload, "workspace_dir", getString(payload, "workspace", "."))
		script = strings.ReplaceAll(script, "${WORKSPACE_DIR:-$SLURM_SUBMIT_DIR}", workspace)
		script = strings.ReplaceAll(script, "${RL_STEPS:-1000}", strconv.Itoa(getInt(payload, "steps", 1000)))
		script = strings.ReplaceAll(script, "${RL_N_ENVS:-4}", strconv.Itoa(getInt(payload, "n_envs", 4)))
		script = strings.ReplaceAll(script, "${RL_LR:-0.001}", getString(payload, "lr", "0.001"))
		script = strings.ReplaceAll(script, "${RL_ENTROPY:-0.01}", getString(payload, "entropy", "0.01"))
		script = strings.ReplaceAll(script, "${RL_REPLAY_CAPACITY:-4096}", strconv.Itoa(getInt(payload, "replay_capacity", 4096)))
		script = strings.ReplaceAll(script, "${RL_REPLAY_BATCH:-256}", strconv.Itoa(getInt(payload, "replay_batch", 256)))
		script = strings.ReplaceAll(script, "${RL_GRAD_CLIP:-1.0}", getString(payload, "grad_clip", "1.0"))
		script = strings.ReplaceAll(script, "${RL_CHECKPOINT_DIR:-$WORKSPACE/logs/rl_checkpoints}", getString(payload, "checkpoint_dir", filepath.Join(workspace, "logs", "rl_checkpoints")))
		script = strings.ReplaceAll(script, "${RL_CHECKPOINT_INTERVAL:-0}", strconv.Itoa(getInt(payload, "checkpoint_interval", 0)))
		script = strings.ReplaceAll(script, "${RL_EARLY_STOP:-0}", strconv.Itoa(getInt(payload, "early_stop", 0)))
		script = strings.ReplaceAll(script, "${RL_LOG_INTERVAL:-10}", strconv.Itoa(getInt(payload, "log_interval", 10)))
		script = strings.ReplaceAll(script, "${RL_SKIP_GEPA:-0}", strconv.Itoa(getInt(payload, "skip_gepa", 0)))
		script = strings.ReplaceAll(script, "${RL_GEPA_MODULES:-}", getString(payload, "gepa_modules", ""))
		script = strings.ReplaceAll(script, "${RL_LOG_JSONL:-$WORKSPACE/logs/rl_task.jsonl}", getString(payload, "log_jsonl", filepath.Join(workspace, "logs", fmt.Sprintf("rl_%s.jsonl", taskID))))
		script = strings.ReplaceAll(script, "${MESH_ENDPOINT:-}", getString(payload, "mesh_endpoint", ""))
		script = strings.ReplaceAll(script, "${MESH_NODE_ID:-9002}", strconv.Itoa(getInt(payload, "mesh_node_id", 9002)))
		script = strings.ReplaceAll(script, "${MESH_DOMAIN:-default}", getString(payload, "mesh_domain", "default"))
		script = strings.ReplaceAll(script, "${SUPERVISOR_GRPC_ADDR:-http://127.0.0.1:7000}", getString(payload, "supervisor_addr", "http://127.0.0.1:7000"))
	}

	// Write to temp file
	tmpDir := sb.submitter.workingDir
	if tmpDir == "" {
		tmpDir = os.TempDir()
	}
	fh, err := os.CreateTemp(tmpDir, fmt.Sprintf("slurm_%s_*.sbatch", taskID))
	if err != nil {
		return "", err
	}
	if _, err := fh.WriteString(script); err != nil {
		fh.Close()
		os.Remove(fh.Name())
		return "", err
	}
	if err := fh.Close(); err != nil {
		os.Remove(fh.Name())
		return "", err
	}

	return fh.Name(), nil
}

// submitToSlurm submits the script to Slurm
func (sb *SlurmBridge) submitToSlurm(scriptPath string) (string, error) {
	cmd := exec.Command(sb.submitter.sbatchPath, scriptPath)
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}

	// Parse job ID from output (format: "Submitted batch job 12345")
	outputStr := strings.TrimSpace(string(output))
	parts := strings.Fields(outputStr)
	if len(parts) < 4 || parts[0] != "Submitted" || parts[1] != "batch" || parts[2] != "job" {
		return "", fmt.Errorf("unexpected sbatch output: %s", outputStr)
	}

	return parts[3], nil
}

// reconciliationLoop continuously reconciles job statuses
func (sb *SlurmBridge) reconciliationLoop(ctx context.Context) {
	ticker := time.NewTicker(sb.monitor.pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-sb.stopCh:
			return
		case taskID := <-sb.reconcileCh:
			sb.reconcileJob(taskID)
		case <-ticker.C:
			sb.reconcileAllJobs()
		}
	}
}

// monitoringLoop continuously monitors job statuses
func (sb *SlurmBridge) monitoringLoop(ctx context.Context) {
	ticker := time.NewTicker(sb.monitor.pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-sb.stopCh:
			return
		case <-ticker.C:
			sb.monitorJobs()
		}
	}
}

// reconcileJob reconciles a specific job
func (sb *SlurmBridge) reconcileJob(taskID string) {
	sb.mu.RLock()
	job, exists := sb.jobStore[taskID]
	sb.mu.RUnlock()

	if !exists {
		return
	}

	// Check job status via squeue
	status, err := sb.getJobStatus(job.ID)
	if err != nil {
		sb.registry.Counter("slurm_reconcile_errors_total").Inc()
		return
	}

	// Update job status
	sb.mu.Lock()
	job.Status = status
	sb.mu.Unlock()

	// If job is completed, get final status
	if status == "completed" || status == "failed" {
		sb.finalizeJob(taskID, job)
	}
}

// reconcileAllJobs reconciles all active jobs
func (sb *SlurmBridge) reconcileAllJobs() {
	sb.mu.RLock()
	activeJobs := make([]string, 0, len(sb.jobStore))
	for taskID, job := range sb.jobStore {
		if job.Status == "pending" || job.Status == "running" {
			activeJobs = append(activeJobs, taskID)
		}
	}
	sb.mu.RUnlock()

	for _, taskID := range activeJobs {
		sb.reconcileJob(taskID)
	}
}

// monitorJobs monitors all jobs using squeue
func (sb *SlurmBridge) monitorJobs() {
	// Get all active jobs from squeue
	activeJobs, err := sb.getActiveJobs()
	if err != nil {
		sb.registry.Counter("slurm_monitor_errors_total").Inc()
		return
	}

	// Update job store
	sb.mu.Lock()
	for taskID, job := range sb.jobStore {
		if status, exists := activeJobs[job.ID]; exists {
			job.Status = status
		} else {
			state := normalizeState(job.Status)
			if state == "pending" || state == "running" {
				// Job not in squeue, check if completed
				sb.checkCompletedJob(taskID, job)
			}
		}
	}
	sb.mu.Unlock()
}

// getJobStatus gets the status of a specific job
func (sb *SlurmBridge) getJobStatus(jobID string) (string, error) {
	cmd := exec.Command(sb.monitor.squeuePath, "-j", jobID, "--format=%T", "--noheader")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}

	status := normalizeState(string(output))
	if status == "" {
		return "completed", nil
	}

	return status, nil
}

// getActiveJobs gets all active jobs from squeue
func (sb *SlurmBridge) getActiveJobs() (map[string]string, error) {
	cmd := exec.Command(sb.monitor.squeuePath, "--format=%i,%T", "--noheader")
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	activeJobs := make(map[string]string)
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) == 2 {
			activeJobs[parts[0]] = normalizeState(parts[1])
		}
	}

	return activeJobs, nil
}

// checkCompletedJob checks if a job has completed
func (sb *SlurmBridge) checkCompletedJob(taskID string, job *SlurmJob) {
	// Use sacct to get job completion status
	cmd := exec.Command(sb.monitor.sacctPath, "-j", job.ID, "--format=JobID,State,ExitCode", "--noheader")
	output, err := cmd.Output()
	if err != nil {
		return
	}

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) >= 3 && parts[0] == job.ID {
			state := normalizeState(parts[1])
			exitCode := parts[2]

			job.Status = state
			if exitCode != "0:0" {
				if code, err := strconv.Atoi(strings.Split(exitCode, ":")[0]); err == nil {
					job.ExitCode = &code
				}
			}

			sb.finalizeJob(taskID, job)
			break
		}
	}
}

// finalizeJob finalizes a completed job
func (sb *SlurmBridge) finalizeJob(taskID string, job *SlurmJob) {
	sb.mu.Lock()
	if job.EndTime != nil {
		sb.mu.Unlock()
		return
	}
	now := time.Now()
	job.EndTime = &now
	status := normalizeState(job.Status)
	job.Status = status
	if status != "completed" && job.Error == "" {
		job.Error = status
	}
	sb.mu.Unlock()

	switch status {
	case "completed":
		sb.registry.Counter("slurm_jobs_completed_total").Inc()
		if sb.eventBus != nil {
			ctx := sb.publishContext()
			if err := sb.eventBus.PublishSlurmJobCompleted(ctx, taskID, job.ID, map[string]interface{}{"status": status}); err != nil {
				fmt.Printf("slurm: failed to publish completion event: %v\n", err)
			}
		}
	default:
		sb.registry.Counter("slurm_jobs_failed_total").Inc()
		if sb.eventBus != nil {
			ctx := sb.publishContext()
			if err := sb.eventBus.PublishSlurmJobCompleted(ctx, taskID, job.ID, map[string]interface{}{"status": status, "error": job.Error}); err != nil {
				fmt.Printf("slurm: failed to publish completion event: %v\n", err)
			}
			if err := sb.eventBus.PublishTaskFailed(ctx, taskID, job.Error); err != nil {
				fmt.Printf("slurm: failed to publish failure event: %v\n", err)
			}
		}
	}

	// Move task to done queue
	sb.moveTaskToDone(taskID, job)
}

// moveTaskToDone moves a completed task to the done queue
func (sb *SlurmBridge) moveTaskToDone(taskID string, job *SlurmJob) {
	pendingPath := filepath.Join(sb.pendDir, taskID+".json")
	donePath := filepath.Join(sb.doneDir, taskID+".json")

	var task map[string]interface{}
	if data, err := os.ReadFile(pendingPath); err == nil {
		if err := json.Unmarshal(data, &task); err != nil {
			task = make(map[string]interface{})
		}
	} else {
		task = make(map[string]interface{})
	}

	if task == nil {
		task = make(map[string]interface{})
	}
	if _, ok := task["id"]; !ok {
		task["id"] = taskID
	}
	if _, ok := task["class"]; !ok {
		task["class"] = "gpu_slurm"
	}
	if job.Payload != nil {
		task["payload"] = job.Payload
	}

	task["slurm_job"] = job
	task["completed_at"] = time.Now()

	if err := os.MkdirAll(sb.doneDir, 0o755); err != nil {
		fmt.Printf("slurm: failed to ensure done dir: %v\n", err)
	}

	doneData, err := json.MarshalIndent(task, "", "  ")
	if err != nil {
		fmt.Printf("slurm: failed to marshal done task: %v\n", err)
		return
	}

	if err := os.WriteFile(donePath, doneData, 0o644); err != nil {
		fmt.Printf("slurm: failed to write done task: %v\n", err)
	}

	_ = os.Remove(pendingPath)
}

// getString safely gets a string value from a map
func getString(m map[string]interface{}, key, defaultValue string) string {
	if val, exists := m[key]; exists {
		if str, ok := val.(string); ok {
			return str
		}
	}
	return defaultValue
}

func getInt(m map[string]interface{}, key string, defaultValue int) int {
	if val, exists := m[key]; exists {
		switch v := val.(type) {
		case int:
			return v
		case int32:
			return int(v)
		case int64:
			return int(v)
		case float64:
			return int(v)
		case float32:
			return int(v)
		case json.Number:
			if iv, err := v.Int64(); err == nil {
				return int(iv)
			}
		case string:
			if iv, err := strconv.Atoi(v); err == nil {
				return iv
			}
		}
	}
	return defaultValue
}

func clonePayload(src map[string]interface{}) map[string]interface{} {
	if src == nil {
		return nil
	}
	cloned := make(map[string]interface{}, len(src))
	for k, v := range src {
		cloned[k] = v
	}
	return cloned
}

func normalizeState(state string) string {
	s := strings.ToLower(strings.TrimSpace(state))
	s = strings.TrimSuffix(s, ",")
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	switch s {
	case "running", "run", "r", "cg", "completing":
		return "running"
	case "pending", "pd", "configuring", "cf", "suspended", "susp", "s":
		return "pending"
	case "completed", "complete", "cd", "done", "success", "ok":
		return "completed"
	}
	if strings.HasPrefix(s, "preempt") {
		return "failed"
	}
	if strings.HasPrefix(s, "cancel") {
		return "failed"
	}
	if strings.HasPrefix(s, "fail") {
		return "failed"
	}
	if strings.Contains(s, "timeout") || strings.Contains(s, "time_limit") {
		return "failed"
	}
	if strings.Contains(s, "node_fail") || strings.Contains(s, "oom") || strings.Contains(s, "dead") {
		return "failed"
	}
	return s
}
