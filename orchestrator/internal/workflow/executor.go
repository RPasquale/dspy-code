package workflow

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/dspy/orchestrator/internal/events"
	"github.com/dspy/orchestrator/internal/runner"
	"github.com/dspy/orchestrator/internal/slurm"
	"github.com/dspy/orchestrator/internal/workflowspec"
)

// Executor coordinates workflow run execution against the adaptive orchestrator.
type Executor struct {
	orchestrator     *Orchestrator
	store            *RunStore
	workflowStore    workflowspec.Store
	runnerClient     *runner.HTTPClient
	slurmBridge      *slurm.SlurmBridge
	eventBus         *events.EventBus
	maxSlurmWait     time.Duration
	hardwareProvider func() map[string]any
}

// ExecutorOption allows configuration overrides when constructing an Executor.
type ExecutorOption func(*Executor)

// WithMaxSlurmWait sets the maximum time the executor will wait for a Slurm job to complete before aborting the step.
func WithMaxSlurmWait(d time.Duration) ExecutorOption {
	return func(e *Executor) {
		e.maxSlurmWait = d
	}
}

// WithHardwareProvider configures a callback used to enrich payloads with runner hardware metadata.
func WithHardwareProvider(provider func() map[string]any) ExecutorOption {
	return func(e *Executor) {
		e.hardwareProvider = provider
	}
}

// NewExecutor constructs an executor.
func NewExecutor(orchestrator *Orchestrator, store *RunStore, wfStore workflowspec.Store, runnerClient *runner.HTTPClient, slurmBridge *slurm.SlurmBridge, eventBus *events.EventBus, opts ...ExecutorOption) *Executor {
	exec := &Executor{
		orchestrator:  orchestrator,
		store:         store,
		workflowStore: wfStore,
		runnerClient:  runnerClient,
		slurmBridge:   slurmBridge,
		eventBus:      eventBus,
		maxSlurmWait:  45 * time.Minute,
	}
	for _, opt := range opts {
		opt(exec)
	}
	if exec.maxSlurmWait <= 0 {
		exec.maxSlurmWait = 45 * time.Minute
	}
	return exec
}

// StartRun materialises a workflow run and schedules its execution. The returned RunRecord reflects the persisted state
// immediately after creation (or retrieval in the idempotent case).
func (e *Executor) StartRun(ctx context.Context, workflowID, idempotencyKey string, input map[string]any) (*RunRecord, error) {
	if e == nil {
		return nil, errors.New("executor is nil")
	}
	if e.orchestrator == nil {
		return nil, errors.New("executor orchestrator is nil")
	}
	if e.store == nil {
		return nil, errors.New("executor store is nil")
	}
	if e.workflowStore == nil {
		return nil, errors.New("executor workflow store is nil")
	}

	wf, err := e.workflowStore.Get(workflowID)
	if err != nil {
		return nil, fmt.Errorf("load workflow %s: %w", workflowID, err)
	}
	plan, err := BuildPlan(wf)
	if err != nil {
		return nil, fmt.Errorf("build plan: %w", err)
	}
	record, reused, err := e.store.CreateRun(plan, wf, idempotencyKey, input)
	if err != nil {
		return nil, fmt.Errorf("create run: %w", err)
	}

	runID := record.ID
	shouldExecute := true
	switch record.Status {
	case RunStatusSucceeded, RunStatusCancelled:
		shouldExecute = false
	case RunStatusRunning:
		if reused {
			shouldExecute = false
		}
	}

	if shouldExecute {
		e.orchestrator.Go("workflow_run_"+runID, func(taskCtx context.Context) error {
			execCtx, cancel := context.WithCancel(taskCtx)
			defer cancel()
			if err := e.executeRun(execCtx, runID, plan, wf); err != nil {
				return err
			}
			return nil
		})
	}

	return record, nil
}

func (e *Executor) executeRun(ctx context.Context, runID string, plan *Plan, wf *workflowspec.Workflow) error {
	if e.eventBus != nil {
		_ = e.eventBus.PublishTaskSubmitted(ctx, runID, map[string]any{"workflow_id": wf.ID, "type": "workflow_run"})
	}

	if _, err := e.store.UpdateRun(runID, func(r *RunRecord) error {
		if r.Status == RunStatusPending {
			r.Status = RunStatusRunning
		}
		return nil
	}); err != nil {
		return err
	}

	completedSteps := make([]PlanStep, 0, len(plan.Steps))

	for _, step := range plan.Steps {
		select {
		case <-ctx.Done():
			_, _ = e.store.UpdateRun(runID, func(r *RunRecord) error {
				r.Status = RunStatusCancelled
				now := time.Now().UTC()
				r.CompletedAt = &now
				for i := range r.Steps {
					if r.Steps[i].NodeID == step.NodeID {
						if r.Steps[i].Status == StepStatusRunning {
							r.Steps[i].Status = StepStatusFailed
							r.Steps[i].Error = "context cancelled"
						}
					} else if r.Steps[i].Status == StepStatusPending {
						r.Steps[i].Status = StepStatusSkipped
					}
				}
				return nil
			})
			return ctx.Err()
		default:
		}

		runSnapshot, err := e.store.UpdateRun(runID, func(r *RunRecord) error {
			s := findRunStep(r, step.NodeID)
			if s == nil {
				return fmt.Errorf("step %s missing in run", step.NodeID)
			}
			switch s.Status {
			case StepStatusSucceeded:
				return nil
			case StepStatusRunning:
				// Treat as retry after interruption
			default:
				// set to running
			}
			now := time.Now().UTC()
			s.Status = StepStatusRunning
			s.Attempts++
			s.StartedAt = &now
			s.CompletedAt = nil
			s.Error = ""
			s.Result = nil
			return nil
		})
		if err != nil {
			return err
		}

		stepResult, execErr := e.executeStep(ctx, runSnapshot, wf, step)
		if execErr != nil {
			rollbackErr := e.rollback(ctx, runSnapshot, wf, step, completedSteps)
			if rollbackErr != nil {
				log.Printf("workflow rollback error for run %s step %s: %v", runID, step.NodeID, rollbackErr)
			}
			_, _ = e.store.UpdateRun(runID, func(r *RunRecord) error {
				s := findRunStep(r, step.NodeID)
				if s != nil {
					now := time.Now().UTC()
					s.CompletedAt = &now
					s.Status = StepStatusFailed
					s.Error = execErr.Error()
				}
				r.Status = RunStatusFailed
				now := time.Now().UTC()
				r.CompletedAt = &now
				for i := range r.Steps {
					if r.Steps[i].NodeID != step.NodeID && r.Steps[i].Status == StepStatusPending {
						r.Steps[i].Status = StepStatusSkipped
					}
				}
				return nil
			})
			if e.eventBus != nil {
				_ = e.eventBus.PublishTaskFailed(ctx, runID, execErr.Error())
			}
			return execErr
		}

		_, err = e.store.UpdateRun(runID, func(r *RunRecord) error {
			s := findRunStep(r, step.NodeID)
			if s == nil {
				return fmt.Errorf("step %s missing in run", step.NodeID)
			}
			now := time.Now().UTC()
			s.CompletedAt = &now
			s.Status = StepStatusSucceeded
			s.Result = stepResult
			return nil
		})
		if err != nil {
			return err
		}
		completedSteps = append(completedSteps, step)
	}

	_, err := e.store.UpdateRun(runID, func(r *RunRecord) error {
		r.Status = RunStatusSucceeded
		now := time.Now().UTC()
		r.CompletedAt = &now
		return nil
	})
	if err != nil {
		return err
	}

	if e.eventBus != nil {
		_ = e.eventBus.PublishTaskCompleted(ctx, runID, map[string]any{"workflow_id": wf.ID, "status": "succeeded"})
	}
	return nil
}

func (e *Executor) executeStep(ctx context.Context, run *RunRecord, wf *workflowspec.Workflow, step PlanStep) (map[string]any, error) {
	switch step.Action.Type {
	case ActionSignature:
		return e.executeSignatureStep(ctx, run, wf, step)
	case ActionVerifier:
		return e.executeVerifierStep(ctx, run, wf, step)
	case ActionReward:
		return e.executeRewardStep(ctx, run, wf, step)
	case ActionTraining:
		return e.executeTrainingStep(ctx, run, wf, step)
	case ActionDeployment:
		return e.executeDeploymentStep(ctx, run, wf, step)
	case ActionCustom:
		res := map[string]any{"ack": true, "node_id": step.NodeID, "type": "custom"}
		if step.Action.Custom != nil {
			res["config"] = step.Action.Custom
		}
		return res, nil
	default:
		return nil, fmt.Errorf("unsupported step action %s", step.Action.Type)
	}
}

func (e *Executor) executeSignatureStep(ctx context.Context, run *RunRecord, wf *workflowspec.Workflow, step PlanStep) (map[string]any, error) {
	if e.runnerClient == nil {
		return nil, errors.New("runner client not configured")
	}
	cfg := step.Action.Signature
	if cfg == nil {
		return nil, fmt.Errorf("signature step %s missing config", step.NodeID)
	}
	payload := map[string]any{
		"workflow_id":      run.WorkflowID,
		"workflow_run_id":  run.ID,
		"node_id":          step.NodeID,
		"prompt":           cfg.Prompt,
		"tools":            append([]string(nil), cfg.Tools...),
		"runtime":          cfg.Runtime,
		"max_tokens":       cfg.MaxTokens,
		"temperature":      cfg.Temperature,
		"execution_domain": cfg.ExecutionDomain,
	}
	for k, v := range run.Input {
		payload[k] = v
	}
	payload["workflow_context"] = BuildWorkflowContext(wf)
	payload["step_kind"] = "signature"
	e.injectHardware(payload)

	req := runner.TaskRequest{
		ID:      fmt.Sprintf("%s-%s", run.ID, step.NodeID),
		Class:   deriveClassFromSignature(cfg),
		Payload: payload,
	}
	resp, err := e.runnerClient.ExecuteTask(ctx, req)
	if err != nil {
		return nil, err
	}
	if e.eventBus != nil {
		_ = e.eventBus.PublishTaskCompleted(ctx, req.ID, map[string]any{
			"workflow_run_id": run.ID,
			"node_id":         step.NodeID,
			"type":            "signature",
			"latency_ms":      resp.LatencyMs,
		})
	}
	result := map[string]any{
		"task_id":    resp.ID,
		"latency_ms": resp.LatencyMs,
		"embeddings": resp.Embeddings,
		"metadata":   resp.Metadata,
	}
	return result, nil
}

func (e *Executor) executeVerifierStep(ctx context.Context, run *RunRecord, wf *workflowspec.Workflow, step PlanStep) (map[string]any, error) {
	if e.runnerClient == nil {
		return nil, errors.New("runner client not configured")
	}
	cfg := step.Action.Verifier
	if cfg == nil {
		return nil, fmt.Errorf("verifier step %s missing config", step.NodeID)
	}
	payload := map[string]any{
		"workflow_id":      run.WorkflowID,
		"workflow_run_id":  run.ID,
		"node_id":          step.NodeID,
		"command":          cfg.Command,
		"weight":           cfg.Weight,
		"penalty":          cfg.Penalty,
		"timeout_sec":      cfg.TimeoutSec,
		"success_signal":   cfg.SuccessSignal,
		"workflow_context": BuildWorkflowContext(wf),
		"step_kind":        "verifier",
	}
	e.injectHardware(payload)
	req := runner.TaskRequest{
		ID:      fmt.Sprintf("%s-%s", run.ID, step.NodeID),
		Class:   "cpu_short",
		Payload: payload,
	}
	resp, err := e.runnerClient.ExecuteTask(ctx, req)
	if err != nil {
		return nil, err
	}
	if e.eventBus != nil {
		_ = e.eventBus.PublishTaskCompleted(ctx, req.ID, map[string]any{
			"workflow_run_id": run.ID,
			"node_id":         step.NodeID,
			"type":            "verifier",
			"latency_ms":      resp.LatencyMs,
		})
	}
	result := map[string]any{
		"latency_ms": resp.LatencyMs,
		"metadata":   resp.Metadata,
		"embeddings": resp.Embeddings,
	}
	return result, nil
}

func (e *Executor) executeRewardStep(ctx context.Context, run *RunRecord, wf *workflowspec.Workflow, step PlanStep) (map[string]any, error) {
	cfg := step.Action.Reward
	if cfg == nil {
		return nil, fmt.Errorf("reward step %s missing config", step.NodeID)
	}
	payload := map[string]any{
		"workflow_id":      run.WorkflowID,
		"workflow_run_id":  run.ID,
		"node_id":          step.NodeID,
		"metric":           cfg.Metric,
		"target":           cfg.Target,
		"aggregator":       cfg.Aggregator,
		"window":           cfg.Window,
		"scale":            cfg.Scale,
		"workflow_context": BuildWorkflowContext(wf),
	}
	e.injectHardware(payload)
	if e.eventBus != nil {
		if err := e.eventBus.PublishTaskSubmitted(ctx, fmt.Sprintf("reward-%s", run.ID), payload); err != nil {
			log.Printf("failed to publish reward event: %v", err)
		}
	}
	return payload, nil
}

func (e *Executor) executeTrainingStep(ctx context.Context, run *RunRecord, wf *workflowspec.Workflow, step PlanStep) (map[string]any, error) {
	if e.slurmBridge == nil {
		return nil, errors.New("slurm bridge not configured")
	}
	cfg := step.Action.Training
	if cfg == nil {
		return nil, fmt.Errorf("training step %s missing config", step.NodeID)
	}
	payload := map[string]any{
		"workflow_id":      run.WorkflowID,
		"workflow_run_id":  run.ID,
		"node_id":          step.NodeID,
		"method":           cfg.Method,
		"skill":            cfg.Skill,
		"dataset":          cfg.Dataset,
		"schedule":         cfg.Schedule,
		"max_steps":        cfg.MaxSteps,
		"slurm_profile":    cfg.SlurmProfile,
		"trigger_metric":   cfg.TriggerMetric,
		"workflow_context": BuildWorkflowContext(wf),
	}
	e.injectHardware(payload)

	job, err := e.slurmBridge.SubmitGPUJob(ctx, fmt.Sprintf("%s-%s", run.ID, step.NodeID), payload)
	if err != nil {
		return nil, err
	}

	waitCtx, cancel := context.WithTimeout(ctx, e.maxSlurmWait)
	defer cancel()
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-waitCtx.Done():
			return map[string]any{"slurm_job": job}, fmt.Errorf("slurm wait exceeded for job %s", job.ID)
		case <-ticker.C:
			latest, exists := e.slurmBridge.GetJobStatus(job.TaskID)
			if !exists {
				continue
			}
			switch latest.Status {
			case "completed":
				return map[string]any{"slurm_job": latest}, nil
			case "failed", "cancelled", "timeout":
				return map[string]any{"slurm_job": latest}, fmt.Errorf("slurm job %s ended with status %s", job.ID, latest.Status)
			}
		}
	}
}

func (e *Executor) executeDeploymentStep(ctx context.Context, run *RunRecord, wf *workflowspec.Workflow, step PlanStep) (map[string]any, error) {
	cfg := step.Action.Deployment
	if cfg == nil {
		return nil, fmt.Errorf("deployment step %s missing config", step.NodeID)
	}
	payload := map[string]any{
		"workflow_id":      run.WorkflowID,
		"workflow_run_id":  run.ID,
		"node_id":          step.NodeID,
		"tenant":           cfg.Tenant,
		"domain":           cfg.Domain,
		"channel":          cfg.Channel,
		"strategy":         cfg.Strategy,
		"autopromote":      cfg.Autopromote,
		"workflow_context": BuildWorkflowContext(wf),
	}
	e.injectHardware(payload)
	if e.eventBus != nil {
		if err := e.eventBus.PublishTaskSubmitted(ctx, fmt.Sprintf("deploy-%s", run.ID), payload); err != nil {
			log.Printf("failed to publish deployment event: %v", err)
		}
	}
	return payload, nil
}

func (e *Executor) rollback(ctx context.Context, run *RunRecord, wf *workflowspec.Workflow, failedStep PlanStep, succeeded []PlanStep) error {
	// Attempt to cancel any Slurm job associated with the failed step or previously succeeded steps.
	if e.slurmBridge == nil {
		return nil
	}
	cancelTargets := append([]PlanStep{failedStep}, succeeded...)
	for _, step := range cancelTargets {
		if step.Action.Type != ActionTraining {
			continue
		}
		taskID := fmt.Sprintf("%s-%s", run.ID, step.NodeID)
		if err := e.slurmBridge.CancelJob(taskID); err != nil {
			log.Printf("failed to cancel slurm job for task %s: %v", taskID, err)
		}
	}
	return nil
}

func deriveClassFromSignature(cfg *workflowspec.SignatureConfig) string {
	if cfg == nil {
		return "cpu_short"
	}
	switch cfg.ExecutionDomain {
	case "gpu", "cuda", "accelerated":
		return "gpu_slurm"
	case "cpu_long":
		return "cpu_long"
	default:
		return "cpu_short"
	}
}

func (e *Executor) injectHardware(payload map[string]any) {
	if payload == nil || e.hardwareProvider == nil {
		return
	}
	if _, exists := payload["runner_hardware"]; exists {
		return
	}
	if snapshot := e.hardwareProvider(); snapshot != nil {
		payload["runner_hardware"] = snapshot
	}
}

func findRunStep(run *RunRecord, nodeID string) *RunStep {
	if run == nil {
		return nil
	}
	for i := range run.Steps {
		if run.Steps[i].NodeID == nodeID {
			return &run.Steps[i]
		}
	}
	return nil
}
