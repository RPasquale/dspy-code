package workflow

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/dspy/orchestrator/internal/workflowspec"
)

// RunStatus describes the overall lifecycle state of a workflow run.
type RunStatus string

const (
	RunStatusPending   RunStatus = "pending"
	RunStatusRunning   RunStatus = "running"
	RunStatusSucceeded RunStatus = "succeeded"
	RunStatusFailed    RunStatus = "failed"
	RunStatusCancelled RunStatus = "cancelled"
)

// StepStatus captures the state of an individual workflow step execution.
type StepStatus string

const (
	StepStatusPending   StepStatus = "pending"
	StepStatusRunning   StepStatus = "running"
	StepStatusSucceeded StepStatus = "succeeded"
	StepStatusFailed    StepStatus = "failed"
	StepStatusSkipped   StepStatus = "skipped"
)

// RunStep records the execution details for a workflow node.
type RunStep struct {
	NodeID      string                `json:"node_id"`
	Name        string                `json:"name"`
	Type        workflowspec.NodeType `json:"type"`
	Action      StepAction            `json:"action"`
	Status      StepStatus            `json:"status"`
	Attempts    int                   `json:"attempts"`
	StartedAt   *time.Time            `json:"started_at,omitempty"`
	CompletedAt *time.Time            `json:"completed_at,omitempty"`
	Result      map[string]any        `json:"result,omitempty"`
	Error       string                `json:"error,omitempty"`
}

// RunRecord persists the high-level execution metadata for a workflow plan.
type RunRecord struct {
	ID              string         `json:"id"`
	WorkflowID      string         `json:"workflow_id"`
	WorkflowName    string         `json:"workflow_name,omitempty"`
	WorkflowVersion string         `json:"workflow_version,omitempty"`
	Status          RunStatus      `json:"status"`
	CreatedAt       time.Time      `json:"created_at"`
	UpdatedAt       time.Time      `json:"updated_at"`
	CompletedAt     *time.Time     `json:"completed_at,omitempty"`
	Steps           []RunStep      `json:"steps"`
	Input           map[string]any `json:"input,omitempty"`
	Metadata        map[string]any `json:"metadata,omitempty"`
	IdempotencyKey  string         `json:"idempotency_key,omitempty"`
}

// RunStore persists workflow run state to disk and supports idempotent creation and updates.
type RunStore struct {
	mu        sync.RWMutex
	dir       string
	runs      map[string]*RunRecord
	idemIndex map[string]string
}

// NewRunStore initialises a run store rooted at dir.
func NewRunStore(dir string) (*RunStore, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, errors.New("run store dir is required")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("ensure run dir: %w", err)
	}

	store := &RunStore{
		dir:       dir,
		runs:      make(map[string]*RunRecord),
		idemIndex: make(map[string]string),
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("scan run dir: %w", err)
	}
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return nil, fmt.Errorf("load run %s: %w", entry.Name(), readErr)
		}
		var record RunRecord
		if err := json.Unmarshal(data, &record); err != nil {
			return nil, fmt.Errorf("decode run %s: %w", entry.Name(), err)
		}
		store.runs[record.ID] = cloneRunRecord(&record)
		if record.IdempotencyKey != "" {
			key := store.idemKey(record.WorkflowID, record.IdempotencyKey)
			store.idemIndex[key] = record.ID
		}
	}

	return store, nil
}

// CreateRun initialises a run for the provided plan. If an idempotencyKey is supplied and a matching run exists,
// the existing run is returned with the second return value set to true.
func (s *RunStore) CreateRun(plan *Plan, wf *workflowspec.Workflow, idempotencyKey string, input map[string]any) (*RunRecord, bool, error) {
	if plan == nil {
		return nil, false, errors.New("plan is nil")
	}
	if wf == nil {
		return nil, false, errors.New("workflow is nil")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if idempotencyKey != "" {
		if existingID, ok := s.idemIndex[s.idemKey(plan.WorkflowID, idempotencyKey)]; ok {
			if existing, found := s.runs[existingID]; found {
				return cloneRunRecord(existing), true, nil
			}
		}
	}

	now := time.Now().UTC()
	runID := generateRunID()
	inputCopy := cloneMap(input)

	steps := make([]RunStep, len(plan.Steps))
	for i, step := range plan.Steps {
		steps[i] = RunStep{
			NodeID:   step.NodeID,
			Name:     step.Name,
			Type:     step.Type,
			Action:   step.Action,
			Status:   StepStatusPending,
			Attempts: 0,
		}
	}

	record := &RunRecord{
		ID:              runID,
		WorkflowID:      wf.ID,
		WorkflowName:    wf.Name,
		WorkflowVersion: wf.Version,
		Status:          RunStatusPending,
		CreatedAt:       now,
		UpdatedAt:       now,
		Steps:           steps,
		Input:           inputCopy,
		Metadata:        map[string]any{"node_count": len(plan.Steps)},
		IdempotencyKey:  idempotencyKey,
	}

	if err := s.persistLocked(record); err != nil {
		return nil, false, err
	}
	s.runs[runID] = cloneRunRecord(record)
	if idempotencyKey != "" {
		s.idemIndex[s.idemKey(plan.WorkflowID, idempotencyKey)] = runID
	}

	return cloneRunRecord(record), false, nil
}

// GetRun fetches a run by identifier.
func (s *RunStore) GetRun(id string) (*RunRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	record, ok := s.runs[id]
	if !ok {
		return nil, fmt.Errorf("run %s not found", id)
	}
	return cloneRunRecord(record), nil
}

// ListRuns returns runs optionally filtered by workflow identifier.
func (s *RunStore) ListRuns(workflowID string, limit int) ([]RunRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var filtered []*RunRecord
	for _, run := range s.runs {
		if workflowID == "" || run.WorkflowID == workflowID {
			filtered = append(filtered, run)
		}
	}
	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].CreatedAt.After(filtered[j].CreatedAt)
	})
	if limit > 0 && len(filtered) > limit {
		filtered = filtered[:limit]
	}

	out := make([]RunRecord, len(filtered))
	for i, run := range filtered {
		out[i] = *cloneRunRecord(run)
	}
	return out, nil
}

// UpdateRun applies a mutation under lock and persists the updated record.
func (s *RunStore) UpdateRun(id string, mutate func(*RunRecord) error) (*RunRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, ok := s.runs[id]
	if !ok {
		return nil, fmt.Errorf("run %s not found", id)
	}
	updated := cloneRunRecord(current)
	if err := mutate(updated); err != nil {
		return nil, err
	}
	updated.UpdatedAt = time.Now().UTC()
	if err := s.persistLocked(updated); err != nil {
		return nil, err
	}
	s.runs[id] = cloneRunRecord(updated)
	if updated.IdempotencyKey != "" {
		s.idemIndex[s.idemKey(updated.WorkflowID, updated.IdempotencyKey)] = updated.ID
	}
	return cloneRunRecord(updated), nil
}

// persistLocked writes the run record to disk. Caller must hold s.mu when invoking.
func (s *RunStore) persistLocked(record *RunRecord) error {
	if record == nil {
		return errors.New("record is nil")
	}
	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal run %s: %w", record.ID, err)
	}
	tmp := filepath.Join(s.dir, record.ID+".tmp")
	dest := filepath.Join(s.dir, record.ID+".json")
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("write tmp run: %w", err)
	}
	if err := os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("commit run: %w", err)
	}
	return nil
}

// idemKey builds the internal indexing key for idempotency lookups.
func (s *RunStore) idemKey(workflowID, key string) string {
	return workflowID + "::" + key
}

func cloneRunRecord(in *RunRecord) *RunRecord {
	if in == nil {
		return nil
	}
	out := *in
	if in.Steps != nil {
		out.Steps = make([]RunStep, len(in.Steps))
		for i := range in.Steps {
			step := in.Steps[i]
			step.Result = cloneMap(step.Result)
			out.Steps[i] = step
		}
	}
	out.Input = cloneMap(in.Input)
	out.Metadata = cloneMap(in.Metadata)
	if in.CompletedAt != nil {
		t := *in.CompletedAt
		out.CompletedAt = &t
	}
	return &out
}

func cloneMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func generateRunID() string {
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err == nil {
		return "run-" + hex.EncodeToString(buf)
	}
	return fmt.Sprintf("run-%d", time.Now().UnixNano())
}

// LoadRun loads a run record from disk without storing it in-memory. Useful for validation during migrations.
func (s *RunStore) LoadRun(id string) (*RunRecord, error) {
	path := filepath.Join(s.dir, id+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("run %s not found", id)
		}
		return nil, fmt.Errorf("read run: %w", err)
	}
	var record RunRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return nil, fmt.Errorf("decode run: %w", err)
	}
	return &record, nil
}
