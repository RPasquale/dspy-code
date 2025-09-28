package slurm

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/dspy/orchestrator/pkg/telemetry"
)

func TestNormalizeState(t *testing.T) {
	cases := map[string]string{
		"PENDING":    "pending",
		"Running":    "running",
		"COMPLETED":  "completed",
		"COMPLETING": "running",
		"FAILED":     "failed",
		"CANCELLED":  "failed",
		"TIMEOUT":    "failed",
		"":           "",
		"unknown":    "unknown",
	}

	for input, expected := range cases {
		if got := normalizeState(input); got != expected {
			t.Fatalf("normalizeState(%q) = %q, expected %q", input, got, expected)
		}
	}
}

func TestMoveTaskToDoneCreatesRecord(t *testing.T) {
	queueDir := t.TempDir()
	registry := telemetry.NewRegistry()
	bridge := NewSlurmBridge(registry, queueDir, nil)

	job := &SlurmJob{
		ID:         "123",
		TaskID:     "task-1",
		Status:     "completed",
		SubmitTime: time.Now(),
		Payload: map[string]interface{}{
			"method": "grpo",
		},
	}

	bridge.moveTaskToDone("task-1", job)

	doneFile := filepath.Join(queueDir, "done", "task-1.json")
	if _, err := os.Stat(doneFile); err != nil {
		t.Fatalf("expected done file to be created: %v", err)
	}

	data, err := os.ReadFile(doneFile)
	if err != nil {
		t.Fatalf("failed to read done file: %v", err)
	}

	var record map[string]interface{}
	if err := json.Unmarshal(data, &record); err != nil {
		t.Fatalf("failed to unmarshal done file: %v", err)
	}

	if record["id"] != "task-1" {
		t.Fatalf("expected task id to be preserved, got %v", record["id"])
	}

	slurmJob, ok := record["slurm_job"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected slurm_job field to be present")
	}
	if slurmJob["id"] != "123" {
		t.Fatalf("expected slurm job id 123, got %v", slurmJob["id"])
	}

	if _, err := os.Stat(filepath.Join(queueDir, "pending", "task-1.json")); !os.IsNotExist(err) {
		t.Fatalf("expected pending file to be removed, got err=%v", err)
	}
}
