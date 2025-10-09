package main

import (
	"sync"
	"testing"
	"time"

	"github.com/dspy/orchestrator/internal/slurm"
	"github.com/dspy/orchestrator/pkg/telemetry"
	"go.uber.org/zap"
)

type stubTraining struct {
	mu     sync.Mutex
	calls  int
	skills []string
}

func (s *stubTraining) StartTraining(req trainingStartRequest) (*slurm.SlurmJob, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls++
	s.skills = append(s.skills, req.Skill)
	return nil, nil
}

func TestSkillPlannerTriggersAfterThreshold(t *testing.T) {
	stub := &stubTraining{}
	metrics := newSupervisorMetrics(telemetry.NewRegistry())
	planner := newSkillPlanner(zap.NewNop(), stub, metrics)
	if planner == nil {
		t.Fatalf("expected planner")
	}
	planner.threshold = 0.5
	planner.minSamples = 4
	planner.cooldown = 0

	for i := 0; i < 3; i++ {
		planner.Observe("codegen", false)
	}
	if stub.calls != 0 {
		t.Fatalf("unexpected trigger before threshold")
	}
	planner.Observe("codegen", false)
	if stub.calls != 1 {
		t.Fatalf("expected one training trigger, got %d", stub.calls)
	}
}

func TestSkillPlannerCooldownResetsCounters(t *testing.T) {
	stub := &stubTraining{}
	metrics := newSupervisorMetrics(telemetry.NewRegistry())
	planner := newSkillPlanner(zap.NewNop(), stub, metrics)
	if planner == nil {
		t.Fatalf("expected planner")
	}
	planner.threshold = 0.25
	planner.minSamples = 2
	planner.cooldown = 10 * time.Millisecond

	planner.Observe("router", false)
	planner.Observe("router", false)
	if stub.calls != 1 {
		t.Fatalf("expected trigger after failures, got %d", stub.calls)
	}

	// Wait for cooldown and feed more samples to ensure counters reset
	time.Sleep(20 * time.Millisecond)
	planner.Observe("router", true)
	planner.Observe("router", false)
	planner.Observe("router", false)
	if stub.calls != 2 {
		t.Fatalf("expected second trigger after cooldown reset, got %d", stub.calls)
	}
}
