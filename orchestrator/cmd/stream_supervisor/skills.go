package main

import (
	"strings"
	"sync"
	"time"

	"github.com/dspy/orchestrator/internal/slurm"
	"go.uber.org/zap"
)

type trainingStarter interface {
	StartTraining(trainingStartRequest) (*slurm.SlurmJob, error)
}

type skillStats struct {
	successes     int64
	failures      int64
	lastTriggered time.Time
}

type skillPlanner struct {
	log        *zap.Logger
	training   trainingStarter
	metrics    *supervisorMetrics
	threshold  float64
	minSamples int
	cooldown   time.Duration

	mu    sync.Mutex
	stats map[string]*skillStats
}

func newSkillPlanner(log *zap.Logger, training trainingStarter, metrics *supervisorMetrics) *skillPlanner {
	if training == nil {
		return nil
	}
	return &skillPlanner{
		log:        log.With(zap.String("component", "skill_planner")),
		training:   training,
		metrics:    metrics,
		threshold:  0.3,
		minSamples: 20,
		cooldown:   15 * time.Minute,
		stats:      make(map[string]*skillStats),
	}
}

func (sp *skillPlanner) Observe(skill string, success bool) {
	if sp == nil {
		return
	}
	skill = strings.TrimSpace(strings.ToLower(skill))
	if skill == "" {
		return
	}

	sp.mu.Lock()
	stat := sp.stats[skill]
	if stat == nil {
		stat = &skillStats{}
		sp.stats[skill] = stat
	}
	if success {
		stat.successes++
	} else {
		stat.failures++
	}
	samples := stat.successes + stat.failures
	failureRate := 0.0
	if samples > 0 {
		failureRate = float64(stat.failures) / float64(samples)
	}
	shouldTrigger := samples >= int64(sp.minSamples) && failureRate >= sp.threshold && time.Since(stat.lastTriggered) >= sp.cooldown
	sp.mu.Unlock()

	if shouldTrigger {
		sp.triggerTraining(skill)
	}
}

func (sp *skillPlanner) triggerTraining(skill string) {
	req := trainingStartRequest{
		Skill:          skill,
		Steps:          1200,
		NEnvs:          4,
		LearningRate:   5e-4,
		EntropyCoeff:   0.02,
		ReplayCapacity: 4096,
		ReplayBatch:    256,
		GradClip:       1.0,
		LogInterval:    20,
	}
	if strings.Contains(skill, "go") {
		req.GEPAModules = []string{"code", "context"}
	}
	if _, err := sp.training.StartTraining(req); err != nil {
		sp.log.Warn("skill-trigger training failed", zap.String("skill", skill), zap.Error(err))
		return
	}
	sp.mu.Lock()
	if stat := sp.stats[skill]; stat != nil {
		stat.lastTriggered = time.Now()
		stat.successes = 0
		stat.failures = 0
	}
	sp.mu.Unlock()
	sp.log.Info("triggered skill training", zap.String("skill", skill))
}
