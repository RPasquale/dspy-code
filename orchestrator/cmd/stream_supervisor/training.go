package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/segmentio/kafka-go"
	"go.uber.org/zap"

	"github.com/dspy/orchestrator/internal/slurm"
)

type trainingConfig struct {
	topic          string
	group          string
	bufferDir      string
	workspace      string
	meshEndpoint   string
	meshNode       uint64
	meshDomain     string
	supervisorAddr string
	maxBuffer      int
	throttleSize   int
	slurmQueueDir  string
}

type trainingCoordinator struct {
	log     *zap.Logger
	reader  *kafka.Reader
	metrics *supervisorMetrics
	cfg     trainingConfig
	slurm   *slurm.SlurmBridge

	mu          sync.Mutex
	buffer      []trainingSample
	datasetSeq  uint64
	rewardSum   int64 // scaled by 1e6
	rewardCount atomic.Int64
	throttle    atomic.Bool
	activeJobs  atomic.Int64

	stopOnce sync.Once
	stopCh   chan struct{}
	running  sync.WaitGroup
}

type trainingSample struct {
	Raw     json.RawMessage
	Reward  float64
	Tenant  string
	TraceID string
	Skill   string
}

type trainingStartRequest struct {
	Steps           int      `json:"steps,omitempty"`
	NEnvs           int      `json:"n_envs,omitempty"`
	LearningRate    float64  `json:"lr,omitempty"`
	EntropyCoeff    float64  `json:"entropy,omitempty"`
	ReplayCapacity  int      `json:"replay_capacity,omitempty"`
	ReplayBatch     int      `json:"replay_batch,omitempty"`
	GradClip        float64  `json:"grad_clip,omitempty"`
	SkipGEPA        bool     `json:"skip_gepa,omitempty"`
	GEPAModules     []string `json:"gepa_modules,omitempty"`
	CheckpointDir   string   `json:"checkpoint_dir,omitempty"`
	CheckpointEvery int      `json:"checkpoint_interval,omitempty"`
	EarlyStop       int      `json:"early_stop,omitempty"`
	LogInterval     int      `json:"log_interval,omitempty"`
	Nodes           int      `json:"nodes,omitempty"`
	GPUs            int      `json:"gpus,omitempty"`
	MemoryGB        int      `json:"memory_gb,omitempty"`
	TimeLimit       string   `json:"time_limit,omitempty"`
	LogJSONL        string   `json:"log_jsonl,omitempty"`
	Skill           string   `json:"skill,omitempty"`
}

func newTrainingCoordinator(log *zap.Logger, reader *kafka.Reader, metrics *supervisorMetrics, cfg trainingConfig, bridge *slurm.SlurmBridge) *trainingCoordinator {
	if reader == nil || cfg.topic == "" || bridge == nil {
		return nil
	}

	if cfg.maxBuffer <= 0 {
		cfg.maxBuffer = 500
	}
	if cfg.throttleSize <= 0 {
		cfg.throttleSize = cfg.maxBuffer * 2
	}

	if err := os.MkdirAll(cfg.bufferDir, 0o755); err != nil {
		log.Warn("failed to create rl buffer dir", zap.String("dir", cfg.bufferDir), zap.Error(err))
	}

	return &trainingCoordinator{
		log:     log.With(zap.String("component", "training")),
		reader:  reader,
		metrics: metrics,
		cfg:     cfg,
		slurm:   bridge,
		buffer:  make([]trainingSample, 0, cfg.maxBuffer),
		stopCh:  make(chan struct{}),
	}
}

func (tc *trainingCoordinator) Start(ctx context.Context) {
	if tc == nil {
		return
	}
	tc.running.Add(1)
	go tc.consumeLoop(ctx)
}

func (tc *trainingCoordinator) consumeLoop(ctx context.Context) {
	defer tc.running.Done()

	for {
		msg, err := tc.reader.FetchMessage(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			tc.log.Warn("rl buffer fetch failed", zap.Error(err))
			time.Sleep(2 * time.Second)
			continue
		}
		tc.handleMessage(msg)
		if err := tc.reader.CommitMessages(ctx, msg); err != nil {
			tc.log.Debug("commit rl offset failed", zap.Error(err))
		}
	}
}

func (tc *trainingCoordinator) handleMessage(msg kafka.Message) {
	sample := trainingSample{Raw: json.RawMessage(append([]byte(nil), msg.Value...))}

	var payload map[string]interface{}
	if err := json.Unmarshal(msg.Value, &payload); err == nil {
		sample.Reward = getFloat(payload, "reward", 0)
		if tenant, ok := payload["tenant"].(string); ok {
			sample.Tenant = tenant
		}
		if trace, ok := payload["trace_id"].(string); ok {
			sample.TraceID = trace
		}
		if skill, ok := payload["skill"].(string); ok {
			sample.Skill = skill
		}
	}
	if sample.Skill == "" {
		for _, header := range msg.Headers {
			if strings.EqualFold(header.Key, "skill") {
				sample.Skill = strings.ToLower(string(header.Value))
				break
			}
		}
	}

	atomic.AddInt64(&tc.rewardSum, int64(sample.Reward*1_000_000))
	tc.rewardCount.Add(1)

	tc.mu.Lock()
	tc.buffer = append(tc.buffer, sample)
	length := len(tc.buffer)
	if length >= tc.cfg.maxBuffer {
		if _, err := tc.flushLocked(); err != nil {
			tc.log.Warn("flush rl buffer failed", zap.Error(err))
		}
	}
	shouldThrottle := length >= tc.cfg.throttleSize
	tc.mu.Unlock()

	tc.updateMetrics(length)
	tc.throttle.Store(shouldThrottle)
}

func (tc *trainingCoordinator) flushLocked() (string, error) {
	if len(tc.buffer) == 0 {
		return "", fmt.Errorf("buffer empty")
	}

	filename := fmt.Sprintf("rl_buffer_%d.jsonl", time.Now().UnixNano())
	path := filepath.Join(tc.cfg.bufferDir, filename)
	file, err := os.Create(path)
	if err != nil {
		return "", err
	}
	writer := bufio.NewWriter(file)
	for _, sample := range tc.buffer {
		if len(sample.Raw) == 0 {
			continue
		}
		if _, err := writer.Write(sample.Raw); err != nil {
			file.Close()
			return "", err
		}
		if err := writer.WriteByte('\n'); err != nil {
			file.Close()
			return "", err
		}
	}
	if err := writer.Flush(); err != nil {
		file.Close()
		return "", err
	}
	if err := file.Close(); err != nil {
		return "", err
	}
	tc.buffer = tc.buffer[:0]
	return path, nil
}

func (tc *trainingCoordinator) Stop() {
	if tc == nil {
		return
	}
	tc.stopOnce.Do(func() {
		close(tc.stopCh)
		tc.reader.Close()
	})
	tc.running.Wait()
}

func (tc *trainingCoordinator) ShouldThrottle() bool {
	if tc == nil {
		return false
	}
	return tc.throttle.Load()
}

func (tc *trainingCoordinator) updateMetrics(bufferLen int) {
	if tc.metrics == nil {
		return
	}
	count := tc.rewardCount.Load()
	sum := atomic.LoadInt64(&tc.rewardSum)
	avg := 0.0
	if count > 0 {
		avg = float64(sum) / float64(count) / 1_000_000
	}
	tc.metrics.observeTrainingQueue(bufferLen)
	tc.metrics.observeTrainingReward(avg)
	tc.metrics.observeTrainingJobs(int(tc.activeJobs.Load()))
}

func (tc *trainingCoordinator) StartTraining(req trainingStartRequest) (*slurm.SlurmJob, error) {
	if tc == nil {
		return nil, fmt.Errorf("training coordinator disabled")
	}

	tc.mu.Lock()
	datasetPath, err := tc.flushLocked()
	if err != nil {
		tc.mu.Unlock()
		return nil, fmt.Errorf("flush buffer: %w", err)
	}
	tc.mu.Unlock()

	tc.updateMetrics(0)
	tc.throttle.Store(false)

	overrides := tc.buildPayload(req, datasetPath)
	taskID := fmt.Sprintf("puffer-rl-%d", time.Now().UnixNano())

	job, err := tc.slurm.SubmitGPUJob(context.Background(), taskID, overrides)
	if err != nil {
		return nil, err
	}
	active := tc.activeJobs.Add(1)
	tc.metrics.observeTrainingJobs(int(active))
	tc.running.Add(1)
	go tc.watchJob(taskID)
	return job, nil
}

func (tc *trainingCoordinator) buildPayload(req trainingStartRequest, dataset string) map[string]interface{} {
	workspace := tc.cfg.workspace
	if workspace == "" {
		workspace = tc.cfg.bufferDir
	}
	payload := map[string]interface{}{
		"method":              "puffer_rl",
		"dataset_path":        dataset,
		"workspace_dir":       workspace,
		"skill":               req.Skill,
		"log_dir":             tc.cfg.bufferDir,
		"mesh_endpoint":       tc.cfg.meshEndpoint,
		"mesh_node_id":        int(tc.cfg.meshNode),
		"mesh_domain":         tc.cfg.meshDomain,
		"supervisor_addr":     tc.cfg.supervisorAddr,
		"steps":               defaultInt(req.Steps, tc.cfg.maxBuffer*2),
		"n_envs":              defaultInt(req.NEnvs, 4),
		"lr":                  defaultFloat(req.LearningRate, 1e-3),
		"entropy":             defaultFloat(req.EntropyCoeff, 0.01),
		"replay_capacity":     defaultInt(req.ReplayCapacity, 4096),
		"replay_batch":        defaultInt(req.ReplayBatch, 256),
		"grad_clip":           defaultFloat(req.GradClip, 1.0),
		"skip_gepa":           boolToInt(req.SkipGEPA),
		"gepa_modules":        strings.Join(req.GEPAModules, " "),
		"checkpoint_dir":      chooseString(req.CheckpointDir, filepath.Join(tc.cfg.bufferDir, "rl_checkpoints")),
		"checkpoint_interval": defaultInt(req.CheckpointEvery, 0),
		"early_stop":          defaultInt(req.EarlyStop, 0),
		"log_interval":        defaultInt(req.LogInterval, 10),
		"log_jsonl":           chooseString(req.LogJSONL, filepath.Join(tc.cfg.bufferDir, fmt.Sprintf("rl_%d.jsonl", time.Now().UnixNano()))),
		"nodes":               defaultInt(req.Nodes, 1),
		"gpus":                defaultInt(req.GPUs, 1),
		"memory_gb":           defaultInt(req.MemoryGB, 48),
		"time_limit":          chooseString(req.TimeLimit, "04:00:00"),
	}
	return payload
}

func defaultInt(v, fallback int) int {
	if v > 0 {
		return v
	}
	return fallback
}

func defaultFloat(v, fallback float64) float64 {
	if v > 0 {
		return v
	}
	return fallback
}

func chooseString(val, fallback string) string {
	if strings.TrimSpace(val) != "" {
		return val
	}
	return fallback
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func getFloat(m map[string]interface{}, key string, fallback float64) float64 {
	if val, ok := m[key]; ok {
		switch v := val.(type) {
		case float64:
			return v
		case float32:
			return float64(v)
		case int:
			return float64(v)
		case int32:
			return float64(v)
		case int64:
			return float64(v)
		case string:
			if parsed, err := strconv.ParseFloat(v, 64); err == nil {
				return parsed
			}
		}
	}
	return fallback
}

func (tc *trainingCoordinator) Status(taskID string) (*slurm.SlurmJob, bool) {
	if tc == nil {
		return nil, false
	}
	return tc.slurm.GetJobStatus(taskID)
}

func (tc *trainingCoordinator) watchJob(taskID string) {
	defer tc.running.Done()
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-tc.stopCh:
			return
		case <-ticker.C:
			if job, ok := tc.slurm.GetJobStatus(taskID); ok {
				state := strings.ToLower(job.Status)
				if state == "completed" || state == "failed" {
					remaining := tc.activeJobs.Add(-1)
					tc.metrics.observeTrainingJobs(int(remaining))
					return
				}
			} else {
				remaining := tc.activeJobs.Add(-1)
				tc.metrics.observeTrainingJobs(int(remaining))
				return
			}
		}
	}
}
