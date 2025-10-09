package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/segmentio/kafka-go"
	"go.uber.org/zap"
	"google.golang.org/grpc"

	"github.com/dspy/orchestrator/internal/meshclient"
	pb "github.com/dspy/orchestrator/internal/pb"
	"github.com/dspy/orchestrator/internal/slurm"
	"github.com/dspy/orchestrator/pkg/telemetry"
)

type config struct {
	listenAddr     string
	kafkaBrokers   []string
	inputTopic     string
	groupID        string
	meshEndpoint   string
	meshSourceNode uint64
	meshDomain     string
	meshServices   []meshServiceConf
	metricsListen  string
	tenantDomains  map[string]string
	rlTopic        string
	rlGroup        string
	rlBufferDir    string
	rlWorkspace    string
	rlMaxBuffer    int
	slurmQueueDir  string
	meshSupervisor string
}

type meshServiceConf struct {
	ID       uint64   `json:"id"`
	Endpoint string   `json:"endpoint"`
	Domain   string   `json:"domain"`
	Tags     []string `json:"tags"`
	RttMs    float64  `json:"rtt_ms"`
	Weight   float64  `json:"weight"`
}

type workerSession struct {
	id          string
	assignCh    chan *pb.TaskAssignment
	credits     int
	maxInflight int
	inflight    map[string]*inflightRecord
	meshEnabled bool
	meshNodeID  uint64
	meshDomain  string
	stats       workerStats
	linkPenalty float64
}

type supervisorServer struct {
	pb.UnimplementedStreamSupervisorServer

	log               *zap.Logger
	reader            *kafka.Reader
	meshPublishers    map[uint64]*meshclient.Publisher
	meshDefault       *meshclient.Publisher
	meshServices      map[uint64]meshServiceConf
	defaultMeshDomain string
	meshQueue         chan meshJob
	meshWorkers       sync.WaitGroup
	metrics           *supervisorMetrics
	tenantDomains     map[string]string
	training          *trainingCoordinator
	skills            *skillPlanner

	mu      sync.Mutex
	workers map[string]*workerSession
	pending []*kafka.Message
}

type meshJob struct {
	workerID   string
	assignment *pb.TaskAssignment
	message    *kafka.Message
	attempts   int
	workflow   workflowMeta
}

const (
	meshDispatchMaxAttempts = 5
	meshDispatchBaseDelay   = 100 * time.Millisecond
)

type inflightRecord struct {
	message    *kafka.Message
	assignedAt time.Time
	attempts   int
	skill      string
	workflow   workflowMeta
}

type workflowMeta struct {
	id     string
	runID  string
	tenant string
}

type workerStats struct {
	emaLatencyMs float64
	lastLatency  float64
	successTotal uint64
	failureTotal uint64
	lastAck      time.Time
}

func (ws *workerStats) observe(now time.Time, latency time.Duration, success bool) {
	ms := float64(latency) / float64(time.Millisecond)
	if ms < 0 {
		ms = 0
	}
	if ws.emaLatencyMs == 0 {
		ws.emaLatencyMs = ms
	} else {
		const smoothing = 0.25
		ws.emaLatencyMs = smoothing*ms + (1-smoothing)*ws.emaLatencyMs
	}
	ws.lastLatency = ms
	ws.lastAck = now
	if success {
		ws.successTotal++
	} else {
		ws.failureTotal++
	}
}

func (ws *workerStats) latencyEstimate() float64 {
	if ws.emaLatencyMs == 0 {
		return 1.0
	}
	return ws.emaLatencyMs
}

func (ws *workerStats) stalePenalty(now time.Time) float64 {
	if ws.lastAck.IsZero() {
		return 1.0
	}
	stale := now.Sub(ws.lastAck)
	if stale <= 5*time.Second {
		return 1.0
	}
	return 1.0 + float64(stale/time.Second)/10.0
}

func linkPenaltyForService(svc meshServiceConf) float64 {
	penalty := 1.0
	if svc.RttMs > 0 {
		penalty += svc.RttMs / 50.0
	}
	if svc.Weight > 0 {
		penalty = penalty / svc.Weight
	}
	if penalty < 0.1 {
		penalty = 0.1
	}
	return penalty
}

type supervisorMetrics struct {
	registry               *telemetry.Registry
	pendingGauge           *telemetry.Gauge
	meshQueueGauge         *telemetry.Gauge
	latencySum             *telemetry.CounterVec
	latencySamples         *telemetry.CounterVec
	workerFailures         *telemetry.CounterVec
	meshDispatchTotals     *telemetry.CounterVec
	trainingQueueGauge     *telemetry.Gauge
	trainingRewardGauge    *telemetry.Gauge
	trainingJobsGauge      *telemetry.Gauge
	skillTotals            *telemetry.CounterVec
	workflowTotals         *telemetry.CounterVec
	workflowLatencySum     *telemetry.CounterVec
	workflowLatencySamples *telemetry.CounterVec
}

func newSupervisorMetrics(reg *telemetry.Registry) *supervisorMetrics {
	return &supervisorMetrics{
		registry:               reg,
		pendingGauge:           telemetry.NewGauge(reg, "supervisor_pending_queue_depth", "Number of assignments waiting for dispatch."),
		meshQueueGauge:         telemetry.NewGauge(reg, "supervisor_mesh_queue_depth", "Number of assignments queued for mesh transport."),
		latencySum:             telemetry.NewCounterVec(reg, "supervisor_worker_latency_seconds_total", "Cumulative observed worker ack latency in seconds.", []string{"worker"}),
		latencySamples:         telemetry.NewCounterVec(reg, "supervisor_worker_latency_samples_total", "Count of worker ack latency samples observed.", []string{"worker"}),
		workerFailures:         telemetry.NewCounterVec(reg, "supervisor_worker_task_failures_total", "Number of task failures reported by workers.", []string{"worker"}),
		meshDispatchTotals:     telemetry.NewCounterVec(reg, "supervisor_mesh_dispatch_total", "Mesh dispatch attempts grouped by status.", []string{"worker", "status"}),
		trainingQueueGauge:     telemetry.NewGauge(reg, "supervisor_training_queue_depth", "Buffered RL transitions awaiting training."),
		trainingRewardGauge:    telemetry.NewGauge(reg, "supervisor_training_reward_mean", "Average reward observed in RL buffer."),
		trainingJobsGauge:      telemetry.NewGauge(reg, "supervisor_training_jobs_inflight", "Number of active RL training jobs."),
		skillTotals:            telemetry.NewCounterVec(reg, "supervisor_skill_events_total", "Success/failure counts grouped by skill.", []string{"skill", "outcome"}),
		workflowTotals:         telemetry.NewCounterVec(reg, "supervisor_workflow_events_total", "Workflow success/failure counts.", []string{"workflow", "outcome"}),
		workflowLatencySum:     telemetry.NewCounterVec(reg, "supervisor_workflow_latency_seconds_total", "Workflow latency accumulation per workflow.", []string{"workflow"}),
		workflowLatencySamples: telemetry.NewCounterVec(reg, "supervisor_workflow_latency_samples_total", "Workflow latency samples.", []string{"workflow"}),
	}
}

func (m *supervisorMetrics) observePending(depth int) {
	if m == nil {
		return
	}
	m.pendingGauge.Set(float64(depth))
}

func (m *supervisorMetrics) observeMeshQueue(depth int) {
	if m == nil {
		return
	}
	m.meshQueueGauge.Set(float64(depth))
}

func (m *supervisorMetrics) observeLatency(worker string, latency time.Duration, success bool) {
	if m == nil {
		return
	}
	if worker == "" {
		worker = "unknown"
	}
	m.latencySamples.WithLabelValues(worker).Inc()
	if latency > 0 {
		m.latencySum.WithLabelValues(worker).Add(latency.Seconds())
	}
	if !success {
		m.workerFailures.WithLabelValues(worker).Inc()
	}
}

func (m *supervisorMetrics) observeMeshDispatch(worker, status string) {
	if m == nil {
		return
	}
	if worker == "" {
		worker = "unknown"
	}
	if status == "" {
		status = "unknown"
	}
	m.meshDispatchTotals.WithLabelValues(worker, status).Inc()
}

func (m *supervisorMetrics) observeTrainingQueue(depth int) {
	if m == nil {
		return
	}
	m.trainingQueueGauge.Set(float64(depth))
}

func (m *supervisorMetrics) observeTrainingReward(avg float64) {
	if m == nil {
		return
	}
	m.trainingRewardGauge.Set(avg)
}

func (m *supervisorMetrics) observeTrainingJobs(count int) {
	if m == nil {
		return
	}
	m.trainingJobsGauge.Set(float64(count))
}

func (m *supervisorMetrics) observeSkill(skill string, success bool) {
	if m == nil || skill == "" {
		return
	}
	outcome := "success"
	if !success {
		outcome = "failure"
	}
	m.skillTotals.WithLabelValues(skill, outcome).Inc()
}

func (m *supervisorMetrics) observeWorkflow(workflow string, success bool) {
	if m == nil || workflow == "" {
		return
	}
	outcome := "success"
	if !success {
		outcome = "failure"
	}
	m.workflowTotals.WithLabelValues(workflow, outcome).Inc()
}

func (m *supervisorMetrics) observeWorkflowLatency(workflow string, latency time.Duration) {
	if m == nil || workflow == "" {
		return
	}
	m.workflowLatencySum.WithLabelValues(workflow).Add(latency.Seconds())
	m.workflowLatencySamples.WithLabelValues(workflow).Inc()
}

func cloneMessage(msg *kafka.Message) *kafka.Message {
	if msg == nil {
		return nil
	}
	copyMsg := *msg
	return &copyMsg
}

func extractSkill(msg *kafka.Message, meta workflowMeta) string {
	if msg == nil {
		return ""
	}
	for _, header := range msg.Headers {
		key := strings.ToLower(header.Key)
		if key == "skill" || key == "language" || key == "domain" {
			return strings.ToLower(string(header.Value))
		}
	}
	if meta.id != "" {
		return fmt.Sprintf("workflow::%s", strings.ToLower(meta.id))
	}
	return ""
}

func fallbackSkill(assignment *pb.TaskAssignment, meta workflowMeta) string {
	if assignment == nil {
		return ""
	}
	if tenant := strings.TrimSpace(assignment.GetTenant()); tenant != "" {
		return strings.ToLower(tenant)
	}
	if topic := strings.TrimSpace(assignment.GetTopic()); topic != "" {
		return strings.ToLower(topic)
	}
	if meta.id != "" {
		return fmt.Sprintf("workflow::%s", strings.ToLower(meta.id))
	}
	return ""
}

func (w *workerSession) trackAssignment(taskID string, msg *kafka.Message, skill string, meta workflowMeta) {
	if taskID == "" || msg == nil {
		return
	}
	if existing, ok := w.inflight[taskID]; ok {
		existing.message = msg
		existing.assignedAt = time.Now()
		existing.attempts++
		existing.skill = skill
		existing.workflow = meta
		return
	}
	w.inflight[taskID] = &inflightRecord{
		message:    msg,
		assignedAt: time.Now(),
		attempts:   1,
		skill:      skill,
		workflow:   meta,
	}
}

func (w *workerSession) score(now time.Time) float64 {
	credits := float64(w.credits)
	if credits < 0 {
		credits = 0
	}
	headroom := credits + 1.0
	outstanding := float64(len(w.inflight))
	maxInflight := float64(w.maxInflight)
	if maxInflight <= 0 {
		maxInflight = 1
	}
	concurrencyPenalty := 1.0 + outstanding/maxInflight
	latencyFactor := 1.0 + w.stats.latencyEstimate()/75.0
	if latencyFactor < 1.0 {
		latencyFactor = 1.0
	}
	linkPenalty := w.linkPenalty
	if linkPenalty <= 0 {
		linkPenalty = 1.0
	}
	return headroom / (concurrencyPenalty * latencyFactor * w.stats.stalePenalty(now) * linkPenalty)
}

func newSupervisorServer(logger *zap.Logger, reader *kafka.Reader, publishers map[uint64]*meshclient.Publisher, services map[uint64]meshServiceConf, defaultPublisher *meshclient.Publisher, defaultDomain string, metrics *supervisorMetrics, tenantDomains map[string]string, training *trainingCoordinator, skills *skillPlanner) *supervisorServer {
	srv := &supervisorServer{
		log:               logger,
		reader:            reader,
		meshPublishers:    publishers,
		meshDefault:       defaultPublisher,
		meshServices:      services,
		defaultMeshDomain: defaultDomain,
		workers:           make(map[string]*workerSession),
		pending:           make([]*kafka.Message, 0, 128),
		metrics:           metrics,
		tenantDomains:     tenantDomains,
		training:          training,
		skills:            skills,
	}
	if len(publishers) > 0 || defaultPublisher != nil {
		srv.meshQueue = make(chan meshJob, 1024)
		srv.startMeshWorkers()
	}
	if metrics != nil {
		metrics.observePending(0)
		metrics.observeMeshQueue(0)
	}
	return srv
}

func (s *supervisorServer) OpenStream(stream pb.StreamSupervisor_OpenStreamServer) error {
	ctx := stream.Context()
	var session *workerSession

	for {
		incoming, err := stream.Recv()
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if session != nil {
				s.log.Warn("worker stream closed", zap.String("worker", session.id), zap.Error(err))
			}
			break
		}
		if incoming == nil || incoming.Msg == nil {
			continue
		}

		switch msg := incoming.Msg.(type) {
		case *pb.WorkerToSupervisor_Hello:
			if session != nil {
				s.log.Warn("duplicate hello", zap.String("worker", session.id))
				continue
			}
			if msg.Hello == nil {
				continue
			}
			session = s.registerWorker(ctx, stream, msg.Hello)
		case *pb.WorkerToSupervisor_Credit:
			if session == nil || msg.Credit == nil {
				continue
			}
			s.handleCredit(session, int(msg.Credit.Credits))
		case *pb.WorkerToSupervisor_Ack:
			if session == nil || msg.Ack == nil {
				continue
			}
			s.handleAck(ctx, session, msg.Ack)
		case *pb.WorkerToSupervisor_Heartbeat:
			if session != nil {
				s.log.Debug("heartbeat", zap.String("worker", session.id))
			}
		}
	}

	if session != nil {
		s.removeWorker(session.id)
	}
	return nil
}

func (s *supervisorServer) registerWorker(ctx context.Context, stream pb.StreamSupervisor_OpenStreamServer, hello *pb.WorkerHello) *workerSession {
	workerID := strings.TrimSpace(hello.WorkerId)
	if workerID == "" {
		workerID = uuid.NewString()
	}
	buffer := int(hello.MaxInflight)
	if buffer <= 0 {
		buffer = 1
	}

	session := &workerSession{
		id:          workerID,
		assignCh:    make(chan *pb.TaskAssignment, buffer),
		credits:     int(hello.MaxInflight),
		maxInflight: int(hello.MaxInflight),
		inflight:    make(map[string]*inflightRecord),
		linkPenalty: 1.0,
	}

	session.meshNodeID = hello.GetMeshNodeId()
	session.meshDomain = hello.GetMeshDomain()
	if svc, ok := s.meshServices[session.meshNodeID]; ok {
		if session.meshDomain == "" && svc.Domain != "" {
			session.meshDomain = svc.Domain
		}
		session.linkPenalty = linkPenaltyForService(svc)
	}
	publisher := s.meshPublisherForNode(session.meshNodeID)
	session.meshEnabled = hello.GetMeshEnabled() && publisher != nil
	if hello.GetMeshEnabled() && publisher == nil {
		s.log.Warn("mesh requested but publisher unavailable", zap.String("worker", workerID), zap.Uint64("mesh_node", session.meshNodeID))
	}

	s.mu.Lock()
	s.workers[workerID] = session
	s.dispatchLocked()
	s.mu.Unlock()

	s.log.Info(
		"worker registered",
		zap.String("worker", workerID),
		zap.Int("max_inflight", session.maxInflight),
		zap.Bool("mesh", session.meshEnabled),
		zap.Uint64("mesh_node", session.meshNodeID),
	)

	if !session.meshEnabled {
		go func() {
			for {
				select {
				case assignment, ok := <-session.assignCh:
					if !ok {
						return
					}
					wrapper := &pb.SupervisorToWorker{
						Msg: &pb.SupervisorToWorker_Assignment{Assignment: assignment},
					}
					if err := stream.Send(wrapper); err != nil {
						s.log.Warn("send failed", zap.String("worker", workerID), zap.Error(err))
						s.removeWorker(workerID)
						return
					}
				case <-ctx.Done():
					s.removeWorker(workerID)
					return
				}
			}
		}()
	}

	return session
}

func (s *supervisorServer) handleCredit(session *workerSession, credits int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	session.credits = credits
	if session.credits > session.maxInflight {
		session.credits = session.maxInflight
	}
	s.dispatchLocked()
}

func (s *supervisorServer) handleAck(ctx context.Context, session *workerSession, ack *pb.TaskAck) {
	var message *kafka.Message
	var latency time.Duration
	skillTag := ""
	now := time.Now()

	s.mu.Lock()
	var workflow workflowMeta
	if record, ok := session.inflight[ack.TaskId]; ok {
		message = record.message
		if !record.assignedAt.IsZero() {
			latency = now.Sub(record.assignedAt)
		}
		delete(session.inflight, ack.TaskId)
		session.stats.observe(now, latency, ack.Success)
		skillTag = record.skill
		workflow = record.workflow
	}
	if !ack.Success && message != nil {
		if msgCopy := cloneMessage(message); msgCopy != nil {
			s.pending = append([]*kafka.Message{msgCopy}, s.pending...)
		}
	}
	s.dispatchLocked()
	s.mu.Unlock()

	s.metrics.observeLatency(session.id, latency, ack.Success)
	identifier := workflow.id
	if identifier == "" {
		identifier = workflow.runID
	}
	if identifier == "" {
		identifier = workflow.tenant
	}
	if identifier != "" {
		s.metrics.observeWorkflow(identifier, ack.Success)
		if latency > 0 {
			s.metrics.observeWorkflowLatency(identifier, latency)
		}
	}
	if skillTag != "" {
		s.metrics.observeSkill(skillTag, ack.Success)
		if s.skills != nil {
			s.skills.Observe(skillTag, ack.Success)
		}
	}

	if message == nil {
		return
	}

	if latency > 0 {
		s.log.Debug(
			"task ack",
			zap.String("worker", session.id),
			zap.String("task_id", ack.TaskId),
			zap.Duration("latency", latency),
			zap.Bool("success", ack.Success),
		)
	}

	if ack.Success {
		commitCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if err := s.reader.CommitMessages(commitCtx, *message); err != nil {
			s.log.Warn("commit failed", zap.Error(err))
		}
	} else {
		s.log.Warn("task failed", zap.String("worker", session.id), zap.String("task_id", ack.TaskId), zap.String("error", ack.Error))
	}
}

func (s *supervisorServer) enqueue(msg kafka.Message) {
	copyMsg := msg
	s.mu.Lock()
	s.pending = append(s.pending, &copyMsg)
	s.dispatchLocked()
	s.metrics.observePending(len(s.pending))
	s.mu.Unlock()
}

func (s *supervisorServer) dispatchLocked() {
	if len(s.pending) == 0 {
		return
	}

	requeue := make([]*kafka.Message, 0)

	for len(s.pending) > 0 {
		worker := s.nextWorkerLocked()
		if worker == nil || worker.credits <= 0 {
			break
		}

		msg := s.pending[0]
		s.pending = s.pending[1:]
		assignment, wfMeta := buildAssignment(msg)
		skillTag := extractSkill(msg, wfMeta)
		if skillTag == "" {
			skillTag = fallbackSkill(assignment, wfMeta)
		}

		if s.training != nil && s.training.ShouldThrottle() && strings.EqualFold(assignment.GetTenant(), "training") {
			requeue = append(requeue, msg)
			continue
		}

		if worker.meshEnabled {
			msgCopy := cloneMessage(msg)
			job := meshJob{
				workerID:   worker.id,
				assignment: assignment,
				message:    msgCopy,
				attempts:   0,
				workflow:   wfMeta,
			}
			if s.enqueueMeshJob(job) {
				worker.trackAssignment(assignment.TaskId, msgCopy, skillTag, wfMeta)
				worker.credits--
				continue
			}
			requeue = append(requeue, msg)
			continue
		}

		select {
		case worker.assignCh <- assignment:
			worker.credits--
			worker.trackAssignment(assignment.TaskId, cloneMessage(msg), skillTag, wfMeta)
		default:
			requeue = append(requeue, msg)
		}
	}

	if len(requeue) > 0 {
		s.pending = append(requeue, s.pending...)
	}

	s.metrics.observePending(len(s.pending))
}

func (s *supervisorServer) sendViaMesh(ctx context.Context, worker *workerSession, assignment *pb.TaskAssignment) error {
	publisher := s.meshPublisherForNode(worker.meshNodeID)
	if publisher == nil {
		return fmt.Errorf("no mesh publisher for node %d", worker.meshNodeID)
	}
	if worker.meshNodeID == 0 {
		return fmt.Errorf("worker %s missing mesh node id", worker.id)
	}
	domain := worker.meshDomain
	if domain == "" {
		domain = s.defaultMeshDomain
	}
	if tenant := strings.TrimSpace(assignment.GetTenant()); tenant != "" {
		if override, ok := s.tenantDomains[strings.ToLower(tenant)]; ok {
			domain = override
		}
	}
	_, err := publisher.SendAssignment(ctx, worker.meshNodeID, domain, assignment)
	return err
}

func (s *supervisorServer) meshPublisherForNode(node uint64) *meshclient.Publisher {
	if pub, ok := s.meshPublishers[node]; ok {
		return pub
	}
	return s.meshDefault
}

func (s *supervisorServer) enqueueMeshJob(job meshJob) bool {
	if s.meshQueue == nil {
		return false
	}
	select {
	case s.meshQueue <- job:
		s.metrics.observeMeshQueue(len(s.meshQueue))
		return true
	default:
		s.metrics.observeMeshQueue(len(s.meshQueue))
		s.log.Debug(
			"mesh queue saturated",
			zap.String("worker", job.workerID),
			zap.Int("depth", len(s.meshQueue)),
		)
		return false
	}
}

func (s *supervisorServer) startMeshWorkers() {
	if s.meshQueue == nil {
		return
	}
	workerCount := 4
	s.meshWorkers.Add(workerCount)
	for i := 0; i < workerCount; i++ {
		go func() {
			defer s.meshWorkers.Done()
			for job := range s.meshQueue {
				s.metrics.observeMeshQueue(len(s.meshQueue))
				s.processMeshJob(job)
			}
		}()
	}
}

func (s *supervisorServer) stopMeshWorkers() {
	if s.meshQueue == nil {
		return
	}
	close(s.meshQueue)
	s.meshWorkers.Wait()
}

func (s *supervisorServer) processMeshJob(job meshJob) {
	s.mu.Lock()
	session, ok := s.workers[job.workerID]
	if !ok {
		if msgCopy := cloneMessage(job.message); msgCopy != nil {
			s.pending = append([]*kafka.Message{msgCopy}, s.pending...)
		}
		s.dispatchLocked()
		s.mu.Unlock()
		s.metrics.observeMeshDispatch(job.workerID, "missing")
		s.log.Warn("mesh dispatch aborted: worker missing", zap.String("worker", job.workerID))
		return
	}
	s.mu.Unlock()

	attempts := job.attempts
	var lastErr error
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		err := s.sendViaMesh(ctx, session, job.assignment)
		cancel()
		if err == nil {
			s.metrics.observeMeshDispatch(job.workerID, "success")
			if attempts > 0 {
				s.log.Info("mesh dispatch recovered", zap.String("worker", job.workerID), zap.Int("attempts", attempts))
			}
			return
		}
		lastErr = err
		attempts++
		if attempts >= meshDispatchMaxAttempts {
			s.metrics.observeMeshDispatch(job.workerID, "failure")
			break
		}
		s.metrics.observeMeshDispatch(job.workerID, "retry")
		backoff := meshDispatchBaseDelay * time.Duration(1<<uint(attempts-1))
		if backoff > 3*time.Second {
			backoff = 3 * time.Second
		}
		s.log.Warn("mesh dispatch retry", zap.String("worker", job.workerID), zap.Int("attempt", attempts), zap.Duration("backoff", backoff), zap.Error(err))
		timer := time.NewTimer(backoff)
		<-timer.C
		if !timer.Stop() {
			// timer already fired; drain implicitly
		}
	}

	s.mu.Lock()
	if current, ok := s.workers[job.workerID]; ok {
		current.credits++
		delete(current.inflight, job.assignment.TaskId)
		if msgCopy := cloneMessage(job.message); msgCopy != nil {
			s.pending = append([]*kafka.Message{msgCopy}, s.pending...)
		}
		s.dispatchLocked()
	}
	s.mu.Unlock()

	if lastErr != nil {
		s.log.Warn("mesh dispatch failed", zap.String("worker", job.workerID), zap.Error(lastErr))
	} else {
		s.log.Warn("mesh dispatch failed", zap.String("worker", job.workerID))
	}
}

func buildAssignment(msg *kafka.Message) (*pb.TaskAssignment, workflowMeta) {
	meta := workflowMetaFromMessage(msg)
	tenant := meta.tenant
	if tenant == "" {
		for _, header := range msg.Headers {
			if strings.EqualFold(header.Key, "tenant") {
				tenant = strings.ToLower(string(header.Value))
				break
			}
		}
	}
	payload := make([]byte, len(msg.Value))
	copy(payload, msg.Value)

	taskID := fmt.Sprintf("%s-%d-%d", msg.Topic, msg.Partition, msg.Offset)
	assignment := &pb.TaskAssignment{
		TaskId:    taskID,
		Tenant:    tenant,
		Topic:     msg.Topic,
		Payload:   payload,
		Offset:    int64(msg.Offset),
		Partition: fmt.Sprintf("%d", msg.Partition),
	}
	return assignment, meta
}

func workflowMetaFromMessage(msg *kafka.Message) workflowMeta {
	meta := workflowMeta{}
	for _, header := range msg.Headers {
		key := strings.ToLower(header.Key)
		switch key {
		case "workflow_id":
			meta.id = string(header.Value)
		case "workflow_run_id":
			meta.runID = string(header.Value)
		case "workflow_tenant":
			meta.tenant = strings.ToLower(string(header.Value))
		case "tenant":
			if meta.tenant == "" {
				meta.tenant = strings.ToLower(string(header.Value))
			}
		}
	}
	if meta.id != "" && meta.tenant != "" && meta.runID != "" {
		return meta
	}
	var payloadMap map[string]any
	if err := json.Unmarshal(msg.Value, &payloadMap); err == nil {
		if meta.id == "" {
			if v, ok := payloadMap["workflow_id"].(string); ok {
				meta.id = v
			}
			if ctx, ok := payloadMap["workflow_context"].(map[string]any); ok {
				if id, ok := ctx["id"].(string); ok && meta.id == "" {
					meta.id = id
				}
				if tenant, ok := ctx["tenant"].(string); ok && meta.tenant == "" {
					meta.tenant = strings.ToLower(tenant)
				}
			}
		}
		if meta.runID == "" {
			if runID, ok := payloadMap["workflow_run_id"].(string); ok {
				meta.runID = runID
			}
		}
		if meta.tenant == "" {
			if t, ok := payloadMap["tenant"].(string); ok {
				meta.tenant = strings.ToLower(t)
			}
		}
	}
	return meta
}

func (s *supervisorServer) nextWorkerLocked() *workerSession {
	var best *workerSession
	bestScore := -1.0
	now := time.Now()
	for _, worker := range s.workers {
		if worker.credits <= 0 {
			continue
		}
		score := worker.score(now)
		if score > bestScore {
			best = worker
			bestScore = score
		}
	}
	return best
}

func (s *supervisorServer) removeWorker(workerID string) {
	s.mu.Lock()
	session, ok := s.workers[workerID]
	if !ok {
		s.mu.Unlock()
		return
	}
	delete(s.workers, workerID)
	close(session.assignCh)

	for _, record := range session.inflight {
		if msgCopy := cloneMessage(record.message); msgCopy != nil {
			s.pending = append([]*kafka.Message{msgCopy}, s.pending...)
		}
	}
	s.dispatchLocked()
	s.mu.Unlock()

	s.log.Info("worker removed", zap.String("worker", workerID))
}

func loadConfig() config {
	listen := envOr("SUPERVISOR_LISTEN", ":7000")
	brokers := envOr("KAFKA_BROKERS", "broker:9092")
	inputTopic := envOr("INPUT_TOPIC", "raw.events.demo")
	group := envOr("SUPERVISOR_GROUP", "stream-supervisor")
	meshEndpoint := strings.TrimSpace(envOr("MESH_PUBLISH_ENDPOINT", ""))
	meshDomain := envOr("MESH_PUBLISH_DOMAIN", "default")
	meshSourceEnv := strings.TrimSpace(envOr("MESH_SOURCE_NODE", "0"))
	meshSourceNode, _ := strconv.ParseUint(meshSourceEnv, 10, 64)
	metricsListen := envOr("SUPERVISOR_METRICS_LISTEN", ":9098")
	meshServicesFile := strings.TrimSpace(os.Getenv("MESH_SERVICES_FILE"))
	meshServicesJSON := strings.TrimSpace(os.Getenv("MESH_SERVICES_JSON"))
	tenantDomainRaw := strings.TrimSpace(os.Getenv("MESH_TENANT_DOMAIN_MAP"))
	defaultBufferDir := filepath.Join(os.TempDir(), "supervisor_rl_buffer")
	defaultSlurmQueueDir := filepath.Join(os.TempDir(), "supervisor_slurm_queue")
	rlTopic := envOr("RL_RESULTS_TOPIC", envOr("RL_TOPIC", ""))
	rlGroup := envOr("RL_RESULTS_GROUP", envOr("RL_GROUP", group+"-rl"))
	rlBufferDir := envOr("RL_BUFFER_DIR", defaultBufferDir)
	rlWorkspace := envOr("RL_WORKSPACE_DIR", envOr("RL_WORKSPACE", ""))
	rlMaxBuffer := parseIntOr(envOr("RL_MAX_BUFFER", "10000"), 10000)
	slurmQueueDir := envOr("SLURM_QUEUE_DIR", defaultSlurmQueueDir)
	meshSupervisor := envOr("MESH_SUPERVISOR_ADDR", listen)
	var meshServices []meshServiceConf
	if meshServicesJSON != "" {
		if err := json.Unmarshal([]byte(meshServicesJSON), &meshServices); err != nil {
			fmt.Fprintf(os.Stderr, "[mesh] failed to parse MESH_SERVICES_JSON: %v\n", err)
		}
	}

	flag.StringVar(&listen, "listen", listen, "gRPC listen address")
	flag.StringVar(&brokers, "brokers", brokers, "Kafka broker list")
	flag.StringVar(&inputTopic, "input", inputTopic, "Kafka topic to consume")
	flag.StringVar(&group, "group", group, "Kafka consumer group")
	flag.StringVar(&meshEndpoint, "mesh-endpoint", meshEndpoint, "Mesh Publish endpoint (optional)")
	flag.StringVar(&meshDomain, "mesh-domain", meshDomain, "Mesh domain name")
	flag.Uint64Var(&meshSourceNode, "mesh-source-node", meshSourceNode, "Mesh source node id")
	flag.StringVar(&metricsListen, "metrics-listen", metricsListen, "HTTP listen address for Prometheus metrics")
	flag.StringVar(&meshServicesFile, "mesh-services-file", meshServicesFile, "Path to mesh services manifest JSON")
	flag.StringVar(&tenantDomainRaw, "tenant-domain-map", tenantDomainRaw, "Comma separated tenant=domain overrides")
	flag.StringVar(&rlTopic, "rl-topic", rlTopic, "Kafka topic with RL transitions")
	flag.StringVar(&rlGroup, "rl-group", rlGroup, "Kafka consumer group for RL buffer")
	flag.StringVar(&rlBufferDir, "rl-buffer-dir", rlBufferDir, "Directory for RL buffer snapshots")
	flag.StringVar(&rlWorkspace, "rl-workspace", rlWorkspace, "Workspace path exposed to training jobs")
	flag.IntVar(&rlMaxBuffer, "rl-max-buffer", rlMaxBuffer, "Maximum buffered RL transitions before throttling")
	flag.StringVar(&slurmQueueDir, "slurm-queue-dir", slurmQueueDir, "Queue directory for Slurm bridge state")
	flag.StringVar(&meshSupervisor, "mesh-supervisor-addr", meshSupervisor, "Supervisor gRPC address for training jobs")
	flag.Parse()

	if meshServicesFile != "" {
		if data, err := os.ReadFile(meshServicesFile); err != nil {
			fmt.Fprintf(os.Stderr, "[mesh] failed to read mesh services file %s: %v\n", meshServicesFile, err)
		} else {
			meshServicesJSON = string(data)
			meshServices = nil
			if err := json.Unmarshal([]byte(meshServicesJSON), &meshServices); err != nil {
				fmt.Fprintf(os.Stderr, "[mesh] failed to parse mesh services file %s: %v\n", meshServicesFile, err)
			}
		}
	}

	tenantDomains := parseTenantDomains(tenantDomainRaw)

	return config{
		listenAddr:     listen,
		kafkaBrokers:   strings.Split(brokers, ","),
		inputTopic:     inputTopic,
		groupID:        group,
		meshEndpoint:   meshEndpoint,
		meshSourceNode: meshSourceNode,
		meshDomain:     meshDomain,
		meshServices:   meshServices,
		metricsListen:  metricsListen,
		tenantDomains:  tenantDomains,
		rlTopic:        rlTopic,
		rlGroup:        rlGroup,
		rlBufferDir:    rlBufferDir,
		rlWorkspace:    rlWorkspace,
		rlMaxBuffer:    rlMaxBuffer,
		slurmQueueDir:  slurmQueueDir,
		meshSupervisor: meshSupervisor,
	}
}

func envOr(key, fallback string) string {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		return val
	}
	return fallback
}

func parseIntOr(val string, fallback int) int {
	if parsed, err := strconv.Atoi(strings.TrimSpace(val)); err == nil {
		return parsed
	}
	return fallback
}

func parseTenantDomains(raw string) map[string]string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	result := make(map[string]string)
	for _, part := range strings.Split(raw, ",") {
		entry := strings.TrimSpace(part)
		if entry == "" {
			continue
		}
		pieces := strings.SplitN(entry, "=", 2)
		if len(pieces) != 2 {
			fmt.Fprintf(os.Stderr, "[mesh] skipping invalid tenant-domain mapping: %s\n", entry)
			continue
		}
		tenant := strings.ToLower(strings.TrimSpace(pieces[0]))
		domain := strings.TrimSpace(pieces[1])
		if tenant == "" || domain == "" {
			continue
		}
		result[tenant] = domain
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

func (s *supervisorServer) runKafka(ctx context.Context) {
	go func() {
		for {
			msg, err := s.reader.FetchMessage(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				s.log.Error("fetch message", zap.Error(err))
				time.Sleep(2 * time.Second)
				continue
			}
			s.enqueue(msg)
		}
	}()
}

func main() {
	cfg := loadConfig()
	logger, err := zap.NewProduction()
	if err != nil {
		panic(err)
	}
	defer logger.Sync()

	registry := telemetry.NewRegistry()
	metrics := newSupervisorMetrics(registry)

	metricsMux := http.NewServeMux()
	metricsMux.Handle("/metrics", registry.Handler())

	var (
		rlReader      *kafka.Reader
		trainingCoord *trainingCoordinator
		slurmBridge   *slurm.SlurmBridge
	)

	if cfg.rlTopic != "" {
		if err := os.MkdirAll(cfg.slurmQueueDir, 0o755); err != nil {
			logger.Warn("failed to create slurm queue dir", zap.String("dir", cfg.slurmQueueDir), zap.Error(err))
		}
		rlReader = kafka.NewReader(kafka.ReaderConfig{
			Brokers:  cfg.kafkaBrokers,
			GroupID:  cfg.rlGroup,
			Topic:    cfg.rlTopic,
			MinBytes: 1 << 10,
			MaxBytes: 10 << 20,
		})
		slurmBridge = slurm.NewSlurmBridge(metrics.registry, cfg.slurmQueueDir, nil)
		trainingCfg := trainingConfig{
			topic:          cfg.rlTopic,
			group:          cfg.rlGroup,
			bufferDir:      cfg.rlBufferDir,
			workspace:      cfg.rlWorkspace,
			meshEndpoint:   cfg.meshEndpoint,
			meshNode:       cfg.meshSourceNode,
			meshDomain:     cfg.meshDomain,
			supervisorAddr: cfg.meshSupervisor,
			maxBuffer:      cfg.rlMaxBuffer,
			slurmQueueDir:  cfg.slurmQueueDir,
		}
		trainingCoord = newTrainingCoordinator(logger, rlReader, metrics, trainingCfg, slurmBridge)
		if trainingCoord == nil {
			logger.Warn("training coordinator disabled (missing configuration)")
			if rlReader != nil {
				rlReader.Close()
				rlReader = nil
			}
		}
	}

	skillsPlanner := newSkillPlanner(logger, trainingCoord, metrics)

	metricsMux.HandleFunc("/training/rl/start", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if trainingCoord == nil {
			http.Error(w, "training disabled", http.StatusServiceUnavailable)
			return
		}
		var req trainingStartRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("invalid payload: %v", err), http.StatusBadRequest)
			return
		}
		job, err := trainingCoord.StartTraining(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"task_id":      job.TaskID,
			"slurm_job_id": job.ID,
			"status":       job.Status,
		})
	})

	metricsMux.HandleFunc("/training/rl/status/", func(w http.ResponseWriter, r *http.Request) {
		if trainingCoord == nil {
			http.Error(w, "training disabled", http.StatusServiceUnavailable)
			return
		}
		taskID := strings.TrimPrefix(r.URL.Path, "/training/rl/status/")
		if taskID == "" {
			http.Error(w, "missing task id", http.StatusBadRequest)
			return
		}
		job, ok := trainingCoord.Status(taskID)
		if !ok {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(job)
	})
	metricsServer := &http.Server{Addr: cfg.metricsListen, Handler: metricsMux}
	metricsErr := make(chan error, 1)
	go func() {
		logger.Info("metrics listener ready", zap.String("listen", cfg.metricsListen))
		if err := metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			metricsErr <- err
		}
	}()

	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:  cfg.kafkaBrokers,
		GroupID:  cfg.groupID,
		Topic:    cfg.inputTopic,
		MinBytes: 1 << 10,
		MaxBytes: 10 << 20,
	})
	defer reader.Close()

	publishers := make(map[uint64]*meshclient.Publisher)
	serviceMap := make(map[uint64]meshServiceConf)
	uniquePublishers := make(map[*meshclient.Publisher]struct{})

	for _, svc := range cfg.meshServices {
		serviceMap[svc.ID] = svc
		ctxMesh, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		pub, err := meshclient.New(ctxMesh, meshclient.Options{
			Endpoint:   svc.Endpoint,
			SourceNode: cfg.meshSourceNode,
			Domain:     svc.Domain,
		})
		cancel()
		if err != nil {
			logger.Warn("mesh publisher unavailable", zap.Uint64("mesh_node", svc.ID), zap.String("endpoint", svc.Endpoint), zap.Error(err))
			continue
		}
		publishers[svc.ID] = pub
		uniquePublishers[pub] = struct{}{}
		logger.Info("mesh publisher ready", zap.Uint64("mesh_node", svc.ID), zap.String("endpoint", svc.Endpoint), zap.String("domain", svc.Domain))
	}

	var meshDefault *meshclient.Publisher
	if cfg.meshEndpoint != "" {
		for id, svc := range serviceMap {
			if svc.Endpoint == cfg.meshEndpoint {
				if pub, ok := publishers[id]; ok {
					meshDefault = pub
				}
				break
			}
		}
		if meshDefault == nil {
			ctxMesh, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			pub, err := meshclient.New(ctxMesh, meshclient.Options{
				Endpoint:   cfg.meshEndpoint,
				SourceNode: cfg.meshSourceNode,
				Domain:     cfg.meshDomain,
			})
			cancel()
			if err != nil {
				logger.Warn("default mesh endpoint unavailable", zap.String("endpoint", cfg.meshEndpoint), zap.Error(err))
			} else {
				meshDefault = pub
				uniquePublishers[pub] = struct{}{}
				logger.Info("mesh default publisher ready", zap.String("endpoint", cfg.meshEndpoint), zap.String("domain", cfg.meshDomain))
			}
		}
	}

	srv := newSupervisorServer(logger, reader, publishers, serviceMap, meshDefault, cfg.meshDomain, metrics, cfg.tenantDomains, trainingCoord, skillsPlanner)

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	if slurmBridge != nil {
		if err := slurmBridge.Start(ctx); err != nil {
			logger.Warn("slurm bridge start failed", zap.Error(err))
		}
	}
	if trainingCoord != nil {
		trainingCoord.Start(ctx)
	}

	srv.runKafka(ctx)

	grpcServer := grpc.NewServer()
	pb.RegisterStreamSupervisorServer(grpcServer, srv)

	listener, err := net.Listen("tcp", cfg.listenAddr)
	if err != nil {
		logger.Fatal("listen", zap.Error(err))
	}

	logger.Info("supervisor ready", zap.String("listen", cfg.listenAddr), zap.Strings("brokers", cfg.kafkaBrokers))

	serveErr := make(chan error, 1)
	go func() {
		serveErr <- grpcServer.Serve(listener)
	}()

	select {
	case <-ctx.Done():
		logger.Info("shutdown signal received")
	case err := <-serveErr:
		if err != nil {
			logger.Error("gRPC server exited", zap.Error(err))
		}
	case err := <-metricsErr:
		if err != nil {
			logger.Error("metrics server exited", zap.Error(err))
		}
	}

	stopCtx, stopCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer stopCancel()
	grpcServer.GracefulStop()
	<-stopCtx.Done()
	srv.stopMeshWorkers()
	ctxShutdown, cancelShutdown := context.WithTimeout(context.Background(), 2*time.Second)
	_ = metricsServer.Shutdown(ctxShutdown)
	cancelShutdown()
	for pub := range uniquePublishers {
		_ = pub.Close()
	}
	if trainingCoord != nil {
		trainingCoord.Stop()
	} else if rlReader != nil {
		rlReader.Close()
	}
	if slurmBridge != nil {
		slurmBridge.Stop()
	}
	logger.Info("supervisor stopped")
}
