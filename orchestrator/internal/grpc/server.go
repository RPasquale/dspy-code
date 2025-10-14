package grpc

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"

	"github.com/dspy/orchestrator/internal/dispatcher"
	"github.com/dspy/orchestrator/internal/events"
	pb "github.com/dspy/orchestrator/internal/pb/orchestrator"
	"github.com/dspy/orchestrator/internal/workflow"
	"github.com/dspy/orchestrator/pkg/telemetry"
)

// Server implements the OrchestratorService gRPC server
type Server struct {
	pb.UnimplementedOrchestratorServiceServer

	orchestrator *workflow.Orchestrator
	eventBus     EventBus
	grpcServer   *grpc.Server
	registry     *telemetry.Registry
	dispatcher   *dispatcher.TaskDispatcher
	pendDir      string
	eventCancel  func()

	// Task tracking
	tasks      map[string]*TaskState
	tasksMutex sync.RWMutex
}

type TaskState struct {
	ID        string
	Status    string
	Result    map[string]string
	Error     string
	StartedAt time.Time
	UpdatedAt time.Time
}

func stringifyValue(v interface{}) string {
	switch val := v.(type) {
	case string:
		return val
	case fmt.Stringer:
		return val.String()
	case []byte:
		return string(val)
	default:
		if b, err := json.Marshal(v); err == nil {
			return string(b)
		}
		return fmt.Sprint(v)
	}
}

type EventBus interface {
	Subscribe(types []string) (<-chan events.Event, func())
}

func decodePayloadValue(raw string) interface{} {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return raw
	}

	lower := strings.ToLower(trimmed)
	switch lower {
	case "true":
		return true
	case "false":
		return false
	case "null":
		return nil
	}

	if i, err := strconv.ParseInt(trimmed, 10, 64); err == nil {
		return i
	}

	if strings.ContainsAny(trimmed, ".eE") {
		if f, err := strconv.ParseFloat(trimmed, 64); err == nil {
			return f
		}
	}

	if (strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}")) ||
		(strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]")) {
		var data interface{}
		if err := json.Unmarshal([]byte(trimmed), &data); err == nil {
			return data
		}
	}

	return raw
}

func decodePayloadMap(input map[string]string) map[string]interface{} {
	out := make(map[string]interface{}, len(input))
	for k, v := range input {
		out[k] = decodePayloadValue(v)
	}

	if raw, ok := out["payload_json"]; ok {
		if str, ok := raw.(string); ok {
			var data map[string]interface{}
			if err := json.Unmarshal([]byte(str), &data); err == nil {
				for dk, dv := range data {
					out[dk] = dv
				}
				delete(out, "payload_json")
			}
		}
	}

	return out
}

func (s *Server) consumeEvents(ch <-chan events.Event) {
	for evt := range ch {
		idVal, ok := evt.Data["task_id"]
		if !ok {
			continue
		}
		id := fmt.Sprint(idVal)
		if strings.TrimSpace(id) == "" {
			continue
		}

		switch evt.Type {
		case "task_submitted":
			s.updateTask(id, func(state *TaskState) {
				state.Status = "pending"
				state.UpdatedAt = time.Now()
			})
		case "task_started":
			s.updateTask(id, func(state *TaskState) {
				state.Status = "running"
				state.Error = ""
				state.UpdatedAt = time.Now()
				if ts, ok := evt.Data["started_at"].(string); ok && ts != "" {
					if parsed, err := time.Parse(time.RFC3339Nano, ts); err == nil {
						state.StartedAt = parsed
					}
				} else {
					state.StartedAt = time.Now()
				}
				for k, v := range evt.Data {
					if k == "task_id" || k == "started_at" {
						continue
					}
					state.Result[k] = stringifyValue(v)
				}
			})
		case "task_completed":
			result := make(map[string]string)
			if raw, ok := evt.Data["result"].(map[string]interface{}); ok {
				for k, v := range raw {
					result[k] = stringifyValue(v)
				}
			} else {
				for k, v := range evt.Data {
					if k == "task_id" {
						continue
					}
					result[k] = stringifyValue(v)
				}
			}
			s.updateTask(id, func(state *TaskState) {
				state.Status = "completed"
				state.Error = ""
				state.Result = result
				state.UpdatedAt = time.Now()
			})
		case "task_failed":
			errMsg := fmt.Sprint(evt.Data["error"])
			s.updateTask(id, func(state *TaskState) {
				state.Status = "failed"
				state.Error = errMsg
				state.UpdatedAt = time.Now()
			})
		}
	}
}

func (s *Server) updateTask(id string, mutate func(*TaskState)) {
	s.tasksMutex.Lock()
	state, ok := s.tasks[id]
	if !ok {
		state = &TaskState{
			ID:        id,
			Status:    "pending",
			Result:    make(map[string]string),
			StartedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		s.tasks[id] = state
	}
	mutate(state)
	s.tasksMutex.Unlock()
}

// NewServer creates a new gRPC server
func NewServer(orch *workflow.Orchestrator, eventBus EventBus, registry *telemetry.Registry, dispatcher *dispatcher.TaskDispatcher, pendDir string) *Server {
	srv := &Server{
		orchestrator: orch,
		eventBus:     eventBus,
		registry:     registry,
		dispatcher:   dispatcher,
		pendDir:      pendDir,
		tasks:        make(map[string]*TaskState),
	}
	if eventBus != nil {
		ch, cancel := eventBus.Subscribe([]string{"task_submitted", "task_started", "task_completed", "task_failed"})
		srv.eventCancel = cancel
		go srv.consumeEvents(ch)
	}
	return srv
}

// Serve starts the gRPC server on the given address
func (s *Server) Serve(addr string) error {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterOrchestratorServiceServer(grpcServer, s)
	reflection.Register(grpcServer)
	s.grpcServer = grpcServer

	log.Printf("gRPC server listening on %s", addr)
	return grpcServer.Serve(lis)
}

// Stop gracefully stops the gRPC server.
func (s *Server) Stop() {
	if s.eventCancel != nil {
		s.eventCancel()
	}
	if s.grpcServer != nil {
		s.grpcServer.GracefulStop()
	}
}

// SubmitTask handles task submission
func (s *Server) SubmitTask(ctx context.Context, req *pb.SubmitTaskRequest) (*pb.SubmitTaskResponse, error) {
	id := strings.TrimSpace(req.Id)
	class := strings.TrimSpace(req.Class)
	if class == "" {
		class = "cpu_short"
	}
	if id == "" {
		id = time.Now().Format("20060102T150405.000000000")
	}

	log.Printf("gRPC: Received task submission: %s (class: %s)", id, class)

	payload := decodePayloadMap(req.Payload)

	envelope := map[string]interface{}{
		"id":      id,
		"class":   class,
		"payload": payload,
	}

	if s.pendDir == "" {
		return nil, status.Error(codes.FailedPrecondition, "queue directory not configured")
	}

	if err := os.MkdirAll(s.pendDir, 0o755); err != nil {
		return nil, status.Errorf(codes.Internal, "ensure queue directory: %v", err)
	}

	fpath := filepath.Join(s.pendDir, id+".json")
	serialized, err := json.Marshal(envelope)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "serialize task: %v", err)
	}
	if err := os.WriteFile(fpath, serialized, 0o644); err != nil {
		return nil, status.Errorf(codes.Internal, "persist task: %v", err)
	}

	if s.dispatcher != nil {
		s.dispatcher.RefreshQueueGauge()
		s.dispatcher.Dispatch(context.Background(), envelope)
	}

	now := time.Now()
	s.tasksMutex.Lock()
	s.tasks[id] = &TaskState{
		ID:        id,
		Status:    "pending",
		Result:    map[string]string{},
		StartedAt: now,
		UpdatedAt: now,
	}
	s.tasksMutex.Unlock()

	return &pb.SubmitTaskResponse{
		Success: true,
		TaskId:  id,
	}, nil
}

// StreamTaskResults streams task results as they complete
func (s *Server) StreamTaskResults(req *pb.StreamTaskResultsRequest, stream pb.OrchestratorService_StreamTaskResultsServer) error {
	log.Printf("gRPC: Streaming task results")

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-stream.Context().Done():
			return nil
		case <-ticker.C:
			// Send updates for all tasks
			s.tasksMutex.RLock()
			for _, task := range s.tasks {
				if len(req.TaskIds) > 0 {
					// Filter by requested task IDs
					found := false
					for _, id := range req.TaskIds {
						if id == task.ID {
							found = true
							break
						}
					}
					if !found {
						continue
					}
				}

				result := &pb.TaskResult{
					TaskId:      task.ID,
					Status:      task.Status,
					Result:      task.Result,
					Error:       task.Error,
					DurationMs:  float64(time.Since(task.StartedAt).Milliseconds()),
					CompletedAt: task.UpdatedAt.Unix(),
				}

				if err := stream.Send(result); err != nil {
					s.tasksMutex.RUnlock()
					return err
				}
			}
			s.tasksMutex.RUnlock()
		}
	}
}

func (s *Server) GetTaskStatus(ctx context.Context, req *pb.GetTaskStatusRequest) (*pb.GetTaskStatusResponse, error) {
	id := strings.TrimSpace(req.TaskId)
	if id == "" {
		return nil, status.Error(codes.InvalidArgument, "task_id is required")
	}

	s.tasksMutex.RLock()
	state, ok := s.tasks[id]
	s.tasksMutex.RUnlock()
	if !ok {
		return nil, status.Error(codes.NotFound, "task not found")
	}

	result := make(map[string]string, len(state.Result))
	for k, v := range state.Result {
		result[k] = v
	}

	return &pb.GetTaskStatusResponse{
		TaskId: state.ID,
		Status: state.Status,
		Result: result,
		Error:  state.Error,
	}, nil
}

// CreateWorkflow creates a new workflow
func (s *Server) CreateWorkflow(ctx context.Context, req *pb.CreateWorkflowRequest) (*pb.CreateWorkflowResponse, error) {
	log.Printf("gRPC: Creating workflow: %s", req.Name)

	// TODO: Implement workflow creation
	return &pb.CreateWorkflowResponse{
		Success:    true,
		WorkflowId: req.Id,
	}, nil
}

// StartWorkflowRun starts a workflow execution
func (s *Server) StartWorkflowRun(ctx context.Context, req *pb.StartWorkflowRunRequest) (*pb.StartWorkflowRunResponse, error) {
	log.Printf("gRPC: Starting workflow run: %s", req.WorkflowId)

	// TODO: Implement workflow execution
	runID := fmt.Sprintf("%s-run-%d", req.WorkflowId, time.Now().Unix())

	return &pb.StartWorkflowRunResponse{
		Success: true,
		RunId:   runID,
	}, nil
}

// StreamWorkflowStatus streams workflow execution status
func (s *Server) StreamWorkflowStatus(req *pb.StreamWorkflowStatusRequest, stream pb.OrchestratorService_StreamWorkflowStatusServer) error {
	log.Printf("gRPC: Streaming workflow status: %s", req.RunId)

	// TODO: Implement real workflow status streaming
	// For now, send a mock update
	update := &pb.WorkflowStatusUpdate{
		RunId:          req.RunId,
		Status:         "running",
		CurrentStep:    "step1",
		StepsCompleted: 1,
		TotalSteps:     3,
	}

	if err := stream.Send(update); err != nil {
		return err
	}

	return nil
}

// GetMetrics returns system metrics
func (s *Server) GetMetrics(ctx context.Context, req *pb.GetMetricsRequest) (*pb.MetricsResponse, error) {
	log.Printf("gRPC: GetMetrics method called")

	metrics := s.snapshotRegistryMetrics()

	// Add task tracking insights from the in-memory queue.
	s.tasksMutex.RLock()
	metrics["tasks_tracked_total"] = float64(len(s.tasks))
	pending := 0
	running := 0
	for _, task := range s.tasks {
		switch strings.ToLower(task.Status) {
		case "pending":
			pending++
		case "running":
			running++
		}
	}
	s.tasksMutex.RUnlock()
	if _, exists := metrics["tasks_pending"]; !exists {
		metrics["tasks_pending"] = float64(pending)
	}
	if _, exists := metrics["tasks_running"]; !exists {
		metrics["tasks_running"] = float64(running)
	}
	if _, exists := metrics["tasks_completed"]; !exists {
		completed := 0
		s.tasksMutex.RLock()
		for _, task := range s.tasks {
			if strings.ToLower(task.Status) == "completed" {
				completed++
			}
		}
		s.tasksMutex.RUnlock()
		metrics["tasks_completed"] = float64(completed)
	}

	return &pb.MetricsResponse{
		Metrics:   metrics,
		Timestamp: time.Now().Unix(),
	}, nil
}

func (s *Server) snapshotRegistryMetrics() map[string]float64 {
	out := make(map[string]float64)
	if s.registry == nil {
		return out
	}

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/metrics", nil)
	s.registry.Handler().ServeHTTP(recorder, req)

	scanner := bufio.NewScanner(recorder.Body)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		name := fields[0]
		valueStr := fields[len(fields)-1]
		value, err := strconv.ParseFloat(valueStr, 64)
		if err != nil {
			continue
		}
		out[name] = value
	}
	return out
}

// StreamEvents streams system events
func (s *Server) StreamEvents(req *pb.StreamEventsRequest, stream pb.OrchestratorService_StreamEventsServer) error {
	log.Printf("gRPC: Streaming events (types: %v)", req.EventTypes)

	eventsCh, cancel := s.eventBus.Subscribe(req.EventTypes)
	defer cancel()

	for {
		select {
		case <-stream.Context().Done():
			return nil
		case evt, ok := <-eventsCh:
			if !ok {
				return nil
			}
			resourceID := ""
			if v, ok := evt.Data["resource_id"]; ok {
				resourceID = fmt.Sprint(v)
			}
			payload := &pb.SystemEvent{
				EventType:  evt.Type,
				ResourceId: resourceID,
				Timestamp:  evt.Timestamp.Unix(),
				Data:       make(map[string]string, len(evt.Data)),
			}
			for k, v := range evt.Data {
				payload.Data[k] = fmt.Sprint(v)
			}
			if err := stream.Send(payload); err != nil {
				return err
			}
		}
	}
}

// Health returns health status
func (s *Server) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	log.Printf("gRPC: Health method called")
	services := map[string]string{
		"orchestrator": "healthy",
		"grpc_server":  "healthy",
	}

	return &pb.HealthResponse{
		Healthy:  true,
		Version:  "0.1.0",
		Services: services,
	}, nil
}
