package grpc

import (
	"context"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"

	pb "github.com/dspy/orchestrator/internal/pb/orchestrator"
	"github.com/dspy/orchestrator/internal/workflow"
)

// Server implements the OrchestratorService gRPC server
type Server struct {
	pb.UnimplementedOrchestratorServiceServer
	
	orchestrator *workflow.Orchestrator
	eventBus     EventBus
	
	// Task tracking
	tasks      map[string]*TaskState
	tasksMutex sync.RWMutex
	
	// Event subscribers
	subscribers    map[string]chan *pb.SystemEvent
	subscribersMux sync.RWMutex
}

type TaskState struct {
	ID        string
	Status    string
	Result    map[string]string
	Error     string
	StartedAt time.Time
	UpdatedAt time.Time
}

type EventBus interface {
	Subscribe(types []string) <-chan Event
}

type Event struct {
	Type       string
	ResourceID string
	Data       map[string]interface{}
	Timestamp  time.Time
}

// NewServer creates a new gRPC server
func NewServer(orch *workflow.Orchestrator, eventBus EventBus) *Server {
	return &Server{
		orchestrator: orch,
		eventBus:    eventBus,
		tasks:       make(map[string]*TaskState),
		subscribers: make(map[string]chan *pb.SystemEvent),
	}
}

// Serve starts the gRPC server on the given address
func (s *Server) Serve(addr string) error {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterOrchestratorServiceServer(grpcServer, s)

	log.Printf("gRPC server listening on %s", addr)
	return grpcServer.Serve(lis)
}

// SubmitTask handles task submission
func (s *Server) SubmitTask(ctx context.Context, req *pb.SubmitTaskRequest) (*pb.SubmitTaskResponse, error) {
	log.Printf("gRPC: Received task submission: %s (class: %s)", req.Id, req.Class)

	// Register task
	s.tasksMutex.Lock()
	s.tasks[req.Id] = &TaskState{
		ID:        req.Id,
		Status:    "pending",
		StartedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	s.tasksMutex.Unlock()

	// TODO: Submit to orchestrator
	// For now, just acknowledge receipt
	
	return &pb.SubmitTaskResponse{
		Success: true,
		TaskId:  req.Id,
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
		RunId:           req.RunId,
		Status:          "running",
		CurrentStep:     "step1",
		StepsCompleted:  1,
		TotalSteps:      3,
	}

	if err := stream.Send(update); err != nil {
		return err
	}

	return nil
}

// GetMetrics returns system metrics
func (s *Server) GetMetrics(ctx context.Context, req *pb.GetMetricsRequest) (*pb.MetricsResponse, error) {
	// TODO: Collect real metrics from orchestrator
	metrics := map[string]float64{
		"tasks_pending":   0,
		"tasks_running":   0,
		"tasks_completed": float64(len(s.tasks)),
	}

	return &pb.MetricsResponse{
		Metrics:   metrics,
		Timestamp: time.Now().Unix(),
	}, nil
}

// StreamEvents streams system events
func (s *Server) StreamEvents(req *pb.StreamEventsRequest, stream pb.OrchestratorService_StreamEventsServer) error {
	log.Printf("gRPC: Streaming events (types: %v)", req.EventTypes)

	subscriberID := fmt.Sprintf("subscriber-%d", time.Now().UnixNano())
	eventChan := make(chan *pb.SystemEvent, 100)

	s.subscribersMux.Lock()
	s.subscribers[subscriberID] = eventChan
	s.subscribersMux.Unlock()

	defer func() {
		s.subscribersMux.Lock()
		delete(s.subscribers, subscriberID)
		close(eventChan)
		s.subscribersMux.Unlock()
	}()

	for {
		select {
		case <-stream.Context().Done():
			return nil
		case event := <-eventChan:
			if err := stream.Send(event); err != nil {
				return err
			}
		}
	}
}

// Health returns health status
func (s *Server) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
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

// PublishEvent publishes an event to all subscribers
func (s *Server) PublishEvent(event *pb.SystemEvent) {
	s.subscribersMux.RLock()
	defer s.subscribersMux.RUnlock()

	for _, ch := range s.subscribers {
		select {
		case ch <- event:
		default:
			// Drop event if channel is full
		}
	}
}

