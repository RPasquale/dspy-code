package envmanager

import (
	"context"
	"fmt"
	"io"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/dspy/orchestrator/internal/pb/envmanager"
)

// Client wraps the gRPC client for env_manager
type Client struct {
	conn   *grpc.ClientConn
	client pb.EnvManagerServiceClient
	addr   string
}

// NewClient creates a new env_manager client
func NewClient(addr string) (*Client, error) {
	conn, err := grpc.Dial(addr, 
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
		grpc.WithTimeout(10*time.Second),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to env_manager: %w", err)
	}

	client := pb.NewEnvManagerServiceClient(conn)

	return &Client{
		conn:   conn,
		client: client,
		addr:   addr,
	}, nil
}

// Close closes the client connection
func (c *Client) Close() error {
	return c.conn.Close()
}

// StartServices starts all services
func (c *Client) StartServices(ctx context.Context, parallel bool) error {
	log.Printf("Starting services via env_manager (parallel: %t)", parallel)

	req := &pb.StartServicesRequest{
		Parallel: parallel,
	}

	stream, err := c.client.StartServices(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to start services: %w", err)
	}

	for {
		update, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("stream error: %w", err)
		}

		log.Printf("[env_manager] %s: %s (progress: %d%%)",
			update.ServiceName, update.Message, update.Progress)
	}

	log.Println("All services started successfully")
	return nil
}

// StopServices stops all services
func (c *Client) StopServices(ctx context.Context, timeout int32) error {
	log.Printf("Stopping services via env_manager")

	req := &pb.StopServicesRequest{
		TimeoutSeconds: timeout,
	}

	resp, err := c.client.StopServices(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to stop services: %w", err)
	}

	if !resp.Success {
		return fmt.Errorf("stop services failed")
	}

	log.Println("All services stopped successfully")
	return nil
}

// GetServicesStatus retrieves the status of all services
func (c *Client) GetServicesStatus(ctx context.Context) (map[string]*pb.ServiceStatus, error) {
	req := &pb.GetServicesStatusRequest{}

	resp, err := c.client.GetServicesStatus(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get services status: %w", err)
	}

	return resp.Services, nil
}

// StreamHealth streams health updates
func (c *Client) StreamHealth(ctx context.Context, intervalSecs int32, callback func(*pb.HealthUpdate)) error {
	req := &pb.StreamHealthRequest{
		IntervalSeconds: intervalSecs,
	}

	stream, err := c.client.StreamHealth(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to stream health: %w", err)
	}

	for {
		update, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("health stream error: %w", err)
		}

		callback(update)
	}

	return nil
}

// GetResourceAvailability queries available resources
func (c *Client) GetResourceAvailability(ctx context.Context, workloadClass string) (*pb.ResourceAvailabilityResponse, error) {
	req := &pb.GetResourceAvailabilityRequest{
		WorkloadClass: workloadClass,
	}

	resp, err := c.client.GetResourceAvailability(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get resource availability: %w", err)
	}

	return resp, nil
}

