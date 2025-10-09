//go:build mesh_integration

package meshclient

import (
	"context"
	"fmt"

	pb "github.com/dspy/orchestrator/internal/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
)

// Publisher streams task assignments into the mesh network so downstream
// workers can receive them via MeshData.Subscribe.
type Publisher struct {
	conn   *grpc.ClientConn
	client pb.MeshDataClient
	srcID  uint64
	domain string
}

// Options configures the publisher.
type Options struct {
	Endpoint   string
	SourceNode uint64
	Domain     string
}

// New creates a mesh publisher using insecure credentials by default.
func New(ctx context.Context, opt Options) (*Publisher, error) {
	if opt.Endpoint == "" {
		return nil, fmt.Errorf("mesh endpoint required")
	}
	if opt.Domain == "" {
		opt.Domain = "default"
	}

	conn, err := grpc.DialContext(ctx, opt.Endpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("dial mesh endpoint: %w", err)
	}

	client := pb.NewMeshDataClient(conn)
	return &Publisher{
		conn:   conn,
		client: client,
		srcID:  opt.SourceNode,
		domain: opt.Domain,
	}, nil
}

// Close releases the underlying gRPC connection.
func (p *Publisher) Close() error {
	if p.conn == nil {
		return nil
	}
	return p.conn.Close()
}

// SendAssignment encodes the supervisor task assignment and publishes it to the mesh.
func (p *Publisher) SendAssignment(ctx context.Context, dstNode uint64, domain string, msg proto.Message) (string, error) {
	assignment, ok := msg.(*pb.TaskAssignment)
	if !ok {
		return "", fmt.Errorf("unsupported message type %T", msg)
	}
	if assignment == nil {
		return "", fmt.Errorf("nil assignment")
	}
	payload, err := proto.Marshal(assignment)
	if err != nil {
		return "", fmt.Errorf("marshal assignment: %w", err)
	}
	useDomain := p.domain
	if domain != "" {
		useDomain = domain
	}
	resp, err := p.client.Send(ctx, &pb.SendRequest{
		SrcNode:    p.srcID,
		DstNode:    dstNode,
		Payload:    payload,
		RequireAck: true,
		Domain:     useDomain,
	})
	if err != nil {
		return "", fmt.Errorf("mesh send: %w", err)
	}
	return resp.GetDeliveryId(), nil
}
