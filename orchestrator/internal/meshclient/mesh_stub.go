//go:build !mesh_integration

package meshclient

import (
	"context"
	"errors"

	"google.golang.org/protobuf/proto"
)

// Options mirrors the build-tagged implementation to keep call sites consistent.
type Options struct {
	Endpoint   string
	SourceNode uint64
	Domain     string
}

// Publisher is a no-op placeholder used when mesh integration is not enabled.
type Publisher struct{}

// New returns an error signalling that mesh support is not compiled in.
func New(context.Context, Options) (*Publisher, error) {
	return nil, errors.New("mesh integration not enabled; rebuild with -tags mesh_integration")
}

// Close is a no-op stub.
func (p *Publisher) Close() error { return nil }

// SendAssignment returns an error explaining that mesh is disabled.
func (p *Publisher) SendAssignment(context.Context, uint64, string, proto.Message) (string, error) {
	return "", errors.New("mesh integration not enabled")
}
