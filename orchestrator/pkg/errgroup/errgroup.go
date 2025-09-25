package errgroup

import (
	"context"
	"sync"
)

// Group is a simplified drop-in replacement for golang.org/x/sync/errgroup.Group
// tailored for offline environments.
type Group struct {
	cancel func()
	ctx    context.Context

	mu      sync.Mutex
	err     error
	errOnce sync.Once

	wg sync.WaitGroup
}

// WithContext returns a new Group and context that is cancelled when any Go function fails
// or when the returned context is otherwise cancelled.
func WithContext(ctx context.Context) (*Group, context.Context) {
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)
	return &Group{cancel: cancel, ctx: ctx}, ctx
}

// Go schedules the provided function on a new goroutine.
func (g *Group) Go(fn func() error) {
	g.wg.Add(1)
	go func() {
		defer g.wg.Done()
		if err := fn(); err != nil {
			g.errOnce.Do(func() {
				g.mu.Lock()
				g.err = err
				g.mu.Unlock()
				if g.cancel != nil {
					g.cancel()
				}
			})
		}
	}()
}

// Wait blocks until all functions have returned and propagates the first error (if any).
func (g *Group) Wait() error {
	g.wg.Wait()
	if g.cancel != nil {
		g.cancel()
	}
	g.mu.Lock()
	err := g.err
	g.mu.Unlock()
	return err
}

// Context returns the context associated with the group.
func (g *Group) Context() context.Context { return g.ctx }
