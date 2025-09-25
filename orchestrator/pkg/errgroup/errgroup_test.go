package errgroup

import (
	"errors"
	"testing"
	"time"
)

func TestGroupWaitSuccess(t *testing.T) {
	g, ctx := WithContext(nil)
	_ = ctx
	g.Go(func() error { return nil })
	if err := g.Wait(); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
}

func TestGroupWaitCancelsOnError(t *testing.T) {
	g, ctx := WithContext(nil)
	done := make(chan struct{})
	g.Go(func() error {
		time.Sleep(20 * time.Millisecond)
		return errors.New("boom")
	})
	g.Go(func() error {
		select {
		case <-ctx.Done():
			close(done)
			return ctx.Err()
		case <-time.After(200 * time.Millisecond):
			t.Fatalf("context not cancelled")
			return nil
		}
	})
	err := g.Wait()
	if err == nil {
		t.Fatalf("expected error")
	}
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Fatalf("secondary goroutine not cancelled")
	}
}
