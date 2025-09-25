package limiter

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestAdaptiveLimiterAcquireRelease(t *testing.T) {
	l := NewAdaptiveLimiter(2)
	ctx := context.Background()
	if err := l.Acquire(ctx); err != nil {
		t.Fatalf("unexpected acquire error: %v", err)
	}
	if err := l.Acquire(ctx); err != nil {
		t.Fatalf("unexpected second acquire error: %v", err)
	}
	done := make(chan struct{})
	go func() {
		if err := l.Acquire(ctx); err != nil {
			t.Errorf("unexpected third acquire err: %v", err)
		}
		close(done)
	}()
	select {
	case <-done:
		t.Fatalf("expected third acquire to block until release")
	case <-time.After(50 * time.Millisecond):
	}
	l.Release()
	select {
	case <-done:
	case <-time.After(200 * time.Millisecond):
		t.Fatalf("expected waiter to unblock after release")
	}
	if got := l.InUse(); got != 2 {
		t.Fatalf("expected 2 in use, got %d", got)
	}
}

func TestAdaptiveLimiterResize(t *testing.T) {
	l := NewAdaptiveLimiter(1)
	ctx := context.Background()
	if err := l.Acquire(ctx); err != nil {
		t.Fatalf("unexpected acquire: %v", err)
	}
	blocked := make(chan struct{})
	go func() {
		if err := l.Acquire(ctx); err != nil {
			t.Errorf("blocked acquire error: %v", err)
		}
		close(blocked)
	}()
	select {
	case <-blocked:
		t.Fatalf("expected second acquire to block")
	case <-time.After(50 * time.Millisecond):
	}
	l.Resize(2)
	select {
	case <-blocked:
	case <-time.After(200 * time.Millisecond):
		t.Fatalf("resize did not unblock waiter")
	}
}

func TestAdaptiveLimiterContextCancellation(t *testing.T) {
	l := NewAdaptiveLimiter(1)
	ctx := context.Background()
	if err := l.Acquire(ctx); err != nil {
		t.Fatalf("unexpected acquire: %v", err)
	}
	cancelCtx, cancel := context.WithCancel(context.Background())
	done := make(chan error)
	go func() { done <- l.Acquire(cancelCtx) }()
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatalf("expected context cancellation error")
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatalf("blocked acquire not cancelled by context")
	}
}

func TestAdaptiveLimiterClose(t *testing.T) {
	l := NewAdaptiveLimiter(1)
	ctx := context.Background()
	if err := l.Acquire(ctx); err != nil {
		t.Fatalf("unexpected acquire: %v", err)
	}
	done := make(chan error)
	go func() { done <- l.Acquire(ctx) }()
	time.Sleep(50 * time.Millisecond)
	l.Close()
	select {
	case err := <-done:
		if err != ErrLimiterClosed {
			t.Fatalf("expected ErrLimiterClosed, got %v", err)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatalf("close did not notify waiter")
	}
}

func TestAdaptiveLimiterResizeLowerThanInUse(t *testing.T) {
	l := NewAdaptiveLimiter(3)
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		if err := l.Acquire(ctx); err != nil {
			t.Fatalf("unexpected acquire %d: %v", i, err)
		}
	}
	done := make(chan struct{})
	go func() {
		if err := l.Acquire(ctx); err != nil {
			t.Errorf("acquire after shrink err: %v", err)
		}
		close(done)
	}()
	l.Resize(1)
	select {
	case <-done:
		t.Fatalf("expected waiter to stay blocked until releases")
	case <-time.After(100 * time.Millisecond):
	}
    l.Release()
    l.Release()
    l.Release()
    select {
    case <-done:
    case <-time.After(200 * time.Millisecond):
        t.Fatalf("waiter should unblock once inUse <= limit")
    }
}

func TestAdaptiveLimiterConcurrentWaiters(t *testing.T) {
	l := NewAdaptiveLimiter(2)
	ctx := context.Background()
	var wg sync.WaitGroup
	start := make(chan struct{})
	worker := func(id int) {
		defer wg.Done()
		<-start
		if err := l.Acquire(ctx); err != nil {
			t.Errorf("worker %d acquire error: %v", id, err)
			return
		}
		time.Sleep(5 * time.Millisecond)
		l.Release()
	}
	wg.Add(5)
	for i := 0; i < 5; i++ {
		go worker(i)
	}
	close(start)
	wg.Wait()
	if got := l.InUse(); got != 0 {
		t.Fatalf("expected zero in use after workers, got %d", got)
	}
}
