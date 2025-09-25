package limiter

import (
	"context"
	"errors"
	"sync"
)

// ErrLimiterClosed signals the limiter has been closed and can no longer grant permits.
var ErrLimiterClosed = errors.New("limiter closed")

// AdaptiveLimiter is a dynamically resizable semaphore that coordinates concurrent workflows.
// It supports context-aware acquisition and ensures FIFO fairness for waiting goroutines.
type AdaptiveLimiter struct {
	mu      sync.Mutex
	limit   int
	inUse   int
	closed  bool
	waiters []chan struct{}
}

// NewAdaptiveLimiter returns a limiter with the provided initial concurrency limit.
// Limit must be non-negative; zero means no work permitted until resized.
func NewAdaptiveLimiter(limit int) *AdaptiveLimiter {
	if limit < 0 {
		panic("limit must be >= 0")
	}
	return &AdaptiveLimiter{limit: limit}
}

// Acquire blocks until capacity is available, the context is cancelled, or the limiter closes.
// FIFO fairness is achieved by queueing waiters in the order they attempt to acquire.
func (l *AdaptiveLimiter) Acquire(ctx context.Context) error {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return ErrLimiterClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	for {
		if ctx.Err() != nil {
			l.mu.Unlock()
			return ctx.Err()
		}
		// If no capacity, queue and wait
		if !l.closed && (l.limit == 0 || l.inUse >= l.limit) {
			waiter := make(chan struct{}, 1)
			l.waiters = append(l.waiters, waiter)
			l.mu.Unlock()
			select {
			case <-ctx.Done():
				l.removeWaiter(waiter)
				return ctx.Err()
			case <-waiter:
			}
			l.mu.Lock()
			if l.closed {
				l.mu.Unlock()
				return ErrLimiterClosed
			}
			continue
		}
		if l.limit == 0 {
			// Still blocked because limit is zero; loop again to wait
			waiter := make(chan struct{}, 1)
			l.waiters = append(l.waiters, waiter)
			l.mu.Unlock()
			select {
			case <-ctx.Done():
				l.removeWaiter(waiter)
				return ctx.Err()
			case <-waiter:
			}
			l.mu.Lock()
			if l.closed {
				l.mu.Unlock()
				return ErrLimiterClosed
			}
			continue
		}
		l.inUse++
		l.mu.Unlock()
		return nil
	}
}

// Release frees a previously acquired slot and wakes the next waiter if any.
func (l *AdaptiveLimiter) Release() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.inUse > 0 {
		l.inUse--
	}
	if len(l.waiters) > 0 && l.available() > 0 {
		waiter := l.waiters[0]
		copy(l.waiters[0:], l.waiters[1:])
		l.waiters = l.waiters[:len(l.waiters)-1]
		select {
		case waiter <- struct{}{}:
		default:
		}
	}
}

// Resize adjusts the concurrency limit and wakes any eligible waiters.
func (l *AdaptiveLimiter) Resize(newLimit int) {
	if newLimit < 0 {
		panic("limit must be >= 0")
	}
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return
	}
	l.limit = newLimit
	available := l.available()
	for available > 0 && len(l.waiters) > 0 {
		waiter := l.waiters[0]
		copy(l.waiters[0:], l.waiters[1:])
		l.waiters = l.waiters[:len(l.waiters)-1]
		select {
		case waiter <- struct{}{}:
		default:
		}
		available--
	}
	l.mu.Unlock()
}

// Close prevents further acquisitions and unblocks waiters with ErrLimiterClosed.
func (l *AdaptiveLimiter) Close() {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return
	}
	l.closed = true
	for _, waiter := range l.waiters {
		select {
		case waiter <- struct{}{}:
		default:
		}
	}
	l.waiters = nil
	l.mu.Unlock()
}

// Limit returns the current configured limit.
func (l *AdaptiveLimiter) Limit() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.limit
}

// InUse returns the number of currently acquired permits.
func (l *AdaptiveLimiter) InUse() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.inUse
}

func (l *AdaptiveLimiter) available() int {
	if l.limit == 0 {
		return 0
	}
	if l.inUse >= l.limit {
		return 0
	}
	return l.limit - l.inUse
}

func (l *AdaptiveLimiter) removeWaiter(target chan struct{}) {
	l.mu.Lock()
	defer l.mu.Unlock()
	for i, waiter := range l.waiters {
		if waiter == target {
			copy(l.waiters[i:], l.waiters[i+1:])
			l.waiters = l.waiters[:len(l.waiters)-1]
			return
		}
	}
}
