package envmanager

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"time"
)

// Lifecycle manages the env_manager process lifecycle
type Lifecycle struct {
	binaryPath string
	addr       string
	cmd        *exec.Cmd
	client     *Client
	mu         sync.Mutex
	running    bool
}

// NewLifecycle creates a new lifecycle manager
func NewLifecycle(addr string) *Lifecycle {
	return &Lifecycle{
		addr: addr,
	}
}

// Start starts the env_manager process and connects to it
func (l *Lifecycle) Start(ctx context.Context) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.running {
		return fmt.Errorf("env_manager already running")
	}

	// Find the env_manager binary
	binaryPath, err := l.findBinary()
	if err != nil {
		return fmt.Errorf("failed to find env_manager binary: %w", err)
	}

	l.binaryPath = binaryPath
	log.Printf("Found env_manager binary at: %s", binaryPath)

	// Start the process
	log.Printf("Starting env_manager on %s", l.addr)
	
	l.cmd = exec.CommandContext(ctx, binaryPath)
	l.cmd.Env = append(os.Environ(),
		fmt.Sprintf("ENV_MANAGER_GRPC_ADDR=%s", l.addr),
		"RUST_LOG=info",
	)
	l.cmd.Stdout = os.Stdout
	l.cmd.Stderr = os.Stderr

	if err := l.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start env_manager: %w", err)
	}

	l.running = true
	log.Printf("env_manager process started (PID: %d)", l.cmd.Process.Pid)

	// Wait for env_manager to be ready
	if err := l.waitForReady(ctx); err != nil {
		l.Stop()
		return fmt.Errorf("env_manager failed to become ready: %w", err)
	}

	// Connect client
	client, err := NewClient(l.addr)
	if err != nil {
		l.Stop()
		return fmt.Errorf("failed to connect to env_manager: %w", err)
	}

	l.client = client
	log.Println("Successfully connected to env_manager")

	return nil
}

// Stop stops the env_manager process
func (l *Lifecycle) Stop() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if !l.running {
		return nil
	}

	log.Println("Stopping env_manager")

	// Close client connection
	if l.client != nil {
		l.client.Close()
		l.client = nil
	}

	// Stop process
	if l.cmd != nil && l.cmd.Process != nil {
		if err := l.cmd.Process.Signal(os.Interrupt); err != nil {
			log.Printf("Failed to send interrupt signal: %v", err)
			return l.cmd.Process.Kill()
		}

		// Wait for graceful shutdown with timeout
		done := make(chan error, 1)
		go func() {
			done <- l.cmd.Wait()
		}()

		select {
		case <-time.After(10 * time.Second):
			log.Println("env_manager did not stop gracefully, forcing kill")
			l.cmd.Process.Kill()
		case err := <-done:
			if err != nil {
				log.Printf("env_manager stopped with error: %v", err)
			} else {
				log.Println("env_manager stopped gracefully")
			}
		}
	}

	l.running = false
	return nil
}

// Client returns the connected client (must call Start first)
func (l *Lifecycle) Client() *Client {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.client
}

// IsRunning returns true if env_manager is running
func (l *Lifecycle) IsRunning() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.running
}

// findBinary locates the env_manager binary
func (l *Lifecycle) findBinary() (string, error) {
	// Check common locations
	candidates := []string{
		"env_manager_rs/target/release/env-manager",
		"env_manager_rs/target/debug/env-manager",
		"../env_manager_rs/target/release/env-manager",
		"../../env_manager_rs/target/release/env-manager", // From cmd/dspy-agent/
		"../../../env_manager_rs/target/release/env-manager",
		"./target/release/env-manager",
		filepath.Join(os.Getenv("HOME"), ".dspy-agent", "bin", "env-manager"),
	}

	// Add platform-specific extension
	if runtime.GOOS == "windows" {
		for i := range candidates {
			candidates[i] += ".exe"
		}
	}

	// Check PATH
	if pathBinary, err := exec.LookPath("env-manager"); err == nil {
		return pathBinary, nil
	}

	// Check candidates
	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			absPath, err := filepath.Abs(candidate)
			if err != nil {
				return candidate, nil
			}
			return absPath, nil
		}
	}

	return "", fmt.Errorf("env-manager binary not found in PATH or common locations")
}

// waitForReady waits for env_manager to be ready to accept connections
func (l *Lifecycle) waitForReady(ctx context.Context) error {
	log.Println("Waiting for env_manager to be ready...")

	maxAttempts := 30
	delay := 500 * time.Millisecond

	for attempt := 0; attempt < maxAttempts; attempt++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
			// Try to connect
			client, err := NewClient(l.addr)
			if err == nil {
				// Connection successful, close and return
				client.Close()
				log.Println("env_manager is ready")
				return nil
			}

			// Exponential backoff
			delay = delay * 2
			if delay > 5*time.Second {
				delay = 5 * time.Second
			}
		}
	}

	return fmt.Errorf("env_manager did not become ready after %d attempts", maxAttempts)
}

