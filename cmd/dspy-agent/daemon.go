package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/dspy/orchestrator/pkg/envmanager"
)

func runStart(workspace string, gpu bool, services []string, daemon bool) error {
	log.Printf("Starting DSPy agent infrastructure...")
	log.Printf("Workspace: %s", workspace)
	log.Printf("GPU enabled: %v", gpu)

	if daemon {
		return startDaemon(workspace, gpu, services)
	}

	return startForeground(workspace, gpu, services)
}

func startForeground(workspace string, gpu bool, services []string) error {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	// Step 1: Start env_manager
	log.Println("→ Starting environment manager...")
	envManagerAddr := "0.0.0.0:50100"
	lifecycle := envmanager.NewLifecycle(envManagerAddr)

	if err := lifecycle.Start(ctx); err != nil {
		return fmt.Errorf("failed to start env_manager: %w", err)
	}
	defer lifecycle.Stop()

	log.Println("✓ Environment manager started")

	// Step 2: Start services via env_manager
	log.Println("→ Starting infrastructure services...")
	client := lifecycle.Client()
	if client == nil {
		return fmt.Errorf("env_manager client not available")
	}

	if err := client.StartServices(ctx, true); err != nil {
		return fmt.Errorf("failed to start services: %w", err)
	}

	log.Println("✓ All services started successfully")

	// Step 3: Start orchestrator
	log.Println("→ Starting orchestrator...")
	orchestratorCmd, err := startOrchestrator(ctx)
	if err != nil {
		return fmt.Errorf("failed to start orchestrator: %w", err)
	}
	defer func() {
		if orchestratorCmd != nil && orchestratorCmd.Process != nil {
			orchestratorCmd.Process.Signal(syscall.SIGTERM)
		}
	}()

	log.Println("✓ Orchestrator started")

	// Print status
	printStatus(client)

	log.Println("\n✅ DSPy agent infrastructure is ready!")
	log.Println("\nPress Ctrl+C to stop all services")

	// Wait for interrupt
	<-ctx.Done()

	log.Println("\n🛑 Shutting down...")
	return nil
}

func startDaemon(workspace string, gpu bool, services []string) error {
	log.Println("Starting in daemon mode...")

	// Create daemon process
	cmd := exec.Command(os.Args[0], "start", "--workspace", workspace)
	if gpu {
		cmd.Args = append(cmd.Args, "--gpu")
	}

	// Detach from terminal
	cmd.Stdout = nil
	cmd.Stderr = nil
	cmd.Stdin = nil

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start daemon: %w", err)
	}

	// Save PID
	pidFile := filepath.Join(os.TempDir(), "dspy-agent.pid")
	if err := os.WriteFile(pidFile, []byte(fmt.Sprintf("%d", cmd.Process.Pid)), 0644); err != nil {
		log.Printf("Warning: Could not write PID file: %v", err)
	}

	log.Printf("✓ Daemon started (PID: %d)", cmd.Process.Pid)
	log.Printf("  PID file: %s", pidFile)
	
	return nil
}

func runStop(timeout int, force bool) error {
	log.Println("Stopping DSPy agent infrastructure...")

	// Try to connect to env_manager
	client, err := envmanager.NewClient("localhost:50100")
	if err != nil {
		log.Printf("Warning: Could not connect to env_manager: %v", err)
		return fmt.Errorf("services may not be running")
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()

	// Stop services
	if err := client.StopServices(ctx, int32(timeout)); err != nil {
		if force {
			log.Printf("Warning: Graceful stop failed, attempting force stop...")
			// TODO: Implement force stop
		} else {
			return fmt.Errorf("failed to stop services: %w", err)
		}
	}

	log.Println("✓ All services stopped")
	return nil
}

func runStatus() error {
	log.Println("DSPy Agent Infrastructure Status\n")

	// Try to connect to env_manager
	client, err := envmanager.NewClient("localhost:50100")
	if err != nil {
		fmt.Println("❌ Services not running (env_manager not accessible)")
		return nil
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Get services status
	services, err := client.GetServicesStatus(ctx)
	if err != nil {
		return fmt.Errorf("failed to get status: %w", err)
	}

	// Print status table
	fmt.Println("Service              Status      Container ID")
	fmt.Println("------------------   ---------   ----------------------------------")

	for name, status := range services {
		statusIcon := "⚪"
		switch status.Status {
		case "running":
			statusIcon = "🟢"
		case "starting":
			statusIcon = "🟡"
		case "stopped":
			statusIcon = "🔴"
		case "unhealthy":
			statusIcon = "🟠"
		}

		containerID := status.ContainerId
		if len(containerID) > 12 {
			containerID = containerID[:12]
		}

		fmt.Printf("%-20s %s %-10s  %s\n", name, statusIcon, status.Status, containerID)
	}

	fmt.Println()
	return nil
}

func runLogs(service string, follow bool, tail int) error {
	// TODO: Implement log streaming
	log.Println("Log streaming not yet implemented")
	return nil
}

func startOrchestrator(ctx context.Context) (*exec.Cmd, error) {
	// Find orchestrator binary
	orchestratorPaths := []string{
		"orchestrator/orchestrator-linux",
		"orchestrator/orchestrator",
		"orchestrator/main",
		"../orchestrator/orchestrator-linux",
		"../orchestrator/orchestrator",
		"../orchestrator/main",
		"./orchestrator/orchestrator-linux",
		"./orchestrator/orchestrator",
		"./orchestrator/main",
		"./orchestrator",
	}

	var orchestratorBin string
	for _, path := range orchestratorPaths {
		absPath, _ := filepath.Abs(path)
		if _, err := os.Stat(absPath); err == nil {
			orchestratorBin = absPath
			break
		}
	}

	if orchestratorBin == "" {
		return nil, fmt.Errorf("orchestrator binary not found in paths: %v", orchestratorPaths)
	}
	
	log.Printf("Found orchestrator binary at: %s", orchestratorBin)

	cmd := exec.CommandContext(ctx, orchestratorBin)
	cmd.Env = append(os.Environ(),
		"ORCHESTRATOR_GRPC_ADDR=:9098",
		"ENV_MANAGER_ADDR=localhost:50100",
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	// Wait a moment for orchestrator to start
	time.Sleep(2 * time.Second)

	return cmd, nil
}

func printStatus(client *envmanager.Client) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	services, err := client.GetServicesStatus(ctx)
	if err != nil {
		log.Printf("Warning: Could not get services status: %v", err)
		return
	}

	fmt.Println("\n📊 Services Status:")
	for name, status := range services {
		statusIcon := "⚪"
		if status.Status == "running" {
			statusIcon = "✓"
		}
		fmt.Printf("  %s %s: %s\n", statusIcon, name, status.Status)
	}
}

