package main

import (
	"context"
	"log"
	"os"

	"github.com/dspy/orchestrator/pkg/envmanager"
	grpcserver "github.com/dspy/orchestrator/internal/grpc"
	"github.com/dspy/orchestrator/internal/workflow"
)

// startGRPCServer starts the gRPC server in a separate goroutine
func startGRPCServer(ctx context.Context, orch *workflow.Orchestrator) {
	grpcAddr := os.Getenv("ORCHESTRATOR_GRPC_ADDR")
	if grpcAddr == "" {
		grpcAddr = ":9098"
	}

	log.Printf("Starting gRPC server on %s", grpcAddr)

	// Create gRPC server with minimal implementation
	server := grpcserver.NewServer(orch, nil)

	go func() {
		if err := server.Serve(grpcAddr); err != nil {
			log.Printf("gRPC server error: %v", err)
		}
	}()
}

// startEnvManager optionally starts the env_manager if configured
func startEnvManager(ctx context.Context) *envmanager.Lifecycle {
	if os.Getenv("DISABLE_ENV_MANAGER") == "1" {
		log.Println("env_manager disabled via DISABLE_ENV_MANAGER=1")
		return nil
	}

	envManagerAddr := os.Getenv("ENV_MANAGER_ADDR")
	if envManagerAddr == "" {
		envManagerAddr = "localhost:50100"
	}

	lifecycle := envmanager.NewLifecycle(envManagerAddr)

	// Start env_manager in the background
	go func() {
		if err := lifecycle.Start(ctx); err != nil {
			log.Printf("Warning: Failed to start env_manager: %v", err)
			log.Println("Continuing without env_manager (containers must be started manually)")
			return
		}

		// Start services via env_manager
		if client := lifecycle.Client(); client != nil {
			log.Println("Starting services via env_manager...")
			if err := client.StartServices(ctx, true); err != nil {
				log.Printf("Warning: Failed to start services: %v", err)
			} else {
				log.Println("All services started successfully")
			}
		}
	}()

	return lifecycle
}

