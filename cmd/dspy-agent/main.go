package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var version = "0.1.0"

func main() {
	rootCmd := &cobra.Command{
		Use:   "dspy-agent",
		Short: "DSPy Agent Infrastructure Manager",
		Long: `DSPy Agent - Unified CLI for managing DSPy agent infrastructure.

This tool manages all infrastructure services (containers, orchestration, etc.)
required for running the DSPy agent, providing a simple single-command interface.`,
		Version: version,
	}

	rootCmd.AddCommand(startCmd())
	rootCmd.AddCommand(stopCmd())
	rootCmd.AddCommand(statusCmd())
	rootCmd.AddCommand(logsCmd())
	rootCmd.AddCommand(configCmd())

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func startCmd() *cobra.Command {
	var (
		workspace string
		gpu       bool
		services  []string
		daemon    bool
	)

	cmd := &cobra.Command{
		Use:   "start",
		Short: "Start all infrastructure services",
		Long: `Start all required infrastructure services for the DSPy agent.

This includes:
  - Environment Manager (container lifecycle)
  - Orchestrator (task scheduling)
  - RedDB (database)
  - Redis (cache)
  - InferMesh (inference services)
  - Ollama (optional, local LLM)

Services are started with dependency resolution and health checking.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runStart(workspace, gpu, services, daemon)
		},
	}

	cmd.Flags().StringVar(&workspace, "workspace", ".", "Workspace directory")
	cmd.Flags().BoolVar(&gpu, "gpu", false, "Enable GPU services")
	cmd.Flags().StringSliceVar(&services, "services", []string{}, "Specific services to start (empty = all)")
	cmd.Flags().BoolVar(&daemon, "daemon", false, "Run in daemon mode")

	return cmd
}

func stopCmd() *cobra.Command {
	var (
		timeout int
		force   bool
	)

	cmd := &cobra.Command{
		Use:   "stop",
		Short: "Stop all infrastructure services",
		Long:  `Gracefully stop all running infrastructure services.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runStop(timeout, force)
		},
	}

	cmd.Flags().IntVar(&timeout, "timeout", 30, "Shutdown timeout in seconds")
	cmd.Flags().BoolVar(&force, "force", false, "Force stop services")

	return cmd
}

func statusCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show status of all services",
		Long:  `Display the current status of all infrastructure services.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runStatus()
		},
	}

	return cmd
}

func logsCmd() *cobra.Command{
	var (
		follow  bool
		tail    int
		service string
	)

	cmd := &cobra.Command{
		Use:   "logs",
		Short: "Show logs from services",
		Long:  `Display logs from infrastructure services.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runLogs(service, follow, tail)
		},
	}

	cmd.Flags().BoolVarP(&follow, "follow", "f", false, "Follow log output")
	cmd.Flags().IntVarP(&tail, "tail", "n", 100, "Number of lines to show")
	cmd.Flags().StringVar(&service, "service", "", "Specific service (empty = all)")

	return cmd
}

func configCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Manage configuration",
		Long:  `View and modify DSPy agent configuration.`,
	}

	cmd.AddCommand(&cobra.Command{
		Use:   "show",
		Short: "Show current configuration",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runConfigShow()
		},
	})

	cmd.AddCommand(&cobra.Command{
		Use:   "init",
		Short: "Initialize configuration",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runConfigInit()
		},
	})

	return cmd
}

