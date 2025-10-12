package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/pelletier/go-toml/v2"
)

type Config struct {
	Workspace          string            `toml:"workspace"`
	OrchestratorAddr   string            `toml:"orchestrator_addr"`
	EnvManagerAddr     string            `toml:"env_manager_addr"`
	GPU                bool              `toml:"gpu"`
	Services           map[string]bool   `toml:"services"`
	Ports              map[string]int    `toml:"ports"`
}

func defaultConfig() *Config {
	return &Config{
		Workspace:        ".",
		OrchestratorAddr: "localhost:9098",
		EnvManagerAddr:   "localhost:50100",
		GPU:              false,
		Services: map[string]bool{
			"reddb":            true,
			"redis":            true,
			"infermesh-router": true,
			"ollama":           false,
		},
		Ports: map[string]int{
			"reddb":            8080,
			"redis":            6379,
			"infermesh-router": 19000,
			"ollama":           11434,
			"orchestrator":     9098,
			"env_manager":      50100,
		},
	}
}

func configPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ".dspy-agent/config.toml"
	}
	return filepath.Join(home, ".dspy-agent", "config.toml")
}

func loadConfig() (*Config, error) {
	path := configPath()
	
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return defaultConfig(), nil
		}
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	var config Config
	if err := toml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	return &config, nil
}

func saveConfig(config *Config) error {
	path := configPath()
	
	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	data, err := toml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}

	return nil
}

func runConfigShow() error {
	config, err := loadConfig()
	if err != nil {
		return err
	}

	fmt.Println("DSPy Agent Configuration")
	fmt.Println("========================\n")
	
	fmt.Printf("Workspace:        %s\n", config.Workspace)
	fmt.Printf("Orchestrator:     %s\n", config.OrchestratorAddr)
	fmt.Printf("Env Manager:      %s\n", config.EnvManagerAddr)
	fmt.Printf("GPU Enabled:      %v\n", config.GPU)
	
	fmt.Println("\nEnabled Services:")
	for service, enabled := range config.Services {
		status := "❌"
		if enabled {
			status = "✓"
		}
		fmt.Printf("  %s %s\n", status, service)
	}

	fmt.Println("\nPorts:")
	for service, port := range config.Ports {
		fmt.Printf("  %-20s %d\n", service+":", port)
	}

	fmt.Printf("\nConfig file: %s\n", configPath())
	
	return nil
}

func runConfigInit() error {
	path := configPath()
	
	// Check if config already exists
	if _, err := os.Stat(path); err == nil {
		fmt.Printf("Configuration already exists at: %s\n", path)
		fmt.Print("Overwrite? (y/N): ")
		
		var response string
		fmt.Scanln(&response)
		if response != "y" && response != "Y" {
			fmt.Println("Aborted")
			return nil
		}
	}

	// Create default config
	config := defaultConfig()
	
	if err := saveConfig(config); err != nil {
		return err
	}

	fmt.Printf("✓ Configuration initialized at: %s\n", path)
	fmt.Println("\nEdit the configuration file to customize settings.")
	
	return nil
}

