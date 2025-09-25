#!/usr/bin/env python3
"""
Advanced Environment Manager for DSPy Agent
Provides intelligent environment detection, configuration, and management.
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import psutil
import docker


@dataclass
class EnvironmentInfo:
    """Environment information"""
    os_name: str
    os_version: str
    python_version: str
    docker_version: Optional[str]
    cpu_count: int
    memory_gb: float
    disk_free_gb: float
    architecture: str
    environment_type: str  # development, staging, production
    capabilities: List[str]


@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    name: str
    environment_type: str
    resource_limits: Dict[str, Any]
    feature_flags: Dict[str, bool]
    service_configs: Dict[str, Dict[str, Any]]
    monitoring_config: Dict[str, Any]
    scaling_config: Dict[str, Any]


class EnvironmentManager:
    """Advanced environment manager with intelligent configuration."""
    
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.config_dir = self.workspace / "config" / "environments"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.warning(f"Could not initialize Docker client: {e}")
            self.docker_client = None
    
    def setup_logging(self):
        """Setup logging."""
        self.logger = logging.getLogger('environment_manager')
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def detect_environment(self) -> EnvironmentInfo:
        """Detect and analyze the current environment."""
        self.logger.info("Detecting environment...")
        
        # System information
        os_name = platform.system()
        os_version = platform.release()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Docker information
        docker_version = None
        if self.docker_client:
            try:
                docker_info = self.docker_client.version()
                docker_version = docker_info.get('Version', 'Unknown')
            except Exception as e:
                self.logger.warning(f"Could not get Docker version: {e}")
        
        # Resource information
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        disk_free_gb = psutil.disk_usage(str(self.workspace)).free / (1024**3)
        architecture = platform.machine()
        
        # Determine environment type
        environment_type = self._determine_environment_type(cpu_count, memory_gb)
        
        # Determine capabilities
        capabilities = self._determine_capabilities(cpu_count, memory_gb, docker_version)
        
        return EnvironmentInfo(
            os_name=os_name,
            os_version=os_version,
            python_version=python_version,
            docker_version=docker_version,
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            disk_free_gb=disk_free_gb,
            architecture=architecture,
            environment_type=environment_type,
            capabilities=capabilities
        )
    
    def _determine_environment_type(self, cpu_count: int, memory_gb: float) -> str:
        """Determine environment type based on resources."""
        # Check environment variables first
        if os.getenv('PRODUCTION', '').lower() in {'true', '1', 'yes'}:
            return 'production'
        if os.getenv('STAGING', '').lower() in {'true', '1', 'yes'}:
            return 'staging'
        
        # Determine based on resources
        if cpu_count >= 8 and memory_gb >= 16:
            return 'production'
        elif cpu_count >= 4 and memory_gb >= 8:
            return 'staging'
        else:
            return 'development'
    
    def _determine_capabilities(self, cpu_count: int, memory_gb: float, docker_version: Optional[str]) -> List[str]:
        """Determine system capabilities."""
        capabilities = []
        
        # Basic capabilities
        capabilities.append('python')
        capabilities.append('file_operations')
        
        # Docker capabilities
        if docker_version:
            capabilities.append('docker')
            capabilities.append('containerization')
        
        # Resource-based capabilities
        if cpu_count >= 4:
            capabilities.append('parallel_processing')
            capabilities.append('multi_service')
        
        if memory_gb >= 8:
            capabilities.append('large_models')
            capabilities.append('caching')
        
        if memory_gb >= 16:
            capabilities.append('advanced_ml')
            capabilities.append('vector_processing')
        
        # Advanced capabilities
        if cpu_count >= 8 and memory_gb >= 16:
            capabilities.append('production_ready')
            capabilities.append('auto_scaling')
            capabilities.append('high_performance')
        
        return capabilities
    
    def create_environment_config(self, env_info: EnvironmentInfo) -> EnvironmentConfig:
        """Create environment-specific configuration."""
        self.logger.info(f"Creating configuration for {env_info.environment_type} environment...")
        
        # Base configuration
        config = EnvironmentConfig(
            name=f"dspy-{env_info.environment_type}",
            environment_type=env_info.environment_type,
            resource_limits={},
            feature_flags={},
            service_configs={},
            monitoring_config={},
            scaling_config={}
        )
        
        # Environment-specific configurations
        if env_info.environment_type == 'production':
            config.resource_limits = {
                'cpu_limit': '4.0',
                'memory_limit': '8G',
                'replicas': 3,
                'max_replicas': 10
            }
            config.feature_flags = {
                'auto_scaling': True,
                'monitoring': True,
                'intelligent_caching': True,
                'adaptive_learning': True,
                'performance_optimization': True,
                'anomaly_detection': True
            }
            config.service_configs = {
                'dspy-agent': {
                    'replicas': 2,
                    'resources': {'cpu': '2.0', 'memory': '4G'},
                    'restart_policy': 'unless-stopped'
                },
                'ollama': {
                    'replicas': 1,
                    'resources': {'cpu': '4.0', 'memory': '8G'},
                    'restart_policy': 'unless-stopped'
                }
            }
            config.monitoring_config = {
                'interval': 30,
                'cpu_threshold': 80,
                'memory_threshold': 85,
                'response_time_threshold': 5.0
            }
            config.scaling_config = {
                'enabled': True,
                'min_replicas': 1,
                'max_replicas': 5,
                'scale_up_threshold': 80,
                'scale_down_threshold': 40
            }
            
        elif env_info.environment_type == 'staging':
            config.resource_limits = {
                'cpu_limit': '2.0',
                'memory_limit': '4G',
                'replicas': 2,
                'max_replicas': 5
            }
            config.feature_flags = {
                'auto_scaling': True,
                'monitoring': True,
                'intelligent_caching': True,
                'adaptive_learning': False,
                'performance_optimization': True,
                'anomaly_detection': True
            }
            config.service_configs = {
                'dspy-agent': {
                    'replicas': 1,
                    'resources': {'cpu': '1.0', 'memory': '2G'},
                    'restart_policy': 'unless-stopped'
                },
                'ollama': {
                    'replicas': 1,
                    'resources': {'cpu': '2.0', 'memory': '4G'},
                    'restart_policy': 'unless-stopped'
                }
            }
            config.monitoring_config = {
                'interval': 60,
                'cpu_threshold': 85,
                'memory_threshold': 90,
                'response_time_threshold': 10.0
            }
            config.scaling_config = {
                'enabled': True,
                'min_replicas': 1,
                'max_replicas': 3,
                'scale_up_threshold': 85,
                'scale_down_threshold': 50
            }
            
        else:  # development
            config.resource_limits = {
                'cpu_limit': '1.0',
                'memory_limit': '2G',
                'replicas': 1,
                'max_replicas': 2
            }
            config.feature_flags = {
                'auto_scaling': False,
                'monitoring': True,
                'intelligent_caching': False,
                'adaptive_learning': False,
                'performance_optimization': False,
                'anomaly_detection': False
            }
            config.service_configs = {
                'dspy-agent': {
                    'replicas': 1,
                    'resources': {'cpu': '0.5', 'memory': '1G'},
                    'restart_policy': 'on-failure'
                },
                'ollama': {
                    'replicas': 1,
                    'resources': {'cpu': '1.0', 'memory': '2G'},
                    'restart_policy': 'on-failure'
                }
            }
            config.monitoring_config = {
                'interval': 120,
                'cpu_threshold': 90,
                'memory_threshold': 95,
                'response_time_threshold': 30.0
            }
            config.scaling_config = {
                'enabled': False,
                'min_replicas': 1,
                'max_replicas': 1,
                'scale_up_threshold': 90,
                'scale_down_threshold': 70
            }
        
        return config
    
    def save_environment_config(self, config: EnvironmentConfig):
        """Save environment configuration to file."""
        config_file = self.config_dir / f"{config.name}.json"
        
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        self.logger.info(f"Environment configuration saved to {config_file}")
    
    def load_environment_config(self, name: str) -> Optional[EnvironmentConfig]:
        """Load environment configuration from file."""
        config_file = self.config_dir / f"{name}.json"
        
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            return EnvironmentConfig(**data)
        except Exception as e:
            self.logger.error(f"Error loading configuration {name}: {e}")
            return None
    
    def generate_docker_compose_config(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """Generate Docker Compose configuration based on environment."""
        compose_config = {
            'version': '3.8',
            'services': {},
            'volumes': {},
            'networks': {}
        }
        
        # Add services based on configuration
        for service_name, service_config in config.service_configs.items():
            compose_config['services'][service_name] = {
                'image': f'dspy-{service_name}:latest',
                'deploy': {
                    'replicas': service_config['replicas'],
                    'resources': {
                        'limits': service_config['resources']
                    },
                    'restart_policy': {
                        'condition': service_config['restart_policy']
                    }
                },
                'environment': {
                    'DEPLOYMENT_ENV': config.environment_type,
                    'DSPY_ENABLE_AUTO_SCALING': str(config.feature_flags.get('auto_scaling', False)).lower(),
                    'DSPY_PERFORMANCE_MODE': 'optimized' if config.feature_flags.get('performance_optimization', False) else 'standard',
                    'DSPY_INTELLIGENT_CACHING': str(config.feature_flags.get('intelligent_caching', False)).lower(),
                    'DSPY_ADAPTIVE_LEARNING': str(config.feature_flags.get('adaptive_learning', False)).lower()
                }
            }
        
        return compose_config
    
    def validate_environment(self, env_info: EnvironmentInfo, config: EnvironmentConfig) -> List[str]:
        """Validate environment against configuration requirements."""
        issues = []
        
        # Check resource requirements
        required_cpu = float(config.resource_limits.get('cpu_limit', '1.0'))
        required_memory = float(config.resource_limits.get('memory_limit', '2G').replace('G', ''))
        
        if env_info.cpu_count < required_cpu:
            issues.append(f"Insufficient CPU: {env_info.cpu_count} < {required_cpu}")
        
        if env_info.memory_gb < required_memory:
            issues.append(f"Insufficient memory: {env_info.memory_gb:.1f}GB < {required_memory}GB")
        
        # Check disk space
        if env_info.disk_free_gb < 10:
            issues.append(f"Low disk space: {env_info.disk_free_gb:.1f}GB (minimum: 10GB)")
        
        # Check Docker availability
        if config.feature_flags.get('auto_scaling', False) and not env_info.docker_version:
            issues.append("Auto-scaling requires Docker")
        
        return issues
    
    def get_environment_summary(self, env_info: EnvironmentInfo, config: EnvironmentConfig) -> Dict[str, Any]:
        """Get comprehensive environment summary."""
        validation_issues = self.validate_environment(env_info, config)
        
        return {
            'environment_info': asdict(env_info),
            'configuration': asdict(config),
            'validation': {
                'valid': len(validation_issues) == 0,
                'issues': validation_issues
            },
            'recommendations': self._get_recommendations(env_info, config),
            'capabilities': env_info.capabilities,
            'resource_utilization': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage(str(self.workspace)).percent
            }
        }
    
    def _get_recommendations(self, env_info: EnvironmentInfo, config: EnvironmentConfig) -> List[str]:
        """Get environment-specific recommendations."""
        recommendations = []
        
        # Resource recommendations
        if env_info.memory_gb < 8:
            recommendations.append("Consider upgrading memory for better performance")
        
        if env_info.cpu_count < 4:
            recommendations.append("Consider upgrading CPU for parallel processing")
        
        # Feature recommendations
        if env_info.environment_type == 'development' and env_info.memory_gb >= 8:
            recommendations.append("Consider enabling intelligent caching for better performance")
        
        if env_info.environment_type in ['staging', 'production'] and not config.feature_flags.get('monitoring', False):
            recommendations.append("Enable monitoring for production environments")
        
        # Docker recommendations
        if not env_info.docker_version:
            recommendations.append("Install Docker for containerized deployment")
        
        return recommendations


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DSPy Agent Environment Manager")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace directory")
    parser.add_argument("--detect", action="store_true", help="Detect current environment")
    parser.add_argument("--configure", action="store_true", help="Configure environment")
    parser.add_argument("--validate", action="store_true", help="Validate environment")
    parser.add_argument("--summary", action="store_true", help="Show environment summary")
    parser.add_argument("--save-config", type=str, help="Save configuration with name")
    parser.add_argument("--load-config", type=str, help="Load configuration by name")
    
    args = parser.parse_args()
    
    # Create environment manager
    manager = EnvironmentManager(args.workspace)
    
    if args.detect:
        env_info = manager.detect_environment()
        print(json.dumps(asdict(env_info), indent=2))
        return
    
    if args.configure:
        env_info = manager.detect_environment()
        config = manager.create_environment_config(env_info)
        manager.save_environment_config(config)
        print(f"Environment configured for {env_info.environment_type}")
        return
    
    if args.validate:
        env_info = manager.detect_environment()
        config = manager.create_environment_config(env_info)
        issues = manager.validate_environment(env_info, config)
        
        if issues:
            print("❌ Environment validation failed:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ Environment validation passed")
        return
    
    if args.summary:
        env_info = manager.detect_environment()
        config = manager.create_environment_config(env_info)
        summary = manager.get_environment_summary(env_info, config)
        print(json.dumps(summary, indent=2))
        return
    
    if args.save_config:
        env_info = manager.detect_environment()
        config = manager.create_environment_config(env_info)
        config.name = args.save_config
        manager.save_environment_config(config)
        print(f"Configuration saved as {args.save_config}")
        return
    
    if args.load_config:
        config = manager.load_environment_config(args.load_config)
        if config:
            print(json.dumps(asdict(config), indent=2))
        else:
            print(f"Configuration {args.load_config} not found")
        return
    
    # Default: show environment summary
    env_info = manager.detect_environment()
    config = manager.create_environment_config(env_info)
    summary = manager.get_environment_summary(env_info, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
