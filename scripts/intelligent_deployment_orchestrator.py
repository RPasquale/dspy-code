#!/usr/bin/env python3
"""
Intelligent Deployment Orchestrator for DSPy Agent
Provides smart deployment management with environment detection, health checks, and rollback capabilities.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import docker
import requests
import psutil


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    environment: str  # development, staging, production
    enable_auto_scaling: bool = True
    enable_monitoring: bool = True
    enable_intelligent_caching: bool = True
    enable_adaptive_learning: bool = True
    resource_limits: Dict[str, str] = None
    health_check_timeout: int = 300
    rollback_enabled: bool = True
    backup_enabled: bool = True


@dataclass
class DeploymentStatus:
    """Deployment status tracking"""
    environment: str
    status: str  # pending, deploying, deployed, failed, rolled_back
    start_time: datetime
    end_time: Optional[datetime] = None
    services: Dict[str, str] = None
    health_checks: Dict[str, bool] = None
    performance_metrics: Dict[str, Any] = None
    error_message: Optional[str] = None


class IntelligentDeploymentOrchestrator:
    """Intelligent deployment orchestrator with advanced capabilities."""
    
    def __init__(self, workspace: Path, config: DeploymentConfig):
        self.workspace = Path(workspace)
        self.config = config
        self.log_dir = self.workspace / "logs" / "deployment"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.error(f"Could not initialize Docker client: {e}")
            self.docker_client = None
        
        # Deployment state
        self.current_deployment: Optional[DeploymentStatus] = None
        self.deployment_history: List[DeploymentStatus] = []
        
        # Service endpoints for health checks
        self.service_endpoints = {
            'agent': 'http://localhost:8765',
            'dashboard': 'http://localhost:18081',
            'ollama': 'http://localhost:11435',
            'kafka': 'localhost:9092',
            'infermesh': 'http://localhost:9000',
            'embed_worker': 'http://localhost:9101',
            'fastapi': 'http://localhost:8767',
            'reddb': 'http://localhost:8082'
        }
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        self.logger = logging.getLogger('deployment_orchestrator')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / "deployment.log")
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def detect_environment(self) -> str:
        """Detect deployment environment based on system characteristics."""
        try:
            # Check for production indicators
            if os.getenv('PRODUCTION', '').lower() in {'true', '1', 'yes'}:
                return 'production'
            
            # Check for staging indicators
            if os.getenv('STAGING', '').lower() in {'true', '1', 'yes'}:
                return 'staging'
            
            # Check system resources
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if cpu_count >= 8 and memory_gb >= 16:
                return 'production'
            elif cpu_count >= 4 and memory_gb >= 8:
                return 'staging'
            else:
                return 'development'
                
        except Exception as e:
            self.logger.warning(f"Error detecting environment: {e}")
            return 'development'
    
    def create_deployment_config(self) -> Dict[str, Any]:
        """Create deployment configuration based on environment."""
        config = {
            'environment': self.config.environment,
            'timestamp': datetime.now().isoformat(),
            'features': {
                'auto_scaling': self.config.enable_auto_scaling,
                'monitoring': self.config.enable_monitoring,
                'intelligent_caching': self.config.enable_intelligent_caching,
                'adaptive_learning': self.config.enable_adaptive_learning
            },
            'resources': self.config.resource_limits or {},
            'health_checks': {
                'timeout': self.config.health_check_timeout,
                'interval': 10,
                'retries': 30
            }
        }
        
        # Environment-specific configurations
        if self.config.environment == 'production':
            config['resources'].update({
                'cpu_limit': '4.0',
                'memory_limit': '8G',
                'replicas': 3
            })
            config['features']['auto_scaling'] = True
            config['features']['monitoring'] = True
            
        elif self.config.environment == 'staging':
            config['resources'].update({
                'cpu_limit': '2.0',
                'memory_limit': '4G',
                'replicas': 2
            })
            config['features']['auto_scaling'] = True
            config['features']['monitoring'] = True
            
        else:  # development
            config['resources'].update({
                'cpu_limit': '1.0',
                'memory_limit': '2G',
                'replicas': 1
            })
            config['features']['auto_scaling'] = False
            config['features']['monitoring'] = True
        
        return config
    
    async def deploy_stack(self) -> DeploymentStatus:
        """Deploy the DSPy Agent stack with intelligent orchestration."""
        self.logger.info(f"Starting deployment to {self.config.environment} environment...")
        
        # Create deployment status
        self.current_deployment = DeploymentStatus(
            environment=self.config.environment,
            status='deploying',
            start_time=datetime.now(),
            services={},
            health_checks={}
        )
        
        try:
            # Step 1: Pre-deployment checks
            await self.pre_deployment_checks()
            
            # Step 2: Create deployment configuration
            deployment_config = self.create_deployment_config()
            config_file = self.workspace / '.env.deployment'
            with open(config_file, 'w') as f:
                json.dump(deployment_config, f, indent=2)
            
            # Step 3: Deploy services
            await self.deploy_services()
            
            # Step 4: Health checks
            await self.perform_health_checks()
            
            # Step 5: Performance validation
            await self.validate_performance()
            
            # Step 6: Enable advanced features
            if self.config.enable_auto_scaling or self.config.enable_monitoring:
                await self.enable_advanced_features()
            
            # Mark deployment as successful
            self.current_deployment.status = 'deployed'
            self.current_deployment.end_time = datetime.now()
            
            self.logger.info("Deployment completed successfully")
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            self.current_deployment.status = 'failed'
            self.current_deployment.error_message = str(e)
            self.current_deployment.end_time = datetime.now()
            
            # Attempt rollback if enabled
            if self.config.rollback_enabled:
                await self.rollback_deployment()
            
            raise
        
        finally:
            # Add to deployment history
            self.deployment_history.append(self.current_deployment)
            
        return self.current_deployment
    
    async def pre_deployment_checks(self):
        """Perform pre-deployment validation checks."""
        self.logger.info("Running pre-deployment checks...")
        
        # Check system resources
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if cpu_count < 2:
            raise Exception(f"Insufficient CPU cores: {cpu_count} (minimum: 2)")
        
        if memory_gb < 4:
            raise Exception(f"Insufficient memory: {memory_gb:.1f}GB (minimum: 4GB)")
        
        # Check Docker availability
        if not self.docker_client:
            raise Exception("Docker is not available")
        
        # Check required ports
        required_ports = [8765, 18081, 11435, 9092, 9000, 9101, 8767, 8082]
        for port in required_ports:
            if self._is_port_in_use(port):
                self.logger.warning(f"Port {port} is already in use")
        
        self.logger.info("Pre-deployment checks passed")
    
    async def deploy_services(self):
        """Deploy Docker services."""
        self.logger.info("Deploying Docker services...")
        
        try:
            # Change to Docker directory
            docker_dir = self.workspace / "docker" / "lightweight"
            os.chdir(docker_dir)
            
            # Set environment variables
            env = os.environ.copy()
            env.update({
                'DEPLOYMENT_ENV': self.config.environment,
                'DSPY_ENABLE_AUTO_SCALING': str(self.config.enable_auto_scaling).lower(),
                'DSPY_PERFORMANCE_MODE': 'optimized',
                'DSPY_INTELLIGENT_CACHING': str(self.config.enable_intelligent_caching).lower(),
                'DSPY_ADAPTIVE_LEARNING': str(self.config.enable_adaptive_learning).lower()
            })
            
            # Build and start services
            subprocess.run(['docker', 'compose', 'build'], check=True, env=env)
            subprocess.run(['docker', 'compose', 'up', '-d'], check=True, env=env)
            
            # Wait for services to start
            await asyncio.sleep(30)
            
            self.logger.info("Docker services deployed successfully")
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to deploy Docker services: {e}")
        except Exception as e:
            raise Exception(f"Error deploying services: {e}")
    
    async def perform_health_checks(self):
        """Perform comprehensive health checks."""
        self.logger.info("Performing health checks...")
        
        health_check_start = time.time()
        timeout = self.config.health_check_timeout
        
        while time.time() - health_check_start < timeout:
            all_healthy = True
            
            for service_name, endpoint in self.service_endpoints.items():
                try:
                    if endpoint.startswith('http'):
                        response = requests.get(f"{endpoint}/health", timeout=5)
                        is_healthy = response.status_code == 200
                    else:
                        # For non-HTTP services, check if container is running
                        is_healthy = self._is_service_running(service_name)
                    
                    self.current_deployment.health_checks[service_name] = is_healthy
                    
                    if not is_healthy:
                        all_healthy = False
                        self.logger.warning(f"Health check failed for {service_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Health check error for {service_name}: {e}")
                    self.current_deployment.health_checks[service_name] = False
                    all_healthy = False
            
            if all_healthy:
                self.logger.info("All health checks passed")
                return
            
            await asyncio.sleep(10)
        
        # If we get here, health checks timed out
        failed_services = [name for name, healthy in self.current_deployment.health_checks.items() if not healthy]
        raise Exception(f"Health checks timed out. Failed services: {', '.join(failed_services)}")
    
    async def validate_performance(self):
        """Validate deployment performance."""
        self.logger.info("Validating deployment performance...")
        
        try:
            # Collect performance metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            # Check if performance is within acceptable limits
            if cpu_usage > 90:
                raise Exception(f"High CPU usage detected: {cpu_usage:.1f}%")
            
            if memory_usage > 90:
                raise Exception(f"High memory usage detected: {memory_usage:.1f}%")
            
            # Store performance metrics
            self.current_deployment.performance_metrics = {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Performance validation passed (CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%)")
            
        except Exception as e:
            raise Exception(f"Performance validation failed: {e}")
    
    async def enable_advanced_features(self):
        """Enable advanced features if configured."""
        self.logger.info("Enabling advanced features...")
        
        try:
            if self.config.enable_auto_scaling:
                self.logger.info("Starting auto-scaler...")
                # Start auto-scaler service
                subprocess.run(['docker', 'compose', 'up', '-d', 'auto-scaler'], check=True)
            
            if self.config.enable_monitoring:
                self.logger.info("Starting performance monitoring...")
                # Start performance monitoring
                subprocess.Popen([
                    'python3', '-c',
                    f"""
from dspy_agent.monitor.performance_monitor import PerformanceMonitor
import asyncio
monitor = PerformanceMonitor('{self.workspace}')
monitor.start_monitoring()
"""
                ])
            
            self.logger.info("Advanced features enabled")
            
        except Exception as e:
            self.logger.warning(f"Error enabling advanced features: {e}")
    
    async def rollback_deployment(self):
        """Rollback the deployment to previous state."""
        self.logger.info("Rolling back deployment...")
        
        try:
            # Stop current services
            subprocess.run(['docker', 'compose', 'down'], check=True)
            
            # Restore from backup if available
            if self.config.backup_enabled:
                await self.restore_from_backup()
            
            self.current_deployment.status = 'rolled_back'
            self.logger.info("Rollback completed")
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
    
    async def restore_from_backup(self):
        """Restore from backup if available."""
        # This would implement backup restoration logic
        self.logger.info("Restoring from backup...")
        # Implementation would depend on backup strategy
        pass
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return False
        except OSError:
            return True
    
    def _is_service_running(self, service_name: str) -> bool:
        """Check if a Docker service is running."""
        try:
            if not self.docker_client:
                return False
            
            containers = self.docker_client.containers.list(filters={'name': service_name})
            return len(containers) > 0 and containers[0].status == 'running'
        except Exception:
            return False
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        if not self.current_deployment:
            return {'status': 'no_deployment'}
        
        return {
            'environment': self.current_deployment.environment,
            'status': self.current_deployment.status,
            'start_time': self.current_deployment.start_time.isoformat(),
            'end_time': self.current_deployment.end_time.isoformat() if self.current_deployment.end_time else None,
            'services': self.current_deployment.services,
            'health_checks': self.current_deployment.health_checks,
            'performance_metrics': self.current_deployment.performance_metrics,
            'error_message': self.current_deployment.error_message
        }
    
    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """Get deployment history."""
        return [
            {
                'environment': d.environment,
                'status': d.status,
                'start_time': d.start_time.isoformat(),
                'end_time': d.end_time.isoformat() if d.end_time else None,
                'duration': (d.end_time - d.start_time).total_seconds() if d.end_time else None,
                'error_message': d.error_message
            }
            for d in self.deployment_history
        ]


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Intelligent DSPy Agent Deployment Orchestrator")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace directory")
    parser.add_argument("--environment", choices=['development', 'staging', 'production'], 
                       default='development', help="Deployment environment")
    parser.add_argument("--no-auto-scaling", action="store_true", help="Disable auto-scaling")
    parser.add_argument("--no-monitoring", action="store_true", help="Disable monitoring")
    parser.add_argument("--no-caching", action="store_true", help="Disable intelligent caching")
    parser.add_argument("--no-learning", action="store_true", help="Disable adaptive learning")
    parser.add_argument("--status", action="store_true", help="Show deployment status")
    parser.add_argument("--history", action="store_true", help="Show deployment history")
    
    args = parser.parse_args()
    
    # Create deployment configuration
    config = DeploymentConfig(
        environment=args.environment,
        enable_auto_scaling=not args.no_auto_scaling,
        enable_monitoring=not args.no_monitoring,
        enable_intelligent_caching=not args.no_caching,
        enable_adaptive_learning=not args.no_learning
    )
    
    # Create orchestrator
    orchestrator = IntelligentDeploymentOrchestrator(args.workspace, config)
    
    if args.status:
        status = orchestrator.get_deployment_status()
        print(json.dumps(status, indent=2))
        return
    
    if args.history:
        history = orchestrator.get_deployment_history()
        print(json.dumps(history, indent=2))
        return
    
    # Run deployment
    try:
        deployment = await orchestrator.deploy_stack()
        print(f"Deployment completed: {deployment.status}")
        
        if deployment.status == 'deployed':
            print("✅ Deployment successful!")
            print(f"Environment: {deployment.environment}")
            print(f"Duration: {(deployment.end_time - deployment.start_time).total_seconds():.1f}s")
        else:
            print("❌ Deployment failed!")
            if deployment.error_message:
                print(f"Error: {deployment.error_message}")
    
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
