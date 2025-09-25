"""
Intelligent Auto-Scaling System for DSPy Agent Stack
Provides dynamic resource scaling based on performance metrics and workload patterns.
"""

import asyncio
import json
import logging
import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import psutil
import requests
import docker
from collections import deque, defaultdict


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning"""
    action: str  # scale_up, scale_down, no_action
    service: str
    current_replicas: int
    target_replicas: int
    reason: str
    confidence: float
    timestamp: datetime
    metrics: Dict[str, Any] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for scaling decisions"""
    cpu_usage: float
    memory_usage: float
    response_time: float
    error_rate: float
    queue_depth: int
    throughput: float
    timestamp: datetime


class AutoScaler:
    """Intelligent auto-scaler for DSPy agent services."""
    
    def __init__(self, workspace: Path, kafka_bootstrap: str = "kafka:9092", reddb_url: str = "http://reddb:8080"):
        self.workspace = Path(workspace)
        self.kafka_bootstrap = kafka_bootstrap
        self.reddb_url = reddb_url
        
        # Setup logging
        self.logger = logging.getLogger('auto_scaler')
        self.logger.setLevel(logging.INFO)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.error(f"Could not initialize Docker client: {e}")
            self.docker_client = None
        
        # Scaling configuration
        self.config = {
            'cpu_threshold_up': float(os.getenv('AUTO_SCALER_CPU_THRESHOLD', '80')),
            'cpu_threshold_down': 40.0,
            'memory_threshold_up': float(os.getenv('AUTO_SCALER_MEMORY_THRESHOLD', '85')),
            'memory_threshold_down': 50.0,
            'response_time_threshold': 5.0,
            'error_rate_threshold': 5.0,
            'min_replicas': 1,
            'max_replicas': 5,
            'scale_up_cooldown': 300,  # 5 minutes
            'scale_down_cooldown': 600,  # 10 minutes
            'monitoring_interval': int(os.getenv('AUTO_SCALER_INTERVAL', '30'))
        }
        
        # Service configurations
        self.services = {
            'dspy-worker': {
                'base_replicas': 1,
                'max_replicas': 3,
                'cpu_weight': 0.4,
                'memory_weight': 0.3,
                'response_weight': 0.3
            },
            'embed-worker': {
                'base_replicas': 1,
                'max_replicas': 2,
                'cpu_weight': 0.5,
                'memory_weight': 0.5,
                'response_weight': 0.0
            }
        }
        
        # Performance history
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))
        self.scaling_history = deque(maxlen=50)
        self.last_scaling = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Initialize data manager
        try:
            from ..db import get_enhanced_data_manager, create_log_entry, Environment
            self.data_manager = get_enhanced_data_manager()
        except Exception as e:
            self.logger.warning(f"Could not initialize data manager: {e}")
            self.data_manager = None
    
    async def collect_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Collect performance metrics for all services."""
        metrics = {}
        
        try:
            # System-wide metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Service-specific metrics
            for service_name in self.services.keys():
                try:
                    # Get container metrics
                    container = self.docker_client.containers.get(service_name) if self.docker_client else None
                    
                    if container:
                        stats = container.stats(stream=False)
                        
                        # Calculate container CPU usage
                        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                                   stats['precpu_stats']['cpu_usage']['total_usage']
                        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                      stats['precpu_stats']['system_cpu_usage']
                        container_cpu = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0
                        
                        # Calculate container memory usage
                        memory_usage_bytes = stats['memory_stats']['usage']
                        memory_limit_bytes = stats['memory_stats']['limit']
                        container_memory = (memory_usage_bytes / memory_limit_bytes) * 100.0 if memory_limit_bytes > 0 else 0
                        
                        # Get response time (simplified)
                        response_time = await self._get_service_response_time(service_name)
                        
                        # Get error rate (simplified)
                        error_rate = await self._get_service_error_rate(service_name)
                        
                        # Get queue depth (for worker services)
                        queue_depth = await self._get_queue_depth(service_name)
                        
                        # Calculate throughput (simplified)
                        throughput = await self._get_service_throughput(service_name)
                        
                    else:
                        # Fallback to system metrics
                        container_cpu = cpu_usage
                        container_memory = memory_usage
                        response_time = 1.0
                        error_rate = 0.0
                        queue_depth = 0
                        throughput = 1.0
                    
                    metrics[service_name] = PerformanceMetrics(
                        cpu_usage=container_cpu,
                        memory_usage=container_memory,
                        response_time=response_time,
                        error_rate=error_rate,
                        queue_depth=queue_depth,
                        throughput=throughput,
                        timestamp=datetime.now()
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Error collecting metrics for {service_name}: {e}")
                    # Use system metrics as fallback
                    metrics[service_name] = PerformanceMetrics(
                        cpu_usage=cpu_usage,
                        memory_usage=memory_usage,
                        response_time=1.0,
                        error_rate=0.0,
                        queue_depth=0,
                        throughput=1.0,
                        timestamp=datetime.now()
                    )
        
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    async def _get_service_response_time(self, service_name: str) -> float:
        """Get service response time."""
        try:
            # This would be implemented based on service-specific health checks
            # For now, return a simplified calculation
            return 1.0
        except Exception:
            return 1.0
    
    async def _get_service_error_rate(self, service_name: str) -> float:
        """Get service error rate."""
        try:
            # This would analyze service logs for error rates
            # For now, return 0
            return 0.0
        except Exception:
            return 0.0
    
    async def _get_queue_depth(self, service_name: str) -> int:
        """Get queue depth for worker services."""
        try:
            # This would check Kafka topic depths or other queue metrics
            # For now, return 0
            return 0
        except Exception:
            return 0
    
    async def _get_service_throughput(self, service_name: str) -> float:
        """Get service throughput."""
        try:
            # This would calculate requests per second or similar
            # For now, return 1.0
            return 1.0
        except Exception:
            return 1.0
    
    def analyze_scaling_need(self, service_name: str, metrics: PerformanceMetrics) -> ScalingDecision:
        """Analyze if scaling is needed for a service."""
        service_config = self.services.get(service_name, {})
        
        # Get current replica count
        current_replicas = self._get_current_replicas(service_name)
        
        # Calculate weighted score
        cpu_score = metrics.cpu_usage * service_config.get('cpu_weight', 0.4)
        memory_score = metrics.memory_usage * service_config.get('memory_weight', 0.3)
        response_score = min(metrics.response_time / self.config['response_time_threshold'], 1.0) * service_config.get('response_weight', 0.3)
        
        weighted_score = cpu_score + memory_score + response_score
        
        # Determine scaling action
        action = "no_action"
        target_replicas = current_replicas
        reason = "No scaling needed"
        confidence = 0.5
        
        # Check for scale up conditions
        if (metrics.cpu_usage > self.config['cpu_threshold_up'] or 
            metrics.memory_usage > self.config['memory_threshold_up'] or
            metrics.response_time > self.config['response_time_threshold'] or
            metrics.error_rate > self.config['error_rate_threshold']):
            
            if current_replicas < service_config.get('max_replicas', 3):
                action = "scale_up"
                target_replicas = min(current_replicas + 1, service_config.get('max_replicas', 3))
                reason = f"High resource usage: CPU={metrics.cpu_usage:.1f}%, Memory={metrics.memory_usage:.1f}%"
                confidence = 0.8
        
        # Check for scale down conditions
        elif (metrics.cpu_usage < self.config['cpu_threshold_down'] and 
              metrics.memory_usage < self.config['memory_threshold_down'] and
              metrics.response_time < self.config['response_time_threshold'] * 0.5 and
              metrics.error_rate < self.config['error_rate_threshold'] * 0.5):
            
            if current_replicas > service_config.get('base_replicas', 1):
                # Check cooldown
                last_scale = self.last_scaling.get(service_name, 0)
                if time.time() - last_scale > self.config['scale_down_cooldown']:
                    action = "scale_down"
                    target_replicas = max(current_replicas - 1, service_config.get('base_replicas', 1))
                    reason = f"Low resource usage: CPU={metrics.cpu_usage:.1f}%, Memory={metrics.memory_usage:.1f}%"
                    confidence = 0.7
        
        return ScalingDecision(
            action=action,
            service=service_name,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            reason=reason,
            confidence=confidence,
            timestamp=datetime.now(),
            metrics={
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'response_time': metrics.response_time,
                'error_rate': metrics.error_rate,
                'weighted_score': weighted_score
            }
        )
    
    def _get_current_replicas(self, service_name: str) -> int:
        """Get current replica count for a service."""
        try:
            if self.docker_client:
                # Count running containers with the service name
                containers = self.docker_client.containers.list(filters={'name': service_name})
                return len([c for c in containers if c.status == 'running'])
        except Exception as e:
            self.logger.warning(f"Error getting replica count for {service_name}: {e}")
        return 1
    
    async def execute_scaling(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision."""
        if decision.action == "no_action":
            return True
        
        try:
            self.logger.info(f"Executing scaling: {decision.action} for {decision.service} "
                           f"({decision.current_replicas} -> {decision.target_replicas})")
            
            # In a real implementation, this would:
            # 1. Update Docker Compose or Kubernetes manifests
            # 2. Restart services with new replica counts
            # 3. Update load balancer configurations
            
            # For now, log the decision
            self.logger.info(f"Scaling decision: {decision.reason}")
            
            # Record scaling action
            self.last_scaling[decision.service] = time.time()
            self.scaling_history.append(decision)
            
            # Log to data manager if available
            if self.data_manager:
                try:
                    from ..db import create_log_entry, Environment
                    scaling_log = create_log_entry(
                        level="INFO",
                        source="monitor.auto_scaler",
                        message=f"Scaling {decision.action}: {decision.service} "
                               f"({decision.current_replicas} -> {decision.target_replicas})",
                        context={
                            "service": decision.service,
                            "action": decision.action,
                            "current_replicas": decision.current_replicas,
                            "target_replicas": decision.target_replicas,
                            "reason": decision.reason,
                            "confidence": decision.confidence,
                            "metrics": decision.metrics
                        },
                        environment=Environment.DEVELOPMENT
                    )
                    self.data_manager.log(scaling_log)
                except Exception as e:
                    self.logger.warning(f"Could not log scaling action: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing scaling for {decision.service}: {e}")
            return False
    
    async def monitoring_loop(self):
        """Main monitoring and scaling loop."""
        self.logger.info("Starting auto-scaler monitoring loop...")
        
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                
                # Analyze each service
                for service_name, service_metrics in metrics.items():
                    # Store metrics in history
                    self.metrics_history[service_name].append(service_metrics)
                    
                    # Analyze scaling need
                    decision = self.analyze_scaling_need(service_name, service_metrics)
                    
                    # Execute scaling if needed
                    if decision.action != "no_action":
                        await self.execute_scaling(decision)
                
                # Wait for next interval
                await asyncio.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    def start_monitoring(self):
        """Start the auto-scaler monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Auto-scaler already running")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._run_monitoring_async)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        self.logger.info("Auto-scaler started")
    
    def _run_monitoring_async(self):
        """Run monitoring in async context."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.monitoring_loop())
    
    def stop_monitoring(self):
        """Stop the auto-scaler monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Auto-scaler stopped")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and history."""
        return {
            'monitoring_active': self.monitoring_active,
            'services': {
                name: {
                    'current_replicas': self._get_current_replicas(name),
                    'max_replicas': config.get('max_replicas', 3),
                    'base_replicas': config.get('base_replicas', 1),
                    'last_scaling': self.last_scaling.get(name, 0)
                }
                for name, config in self.services.items()
            },
            'recent_scaling': [
                {
                    'service': d.service,
                    'action': d.action,
                    'replicas': f"{d.current_replicas} -> {d.target_replicas}",
                    'reason': d.reason,
                    'timestamp': d.timestamp.isoformat()
                }
                for d in list(self.scaling_history)[-10:]  # Last 10 scaling actions
            ],
            'config': self.config
        }


def main():
    """Main entry point for auto-scaler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DSPy Agent Auto-Scaler")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace directory")
    parser.add_argument("--kafka", default="kafka:9092", help="Kafka bootstrap servers")
    parser.add_argument("--reddb", default="http://reddb:8080", help="RedDB URL")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start auto-scaler
    scaler = AutoScaler(args.workspace, args.kafka, args.reddb)
    
    if args.daemon:
        scaler.start_monitoring()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scaler.stop_monitoring()
    else:
        # Run single check
        metrics = asyncio.run(scaler.collect_metrics())
        print("Auto-scaler metrics:")
        for service, metric in metrics.items():
            print(f"  {service}: CPU={metric.cpu_usage:.1f}%, Memory={metric.memory_usage:.1f}%")


if __name__ == "__main__":
    main()
