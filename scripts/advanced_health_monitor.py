#!/usr/bin/env python3
"""
Advanced Health Monitor for DSPy Agent Stack
Provides comprehensive health monitoring with intelligent analysis and recommendations.
"""

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import psutil
import requests
import docker
from collections import deque, defaultdict


@dataclass
class HealthMetric:
    """Health metric data"""
    name: str
    value: float
    unit: str
    threshold: float
    status: str  # healthy, warning, critical
    timestamp: datetime
    trend: str = "stable"  # improving, stable, degrading


@dataclass
class ServiceHealth:
    """Service health status"""
    name: str
    status: str  # healthy, warning, critical, unknown
    response_time: float
    error_rate: float
    uptime: float
    last_check: datetime
    issues: List[str] = None


@dataclass
class HealthReport:
    """Comprehensive health report"""
    timestamp: datetime
    overall_status: str
    system_metrics: Dict[str, HealthMetric]
    service_health: Dict[str, ServiceHealth]
    recommendations: List[str]
    alerts: List[str]
    performance_score: float


class AdvancedHealthMonitor:
    """Advanced health monitoring system with intelligent analysis."""
    
    def __init__(self, workspace: Path, monitoring_interval: int = 30):
        self.workspace = Path(workspace)
        self.monitoring_interval = monitoring_interval
        self.log_dir = self.workspace / "logs" / "health"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.warning(f"Could not initialize Docker client: {e}")
            self.docker_client = None
        
        # Service endpoints
        self.service_endpoints = {
            'agent': 'http://localhost:8765',
            'dashboard': 'http://localhost:18081',
            'ollama': 'http://localhost:11435',
            'infermesh': 'http://localhost:9000',
            'embed_worker': 'http://localhost:9101',
            'fastapi': 'http://localhost:8767',
            'reddb': 'http://localhost:8082'
        }
        
        # Health thresholds
        self.thresholds = {
            'cpu_usage': {'warning': 70, 'critical': 90},
            'memory_usage': {'warning': 80, 'critical': 95},
            'disk_usage': {'warning': 80, 'critical': 95},
            'response_time': {'warning': 5.0, 'critical': 10.0},
            'error_rate': {'warning': 5.0, 'critical': 10.0}
        }
        
        # Historical data
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))
        self.health_reports = deque(maxlen=50)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
    
    def setup_logging(self):
        """Setup logging."""
        self.logger = logging.getLogger('advanced_health_monitor')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / "health_monitor.log")
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
    
    async def collect_system_metrics(self) -> Dict[str, HealthMetric]:
        """Collect comprehensive system metrics."""
        metrics = {}
        
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            metrics['cpu_usage'] = HealthMetric(
                name='CPU Usage',
                value=cpu_usage,
                unit='%',
                threshold=self.thresholds['cpu_usage']['critical'],
                status=self._get_metric_status(cpu_usage, 'cpu_usage'),
                timestamp=datetime.now(),
                trend=self._calculate_trend('cpu_usage', cpu_usage)
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            metrics['memory_usage'] = HealthMetric(
                name='Memory Usage',
                value=memory_usage,
                unit='%',
                threshold=self.thresholds['memory_usage']['critical'],
                status=self._get_metric_status(memory_usage, 'memory_usage'),
                timestamp=datetime.now(),
                trend=self._calculate_trend('memory_usage', memory_usage)
            )
            
            # Disk metrics
            disk = psutil.disk_usage(str(self.workspace))
            disk_usage = disk.percent
            disk_free_gb = disk.free / (1024**3)
            
            metrics['disk_usage'] = HealthMetric(
                name='Disk Usage',
                value=disk_usage,
                unit='%',
                threshold=self.thresholds['disk_usage']['critical'],
                status=self._get_metric_status(disk_usage, 'disk_usage'),
                timestamp=datetime.now(),
                trend=self._calculate_trend('disk_usage', disk_usage)
            )
            
            # Network metrics
            net_io = psutil.net_io_counters()
            network_utilization = (net_io.bytes_sent + net_io.bytes_recv) / (1024**3)  # GB
            
            metrics['network_utilization'] = HealthMetric(
                name='Network Utilization',
                value=network_utilization,
                unit='GB',
                threshold=100.0,  # 100GB threshold
                status='healthy' if network_utilization < 100 else 'warning',
                timestamp=datetime.now(),
                trend=self._calculate_trend('network_utilization', network_utilization)
            )
            
            # Store metrics in history
            for name, metric in metrics.items():
                self.metrics_history[name].append(metric.value)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    async def check_service_health(self) -> Dict[str, ServiceHealth]:
        """Check health of all services."""
        service_health = {}
        
        for service_name, endpoint in self.service_endpoints.items():
            try:
                start_time = time.time()
                
                if endpoint.startswith('http'):
                    # HTTP service health check
                    response = requests.get(f"{endpoint}/health", timeout=5)
                    response_time = time.time() - start_time
                    
                    status = 'healthy' if response.status_code == 200 else 'critical'
                    error_rate = 0.0 if response.status_code == 200 else 100.0
                    
                else:
                    # Non-HTTP service (check Docker container)
                    if self.docker_client:
                        containers = self.docker_client.containers.list(filters={'name': service_name})
                        if containers and containers[0].status == 'running':
                            status = 'healthy'
                            response_time = 0.1
                            error_rate = 0.0
                        else:
                            status = 'critical'
                            response_time = 0.0
                            error_rate = 100.0
                    else:
                        status = 'unknown'
                        response_time = 0.0
                        error_rate = 0.0
                
                # Calculate uptime (simplified)
                uptime = 100.0 if status == 'healthy' else 0.0
                
                service_health[service_name] = ServiceHealth(
                    name=service_name,
                    status=status,
                    response_time=response_time,
                    error_rate=error_rate,
                    uptime=uptime,
                    last_check=datetime.now(),
                    issues=[] if status == 'healthy' else [f"Service {service_name} is {status}"]
                )
                
            except Exception as e:
                self.logger.warning(f"Error checking health for {service_name}: {e}")
                service_health[service_name] = ServiceHealth(
                    name=service_name,
                    status='unknown',
                    response_time=0.0,
                    error_rate=100.0,
                    uptime=0.0,
                    last_check=datetime.now(),
                    issues=[f"Health check failed: {e}"]
                )
        
        return service_health
    
    def _get_metric_status(self, value: float, metric_name: str) -> str:
        """Get metric status based on thresholds."""
        thresholds = self.thresholds.get(metric_name, {'warning': 70, 'critical': 90})
        
        if value >= thresholds['critical']:
            return 'critical'
        elif value >= thresholds['warning']:
            return 'warning'
        else:
            return 'healthy'
    
    def _calculate_trend(self, metric_name: str, current_value: float) -> str:
        """Calculate trend for a metric."""
        if metric_name not in self.metrics_history or len(self.metrics_history[metric_name]) < 5:
            return 'stable'
        
        recent_values = list(self.metrics_history[metric_name])[-5:]
        
        # Simple trend calculation
        if len(recent_values) >= 3:
            first_half = sum(recent_values[:len(recent_values)//2]) / (len(recent_values)//2)
            second_half = sum(recent_values[len(recent_values)//2:]) / (len(recent_values) - len(recent_values)//2)
            
            if second_half > first_half * 1.1:
                return 'degrading'
            elif second_half < first_half * 0.9:
                return 'improving'
            else:
                return 'stable'
        
        return 'stable'
    
    def generate_recommendations(self, system_metrics: Dict[str, HealthMetric], 
                               service_health: Dict[str, ServiceHealth]) -> List[str]:
        """Generate intelligent recommendations based on health data."""
        recommendations = []
        
        # System recommendations
        for metric_name, metric in system_metrics.items():
            if metric.status == 'critical':
                if metric_name == 'cpu_usage':
                    recommendations.append("Critical CPU usage detected. Consider scaling up or optimizing CPU-intensive operations.")
                elif metric_name == 'memory_usage':
                    recommendations.append("Critical memory usage detected. Consider increasing memory or optimizing memory usage.")
                elif metric_name == 'disk_usage':
                    recommendations.append("Critical disk usage detected. Consider cleaning up disk space or expanding storage.")
            
            elif metric.status == 'warning':
                if metric_name == 'cpu_usage':
                    recommendations.append("High CPU usage detected. Monitor for potential performance issues.")
                elif metric_name == 'memory_usage':
                    recommendations.append("High memory usage detected. Consider memory optimization.")
                elif metric_name == 'disk_usage':
                    recommendations.append("High disk usage detected. Consider disk cleanup.")
            
            # Trend-based recommendations
            if metric.trend == 'degrading':
                recommendations.append(f"{metric.name} is showing degrading trend. Monitor closely.")
            elif metric.trend == 'improving':
                recommendations.append(f"{metric.name} is showing improving trend. Good performance.")
        
        # Service recommendations
        for service_name, health in service_health.items():
            if health.status == 'critical':
                recommendations.append(f"Service {service_name} is critical. Immediate attention required.")
            elif health.status == 'warning':
                recommendations.append(f"Service {service_name} has issues. Monitor and investigate.")
            
            if health.response_time > self.thresholds['response_time']['warning']:
                recommendations.append(f"Service {service_name} has slow response time. Consider optimization.")
            
            if health.error_rate > self.thresholds['error_rate']['warning']:
                recommendations.append(f"Service {service_name} has high error rate. Investigate issues.")
        
        return recommendations
    
    def generate_alerts(self, system_metrics: Dict[str, HealthMetric], 
                       service_health: Dict[str, ServiceHealth]) -> List[str]:
        """Generate alerts for critical issues."""
        alerts = []
        
        # System alerts
        for metric_name, metric in system_metrics.items():
            if metric.status == 'critical':
                alerts.append(f"CRITICAL: {metric.name} is at {metric.value:.1f}{metric.unit} (threshold: {metric.threshold}{metric.unit})")
        
        # Service alerts
        for service_name, health in service_health.items():
            if health.status == 'critical':
                alerts.append(f"CRITICAL: Service {service_name} is down")
            elif health.status == 'warning':
                alerts.append(f"WARNING: Service {service_name} has issues")
        
        return alerts
    
    def calculate_performance_score(self, system_metrics: Dict[str, HealthMetric], 
                                   service_health: Dict[str, ServiceHealth]) -> float:
        """Calculate overall performance score (0-100)."""
        score = 100.0
        
        # Deduct points for system metrics
        for metric_name, metric in system_metrics.items():
            if metric.status == 'critical':
                score -= 30
            elif metric.status == 'warning':
                score -= 15
        
        # Deduct points for service health
        for service_name, health in service_health.items():
            if health.status == 'critical':
                score -= 20
            elif health.status == 'warning':
                score -= 10
            elif health.status == 'unknown':
                score -= 5
        
        # Deduct points for slow response times
        for service_name, health in service_health.items():
            if health.response_time > self.thresholds['response_time']['critical']:
                score -= 10
            elif health.response_time > self.thresholds['response_time']['warning']:
                score -= 5
        
        return max(0.0, score)
    
    async def generate_health_report(self) -> HealthReport:
        """Generate comprehensive health report."""
        self.logger.info("Generating health report...")
        
        # Collect metrics
        system_metrics = await self.collect_system_metrics()
        service_health = await self.check_service_health()
        
        # Generate recommendations and alerts
        recommendations = self.generate_recommendations(system_metrics, service_health)
        alerts = self.generate_alerts(system_metrics, service_health)
        
        # Calculate performance score
        performance_score = self.calculate_performance_score(system_metrics, service_health)
        
        # Determine overall status
        critical_issues = len([m for m in system_metrics.values() if m.status == 'critical']) + \
                         len([h for h in service_health.values() if h.status == 'critical'])
        
        if critical_issues > 0:
            overall_status = 'critical'
        elif len(alerts) > 0:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        # Create health report
        report = HealthReport(
            timestamp=datetime.now(),
            overall_status=overall_status,
            system_metrics=system_metrics,
            service_health=service_health,
            recommendations=recommendations,
            alerts=alerts,
            performance_score=performance_score
        )
        
        # Store report
        self.health_reports.append(report)
        
        return report
    
    async def monitoring_loop(self):
        """Main monitoring loop."""
        self.logger.info("Starting advanced health monitoring...")
        
        while self.monitoring_active:
            try:
                # Generate health report
                report = await self.generate_health_report()
                
                # Log critical issues
                if report.overall_status == 'critical':
                    self.logger.critical(f"CRITICAL: {len(report.alerts)} critical issues detected")
                    for alert in report.alerts:
                        self.logger.critical(alert)
                
                # Log warnings
                elif report.overall_status == 'warning':
                    self.logger.warning(f"WARNING: {len(report.alerts)} issues detected")
                    for alert in report.alerts:
                        self.logger.warning(alert)
                
                # Log performance score
                self.logger.info(f"Performance score: {report.performance_score:.1f}/100")
                
                # Save report to file
                report_file = self.log_dir / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, 'w') as f:
                    json.dump(asdict(report), f, indent=2, default=str)
                
                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            self.logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self.monitoring_loop())
        self.logger.info("Advanced health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        self.logger.info("Advanced health monitoring stopped")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary."""
        if not self.health_reports:
            return {'status': 'no_data'}
        
        latest_report = self.health_reports[-1]
        
        return {
            'timestamp': latest_report.timestamp.isoformat(),
            'overall_status': latest_report.overall_status,
            'performance_score': latest_report.performance_score,
            'critical_issues': len([a for a in latest_report.alerts if 'CRITICAL' in a]),
            'warnings': len([a for a in latest_report.alerts if 'WARNING' in a]),
            'recommendations_count': len(latest_report.recommendations),
            'system_metrics': {
                name: {
                    'value': metric.value,
                    'unit': metric.unit,
                    'status': metric.status,
                    'trend': metric.trend
                }
                for name, metric in latest_report.system_metrics.items()
            },
            'service_health': {
                name: {
                    'status': health.status,
                    'response_time': health.response_time,
                    'error_rate': health.error_rate,
                    'uptime': health.uptime
                }
                for name, health in latest_report.service_health.items()
            }
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Advanced DSPy Agent Health Monitor")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace directory")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--report", action="store_true", help="Generate health report")
    parser.add_argument("--summary", action="store_true", help="Show health summary")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create health monitor
    monitor = AdvancedHealthMonitor(args.workspace, args.interval)
    
    if args.report:
        # Generate single health report
        report = await monitor.generate_health_report()
        print(json.dumps(asdict(report), indent=2, default=str))
        return
    
    if args.summary:
        # Show health summary
        summary = monitor.get_health_summary()
        print(json.dumps(summary, indent=2))
        return
    
    if args.daemon:
        # Run as daemon
        monitor.start_monitoring()
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
    else:
        # Run single check
        report = await monitor.generate_health_report()
        print(f"Health Status: {report.overall_status}")
        print(f"Performance Score: {report.performance_score:.1f}/100")
        print(f"Critical Issues: {len([a for a in report.alerts if 'CRITICAL' in a])}")
        print(f"Warnings: {len([a for a in report.alerts if 'WARNING' in a])}")
        print(f"Recommendations: {len(report.recommendations)}")


if __name__ == "__main__":
    asyncio.run(main())
