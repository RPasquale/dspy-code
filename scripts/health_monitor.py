#!/usr/bin/env python3
"""
Comprehensive Health Monitor for DSPy Agent Stack
This script monitors all components and provides detailed health reports.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import requests
import psutil
import docker
from dataclasses import dataclass, asdict
import argparse


@dataclass
class HealthStatus:
    """Health status for a component."""
    name: str
    status: str  # healthy, warning, critical, unknown
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class HealthMonitor:
    """Comprehensive health monitor for DSPy agent stack."""
    
    def __init__(self, workspace: Path, log_level: str = "INFO"):
        self.workspace = Path(workspace)
        self.log_dir = self.workspace / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "health_monitor.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.warning(f"Could not initialize Docker client: {e}")
            self.docker_client = None
        
        # Health check results
        self.health_results: List[HealthStatus] = []
        
        # Service endpoints
        self.endpoints = {
            'agent': 'http://localhost:8765',
            'dashboard': 'http://localhost:18081',
            'ollama': 'http://localhost:11435',
            'kafka': 'localhost:9092',
            'infermesh': 'http://localhost:9000',
            'embed_worker': 'http://localhost:9101',
            'spark': 'http://localhost:4041'
        }
    
    async def check_system_resources(self) -> HealthStatus:
        """Check system resource usage with enhanced performance metrics."""
        try:
            # CPU usage with detailed metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory usage with detailed breakdown
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk usage with I/O metrics
            disk = psutil.disk_usage(str(self.workspace))
            disk_percent = disk.percent
            disk_free_gb = disk.free / (1024**3)
            disk_used_gb = disk.used / (1024**3)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            net_bytes_sent = net_io.bytes_sent
            net_bytes_recv = net_io.bytes_recv
            
            # Process-specific metrics
            current_process = psutil.Process()
            process_memory = current_process.memory_info()
            process_cpu = current_process.cpu_percent()
            
            # Enhanced metrics collection
            metrics = {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'cpu_freq_current': cpu_freq.current if cpu_freq else None,
                'memory_percent': memory_percent,
                'memory_available_gb': memory_available_gb,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'disk_percent': disk_percent,
                'disk_free_gb': disk_free_gb,
                'disk_used_gb': disk_used_gb,
                'net_bytes_sent': net_bytes_sent,
                'net_bytes_recv': net_bytes_recv,
                'process_memory_mb': process_memory.rss / (1024**2),
                'process_cpu_percent': process_cpu
            }
            
            # Enhanced status determination with performance recommendations
            status = "healthy"
            message = "System resources normal"
            recommendations = []
            
            if cpu_percent > 90:
                status = "critical"
                message = f"High CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > 80:
                status = "warning"
                message = f"Elevated CPU usage: {cpu_percent:.1f}%"
            
            if memory_percent > 95:
                status = "critical"
                message = f"Critical memory usage: {memory_percent:.1f}%"
            elif memory_percent > 85:
                status = "warning"
                message = f"High memory usage: {memory_percent:.1f}%"
            
            if disk_percent > 95:
                status = "critical"
                message = f"Critical disk usage: {disk_percent:.1f}%"
            elif disk_percent > 85:
                status = "warning"
                message = f"High disk usage: {disk_percent:.1f}%"
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_gb': memory_available_gb,
                'disk_percent': disk_percent,
                'disk_free_gb': disk_free_gb
            }
            
            return HealthStatus(
                name="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics
            )
            
        except Exception as e:
            return HealthStatus(
                name="system_resources",
                status="unknown",
                message=f"Failed to check system resources: {e}",
                timestamp=datetime.now()
            )
    
    async def check_docker_services(self) -> List[HealthStatus]:
        """Check Docker services health."""
        results = []
        
        if not self.docker_client:
            results.append(HealthStatus(
                name="docker_services",
                status="unknown",
                message="Docker client not available",
                timestamp=datetime.now()
            ))
            return results
        
        try:
            # Get all containers
            containers = self.docker_client.containers.list(all=True)
            
            # Check each service
            service_containers = {
                'dspy-agent': [c for c in containers if 'dspy-agent' in c.name],
                'ollama': [c for c in containers if 'ollama' in c.name],
                'kafka': [c for c in containers if 'kafka' in c.name],
                'zookeeper': [c for c in containers if 'zookeeper' in c.name],
                'infermesh': [c for c in containers if 'infermesh' in c.name],
                'embed-worker': [c for c in containers if 'embed-worker' in c.name],
                'spark': [c for c in containers if 'spark' in c.name]
            }
            
            for service_name, service_containers in service_containers.items():
                if not service_containers:
                    results.append(HealthStatus(
                        name=f"docker_{service_name}",
                        status="critical",
                        message=f"No {service_name} containers found",
                        timestamp=datetime.now()
                    ))
                    continue
                
                for container in service_containers:
                    container_name = container.name
                    status = container.status
                    
                    if status == 'running':
                        container_status = "healthy"
                        message = f"Container {container_name} is running"
                    elif status == 'exited':
                        container_status = "critical"
                        message = f"Container {container_name} has exited"
                    else:
                        container_status = "warning"
                        message = f"Container {container_name} status: {status}"
                    
                    # Get container stats
                    try:
                        stats = container.stats(stream=False)
                        metrics = {
                            'cpu_usage': stats.get('cpu_stats', {}).get('cpu_usage', {}).get('total_usage', 0),
                            'memory_usage': stats.get('memory_stats', {}).get('usage', 0),
                            'memory_limit': stats.get('memory_stats', {}).get('limit', 0)
                        }
                    except Exception:
                        metrics = {}
                    
                    results.append(HealthStatus(
                        name=f"docker_{container_name}",
                        status=container_status,
                        message=message,
                        timestamp=datetime.now(),
                        metrics=metrics
                    ))
            
        except Exception as e:
            results.append(HealthStatus(
                name="docker_services",
                status="unknown",
                message=f"Failed to check Docker services: {e}",
                timestamp=datetime.now()
            ))
        
        return results
    
    async def check_http_endpoints(self) -> List[HealthStatus]:
        """Check HTTP endpoint health."""
        results = []
        
        for service_name, endpoint in self.endpoints.items():
            try:
                if endpoint.startswith('http'):
                    response = requests.get(f"{endpoint}/health", timeout=5)
                    if response.status_code == 200:
                        status = "healthy"
                        message = f"{service_name} endpoint responding"
                    else:
                        status = "warning"
                        message = f"{service_name} endpoint returned {response.status_code}"
                else:
                    # For non-HTTP endpoints, just check if port is open
                    import socket
                    host, port = endpoint.split(':')
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, int(port)))
                    sock.close()
                    
                    if result == 0:
                        status = "healthy"
                        message = f"{service_name} port {port} is open"
                    else:
                        status = "critical"
                        message = f"{service_name} port {port} is not accessible"
                
                results.append(HealthStatus(
                    name=f"endpoint_{service_name}",
                    status=status,
                    message=message,
                    timestamp=datetime.now()
                ))
                
            except requests.exceptions.RequestException as e:
                results.append(HealthStatus(
                    name=f"endpoint_{service_name}",
                    status="critical",
                    message=f"{service_name} endpoint error: {e}",
                    timestamp=datetime.now()
                ))
            except Exception as e:
                results.append(HealthStatus(
                    name=f"endpoint_{service_name}",
                    status="unknown",
                    message=f"{service_name} endpoint check failed: {e}",
                    timestamp=datetime.now()
                ))
        
        return results
    
    async def check_agent_logs(self) -> HealthStatus:
        """Check agent logs for errors."""
        try:
            log_files = [
                self.log_dir / "agent_actions.jsonl",
                self.log_dir / "agent_learning.jsonl",
                self.log_dir / "agent_thoughts.jsonl"
            ]
            
            error_count = 0
            warning_count = 0
            total_entries = 0
            
            for log_file in log_files:
                if not log_file.exists():
                    continue
                
                # Check recent entries (last hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            total_entries += 1
                            
                            # Check timestamp
                            entry_time = datetime.fromisoformat(entry.get('timestamp', ''))
                            if entry_time < cutoff_time:
                                continue
                            
                            # Check for errors
                            if 'error' in entry.get('level', '').lower():
                                error_count += 1
                            elif 'warning' in entry.get('level', '').lower():
                                warning_count += 1
                                
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            
            # Determine status
            if error_count > 10:
                status = "critical"
                message = f"High error rate: {error_count} errors in last hour"
            elif error_count > 5:
                status = "warning"
                message = f"Elevated error rate: {error_count} errors in last hour"
            elif warning_count > 20:
                status = "warning"
                message = f"High warning rate: {warning_count} warnings in last hour"
            else:
                status = "healthy"
                message = f"Logs normal: {error_count} errors, {warning_count} warnings"
            
            metrics = {
                'error_count': error_count,
                'warning_count': warning_count,
                'total_entries': total_entries
            }
            
            return HealthStatus(
                name="agent_logs",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics
            )
            
        except Exception as e:
            return HealthStatus(
                name="agent_logs",
                status="unknown",
                message=f"Failed to check agent logs: {e}",
                timestamp=datetime.now()
            )
    
    async def check_workspace_health(self) -> HealthStatus:
        """Check workspace health and permissions."""
        try:
            issues = []
            
            # Check workspace permissions
            if not os.access(self.workspace, os.R_OK | os.W_OK):
                issues.append("Workspace not readable/writable")
            
            # Check log directory
            if not os.access(self.log_dir, os.R_OK | os.W_OK):
                issues.append("Log directory not accessible")
            
            # Check cache directory
            cache_dir = self.workspace / ".dspy_cache"
            if cache_dir.exists() and not os.access(cache_dir, os.R_OK | os.W_OK):
                issues.append("Cache directory not accessible")
            
            # Check disk space
            disk_usage = psutil.disk_usage(str(self.workspace))
            if disk_usage.percent > 90:
                issues.append(f"Low disk space: {disk_usage.percent:.1f}% used")
            
            if issues:
                status = "warning"
                message = "; ".join(issues)
            else:
                status = "healthy"
                message = "Workspace health normal"
            
            metrics = {
                'disk_usage_percent': disk_usage.percent,
                'disk_free_gb': disk_usage.free / (1024**3),
                'issues': issues
            }
            
            return HealthStatus(
                name="workspace_health",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics
            )
            
        except Exception as e:
            return HealthStatus(
                name="workspace_health",
                status="unknown",
                message=f"Failed to check workspace health: {e}",
                timestamp=datetime.now()
            )
    
    async def run_health_checks(self) -> List[HealthStatus]:
        """Run all health checks."""
        self.logger.info("Starting comprehensive health checks...")
        
        all_results = []
        
        # System resources
        self.logger.info("Checking system resources...")
        system_health = await self.check_system_resources()
        all_results.append(system_health)
        
        # Docker services
        self.logger.info("Checking Docker services...")
        docker_health = await self.check_docker_services()
        all_results.extend(docker_health)
        
        # HTTP endpoints
        self.logger.info("Checking HTTP endpoints...")
        endpoint_health = await self.check_http_endpoints()
        all_results.extend(endpoint_health)
        
        # Agent logs
        self.logger.info("Checking agent logs...")
        log_health = await self.check_agent_logs()
        all_results.append(log_health)
        
        # Workspace health
        self.logger.info("Checking workspace health...")
        workspace_health = await self.check_workspace_health()
        all_results.append(workspace_health)
        
        self.health_results = all_results
        return all_results
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        if not self.health_results:
            return {"error": "No health check results available"}
        
        # Categorize results
        healthy_count = sum(1 for r in self.health_results if r.status == "healthy")
        warning_count = sum(1 for r in self.health_results if r.status == "warning")
        critical_count = sum(1 for r in self.health_results if r.status == "critical")
        unknown_count = sum(1 for r in self.health_results if r.status == "unknown")
        
        # Overall status
        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"
        elif unknown_count > 0:
            overall_status = "unknown"
        else:
            overall_status = "healthy"
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_checks": len(self.health_results),
                "healthy": healthy_count,
                "warning": warning_count,
                "critical": critical_count,
                "unknown": unknown_count
            },
            "components": [result.to_dict() for result in self.health_results],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on health check results."""
        recommendations = []
        
        for result in self.health_results:
            if result.status == "critical":
                if "docker" in result.name:
                    recommendations.append(f"Restart {result.name}: {result.message}")
                elif "endpoint" in result.name:
                    recommendations.append(f"Check {result.name} service: {result.message}")
                elif "system_resources" in result.name:
                    recommendations.append(f"Monitor system resources: {result.message}")
            
            elif result.status == "warning":
                if "memory" in result.message.lower():
                    recommendations.append("Consider increasing available memory")
                elif "disk" in result.message.lower():
                    recommendations.append("Consider cleaning up disk space")
                elif "cpu" in result.message.lower():
                    recommendations.append("Monitor CPU usage and consider scaling")
        
        return recommendations
    
    def save_health_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """Save health report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_report_{timestamp}.json"
        
        report_path = self.log_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Health report saved to {report_path}")
        return report_path


async def main():
    """Main function for health monitoring."""
    parser = argparse.ArgumentParser(description="DSPy Agent Health Monitor")
    parser.add_argument("--workspace", type=str, default=".", help="Workspace directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument("--output", type=str, help="Output file for health report")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    
    args = parser.parse_args()
    
    workspace = Path(args.workspace).resolve()
    
    monitor = HealthMonitor(workspace, args.log_level)
    
    if args.continuous:
        monitor.logger.info(f"Starting continuous health monitoring (interval: {args.interval}s)")
        
        while True:
            try:
                # Run health checks
                await monitor.run_health_checks()
                
                # Generate and save report
                report = monitor.generate_health_report()
                report_path = monitor.save_health_report(report)
                
                # Print summary
                print(f"\n=== Health Check Summary ===")
                print(f"Overall Status: {report['overall_status'].upper()}")
                print(f"Healthy: {report['summary']['healthy']}")
                print(f"Warning: {report['summary']['warning']}")
                print(f"Critical: {report['summary']['critical']}")
                print(f"Unknown: {report['summary']['unknown']}")
                
                if report['recommendations']:
                    print(f"\nRecommendations:")
                    for rec in report['recommendations']:
                        print(f"  - {rec}")
                
                print(f"\nReport saved to: {report_path}")
                print(f"Next check in {args.interval} seconds...\n")
                
                # Wait for next check
                await asyncio.sleep(args.interval)
                
            except KeyboardInterrupt:
                monitor.logger.info("Health monitoring stopped by user")
                break
            except Exception as e:
                monitor.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    else:
        # Single health check
        await monitor.run_health_checks()
        report = monitor.generate_health_report()
        
        if args.output:
            report_path = monitor.save_health_report(report, args.output)
        else:
            report_path = monitor.save_health_report(report)
        
        # Print results
        print(json.dumps(report, indent=2))
        
        # Exit with appropriate code
        if report['overall_status'] == 'critical':
            sys.exit(2)
        elif report['overall_status'] == 'warning':
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
