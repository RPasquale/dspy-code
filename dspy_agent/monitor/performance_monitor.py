"""
Advanced Performance Monitoring System for DSPy Agent
Provides real-time performance analysis, anomaly detection, and optimization recommendations.
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import psutil
import numpy as np
from scipy import stats


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    process_metrics: Dict[str, Any]
    service_metrics: Dict[str, Any]


@dataclass
class AnomalyAlert:
    """Anomaly detection alert"""
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    severity: str  # low, medium, high, critical
    description: str
    timestamp: datetime
    confidence: float


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    category: str  # memory, cpu, disk, network, cache
    priority: str  # low, medium, high, critical
    title: str
    description: str
    impact: str
    effort: str
    confidence: float
    metrics: Dict[str, Any]


class PerformanceMonitor:
    """Advanced performance monitoring with anomaly detection and optimization recommendations."""
    
    def __init__(self, workspace: Path, monitoring_interval: int = 30):
        self.workspace = Path(workspace)
        self.monitoring_interval = monitoring_interval
        
        # Setup logging
        self.logger = logging.getLogger('performance_monitor')
        self.logger.setLevel(logging.INFO)
        
        # Performance history
        self.performance_history = deque(maxlen=1000)
        self.anomaly_history = deque(maxlen=100)
        self.recommendations_history = deque(maxlen=50)
        
        # Anomaly detection
        self.baseline_metrics = {}
        self.anomaly_thresholds = {
            'cpu_usage': {'z_score': 2.5, 'percentile': 95},
            'memory_usage': {'z_score': 2.0, 'percentile': 90},
            'response_time': {'z_score': 3.0, 'percentile': 99},
            'error_rate': {'z_score': 2.0, 'percentile': 95}
        }
        
        # Optimization patterns
        self.optimization_patterns = {
            'memory_leak': {
                'indicators': ['memory_usage_trend', 'gc_frequency'],
                'threshold': 0.8,
                'recommendation': 'Check for memory leaks in application code'
            },
            'cpu_spike': {
                'indicators': ['cpu_usage_spike', 'process_count'],
                'threshold': 0.9,
                'recommendation': 'Optimize CPU-intensive operations'
            },
            'disk_contention': {
                'indicators': ['disk_io_wait', 'disk_usage'],
                'threshold': 0.7,
                'recommendation': 'Optimize disk I/O operations'
            }
        }
        
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
    
    async def collect_performance_snapshot(self) -> PerformanceSnapshot:
        """Collect comprehensive performance snapshot."""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_metrics = {
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0,
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0
            }
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_metrics = {
                'bytes_sent': net_io.bytes_sent if net_io else 0,
                'bytes_recv': net_io.bytes_recv if net_io else 0,
                'packets_sent': net_io.packets_sent if net_io else 0,
                'packets_recv': net_io.packets_recv if net_io else 0
            }
            
            # Process metrics
            current_process = psutil.Process()
            process_metrics = {
                'cpu_percent': current_process.cpu_percent(),
                'memory_percent': current_process.memory_percent(),
                'memory_info': current_process.memory_info()._asdict(),
                'num_threads': current_process.num_threads(),
                'num_fds': current_process.num_fds() if hasattr(current_process, 'num_fds') else 0
            }
            
            # Service metrics (simplified)
            service_metrics = await self._collect_service_metrics()
            
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_io=disk_metrics,
                network_io=network_metrics,
                process_metrics=process_metrics,
                service_metrics=service_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting performance snapshot: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_io={},
                network_io={},
                process_metrics={},
                service_metrics={}
            )
    
    async def _collect_service_metrics(self) -> Dict[str, Any]:
        """Collect service-specific metrics."""
        try:
            # This would collect metrics from various services
            # For now, return simplified metrics
            return {
                'agent_response_time': 1.0,
                'ollama_response_time': 2.0,
                'kafka_lag': 0,
                'reddb_connections': 5
            }
        except Exception as e:
            self.logger.warning(f"Error collecting service metrics: {e}")
            return {}
    
    def detect_anomalies(self, snapshot: PerformanceSnapshot) -> List[AnomalyAlert]:
        """Detect performance anomalies."""
        alerts = []
        
        # Update baseline metrics
        self._update_baseline_metrics(snapshot)
        
        # Check CPU anomalies
        if len(self.performance_history) > 10:
            cpu_values = [s.cpu_usage for s in list(self.performance_history)[-20:]]
            if snapshot.cpu_usage > self._calculate_threshold(cpu_values, 'cpu_usage'):
                alerts.append(AnomalyAlert(
                    metric_name='cpu_usage',
                    current_value=snapshot.cpu_usage,
                    expected_range=self._get_expected_range(cpu_values),
                    severity='high' if snapshot.cpu_usage > 90 else 'medium',
                    description=f"High CPU usage detected: {snapshot.cpu_usage:.1f}%",
                    timestamp=snapshot.timestamp,
                    confidence=0.8
                ))
        
        # Check memory anomalies
        if len(self.performance_history) > 10:
            memory_values = [s.memory_usage for s in list(self.performance_history)[-20:]]
            if snapshot.memory_usage > self._calculate_threshold(memory_values, 'memory_usage'):
                alerts.append(AnomalyAlert(
                    metric_name='memory_usage',
                    current_value=snapshot.memory_usage,
                    expected_range=self._get_expected_range(memory_values),
                    severity='critical' if snapshot.memory_usage > 95 else 'high',
                    description=f"High memory usage detected: {snapshot.memory_usage:.1f}%",
                    timestamp=snapshot.timestamp,
                    confidence=0.9
                ))
        
        # Check for memory leak patterns
        if self._detect_memory_leak():
            alerts.append(AnomalyAlert(
                metric_name='memory_leak',
                current_value=snapshot.memory_usage,
                expected_range=(0, 80),
                severity='high',
                description="Potential memory leak detected based on trend analysis",
                timestamp=snapshot.timestamp,
                confidence=0.7
            ))
        
        return alerts
    
    def _update_baseline_metrics(self, snapshot: PerformanceSnapshot):
        """Update baseline metrics for anomaly detection."""
        if 'cpu_usage' not in self.baseline_metrics:
            self.baseline_metrics['cpu_usage'] = []
        if 'memory_usage' not in self.baseline_metrics:
            self.baseline_metrics['memory_usage'] = []
        
        self.baseline_metrics['cpu_usage'].append(snapshot.cpu_usage)
        self.baseline_metrics['memory_usage'].append(snapshot.memory_usage)
        
        # Keep only recent values
        for metric in self.baseline_metrics:
            if len(self.baseline_metrics[metric]) > 100:
                self.baseline_metrics[metric] = self.baseline_metrics[metric][-100:]
    
    def _calculate_threshold(self, values: List[float], metric_name: str) -> float:
        """Calculate anomaly threshold for a metric."""
        if len(values) < 5:
            return 100.0  # Default high threshold
        
        try:
            mean = np.mean(values)
            std = np.std(values)
            z_score = self.anomaly_thresholds.get(metric_name, {}).get('z_score', 2.0)
            return mean + (z_score * std)
        except Exception:
            return 100.0
    
    def _get_expected_range(self, values: List[float]) -> Tuple[float, float]:
        """Get expected range for a metric."""
        if len(values) < 5:
            return (0.0, 100.0)
        
        try:
            mean = np.mean(values)
            std = np.std(values)
            return (max(0, mean - 2 * std), min(100, mean + 2 * std))
        except Exception:
            return (0.0, 100.0)
    
    def _detect_memory_leak(self) -> bool:
        """Detect potential memory leak based on trend analysis."""
        if len(self.performance_history) < 20:
            return False
        
        try:
            memory_values = [s.memory_usage for s in list(self.performance_history)[-20:]]
            
            # Check for upward trend
            x = np.arange(len(memory_values))
            slope, _, r_value, _, _ = stats.linregress(x, memory_values)
            
            # Memory leak if positive slope with good correlation
            return slope > 0.5 and r_value > 0.7
        except Exception:
            return False
    
    def generate_optimization_recommendations(self, snapshot: PerformanceSnapshot) -> List[OptimizationRecommendation]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # CPU optimization
        if snapshot.cpu_usage > 80:
            recommendations.append(OptimizationRecommendation(
                category='cpu',
                priority='high',
                title='Optimize CPU Usage',
                description=f'CPU usage is {snapshot.cpu_usage:.1f}%. Consider optimizing CPU-intensive operations.',
                impact='High - Reduces system load and improves responsiveness',
                effort='Medium - Requires code analysis and optimization',
                confidence=0.8,
                metrics={'cpu_usage': snapshot.cpu_usage}
            ))
        
        # Memory optimization
        if snapshot.memory_usage > 85:
            recommendations.append(OptimizationRecommendation(
                category='memory',
                priority='critical',
                title='Optimize Memory Usage',
                description=f'Memory usage is {snapshot.memory_usage:.1f}%. Check for memory leaks and optimize memory allocation.',
                impact='Critical - Prevents out-of-memory errors',
                effort='High - Requires memory profiling and optimization',
                confidence=0.9,
                metrics={'memory_usage': snapshot.memory_usage}
            ))
        
        # Cache optimization
        if snapshot.process_metrics.get('memory_percent', 0) > 70:
            recommendations.append(OptimizationRecommendation(
                category='cache',
                priority='medium',
                title='Optimize Cache Usage',
                description='High process memory usage suggests cache optimization opportunities.',
                impact='Medium - Improves memory efficiency',
                effort='Low - Adjust cache size and TTL settings',
                confidence=0.6,
                metrics={'process_memory': snapshot.process_metrics.get('memory_percent', 0)}
            ))
        
        # Disk I/O optimization
        disk_usage = snapshot.disk_io.get('read_bytes', 0) + snapshot.disk_io.get('write_bytes', 0)
        if disk_usage > 1e9:  # 1GB
            recommendations.append(OptimizationRecommendation(
                category='disk',
                priority='medium',
                title='Optimize Disk I/O',
                description='High disk I/O activity detected. Consider optimizing file operations.',
                impact='Medium - Reduces disk contention',
                effort='Medium - Optimize file I/O patterns',
                confidence=0.7,
                metrics={'disk_io': disk_usage}
            ))
        
        return recommendations
    
    async def monitoring_loop(self):
        """Main performance monitoring loop."""
        self.logger.info("Starting performance monitoring loop...")
        
        while self.monitoring_active:
            try:
                # Collect performance snapshot
                snapshot = await self.collect_performance_snapshot()
                self.performance_history.append(snapshot)
                
                # Detect anomalies
                anomalies = self.detect_anomalies(snapshot)
                for anomaly in anomalies:
                    self.anomaly_history.append(anomaly)
                    self.logger.warning(f"ANOMALY: {anomaly.description} (severity: {anomaly.severity})")
                
                # Generate recommendations
                recommendations = self.generate_optimization_recommendations(snapshot)
                for rec in recommendations:
                    if rec.priority in ['high', 'critical']:
                        self.recommendations_history.append(rec)
                        self.logger.info(f"RECOMMENDATION: {rec.title} (priority: {rec.priority})")
                
                # Log performance data
                if self.data_manager:
                    try:
                        from ..db import create_log_entry, Environment
                        perf_log = create_log_entry(
                            level="INFO",
                            source="monitor.performance",
                            message=f"Performance snapshot: CPU={snapshot.cpu_usage:.1f}%, Memory={snapshot.memory_usage:.1f}%",
                            context={
                                'cpu_usage': snapshot.cpu_usage,
                                'memory_usage': snapshot.memory_usage,
                                'anomalies_count': len(anomalies),
                                'recommendations_count': len(recommendations)
                            },
                            environment=Environment.DEVELOPMENT
                        )
                        self.data_manager.log(perf_log)
                    except Exception as e:
                        self.logger.warning(f"Could not log performance data: {e}")
                
                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Performance monitoring already running")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._run_monitoring_async)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        self.logger.info("Performance monitoring started")
    
    def _run_monitoring_async(self):
        """Run monitoring in async context."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.monitoring_loop())
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.performance_history:
            return {'error': 'No performance data available'}
        
        recent_snapshots = list(self.performance_history)[-10:]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_snapshots': len(self.performance_history),
                'recent_anomalies': len([a for a in self.anomaly_history if (datetime.now() - a.timestamp).seconds < 3600]),
                'active_recommendations': len([r for r in self.recommendations_history if r.priority in ['high', 'critical']])
            },
            'current_metrics': {
                'cpu_usage': recent_snapshots[-1].cpu_usage if recent_snapshots else 0,
                'memory_usage': recent_snapshots[-1].memory_usage if recent_snapshots else 0,
                'avg_cpu': np.mean([s.cpu_usage for s in recent_snapshots]) if recent_snapshots else 0,
                'avg_memory': np.mean([s.memory_usage for s in recent_snapshots]) if recent_snapshots else 0
            },
            'recent_anomalies': [
                {
                    'metric': a.metric_name,
                    'value': a.current_value,
                    'severity': a.severity,
                    'description': a.description,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in list(self.anomaly_history)[-5:]
            ],
            'recommendations': [
                {
                    'category': r.category,
                    'priority': r.priority,
                    'title': r.title,
                    'description': r.description,
                    'impact': r.impact,
                    'effort': r.effort
                }
                for r in list(self.recommendations_history)[-5:]
            ]
        }


def main():
    """Main entry point for performance monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DSPy Agent Performance Monitor")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace directory")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create performance monitor
    monitor = PerformanceMonitor(args.workspace, args.interval)
    
    if args.report:
        # Generate performance report
        report = monitor.get_performance_report()
        print(json.dumps(report, indent=2))
        return
    
    if args.daemon:
        monitor.start_monitoring()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
    else:
        # Run single check
        snapshot = asyncio.run(monitor.collect_performance_snapshot())
        anomalies = monitor.detect_anomalies(snapshot)
        recommendations = monitor.generate_optimization_recommendations(snapshot)
        
        print("Performance Monitor Report:")
        print(f"  CPU Usage: {snapshot.cpu_usage:.1f}%")
        print(f"  Memory Usage: {snapshot.memory_usage:.1f}%")
        print(f"  Anomalies Detected: {len(anomalies)}")
        print(f"  Recommendations: {len(recommendations)}")


if __name__ == "__main__":
    main()
