#!/usr/bin/env python3
"""
Intelligent optimization analysis script for DSPy Agent system.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dspy_agent.monitor.performance_monitor import PerformanceMonitor
    from dspy_agent.monitor.auto_scaler import AutoScaler
except ImportError as e:
    print(f"Warning: Could not import monitoring modules: {e}")
    print("Running basic system analysis instead...")
    
    # Fallback basic analysis
    import psutil
    import time
    
    async def basic_analysis():
        print('=== Basic System Analysis ===')
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f'CPU Usage: {cpu_percent:.1f}%')
        
        # Memory usage
        memory = psutil.virtual_memory()
        print(f'Memory Usage: {memory.percent:.1f}%')
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            print(f'Disk Read: {disk_io.read_bytes / 1024 / 1024:.1f} MB')
            print(f'Disk Write: {disk_io.write_bytes / 1024 / 1024:.1f} MB')
        
        # Network I/O
        net_io = psutil.net_io_counters()
        if net_io:
            print(f'Network Sent: {net_io.bytes_sent / 1024 / 1024:.1f} MB')
            print(f'Network Received: {net_io.bytes_recv / 1024 / 1024:.1f} MB')
        
        print('\n=== Basic Analysis Complete ===')
        return True
    
    asyncio.run(basic_analysis())
    sys.exit(0)

async def run_optimization():
    """Run comprehensive optimization analysis."""
    try:
        # Performance analysis
        perf_monitor = PerformanceMonitor(str(project_root))
        snapshot = await perf_monitor.collect_performance_snapshot()
        anomalies = perf_monitor.detect_anomalies(snapshot)
        recommendations = perf_monitor.generate_optimization_recommendations(snapshot)
        
        print('=== Intelligent Optimization Report ===')
        print(f'CPU Usage: {snapshot.cpu_usage:.1f}%')
        print(f'Memory Usage: {snapshot.memory_usage:.1f}%')
        print(f'Disk I/O: {snapshot.disk_io:.1f} MB/s')
        print(f'Network I/O: {snapshot.network_io:.1f} MB/s')
        print(f'Active Connections: {snapshot.active_connections}')
        print(f'Queue Depth: {snapshot.queue_depth}')
        print(f'Response Time: {snapshot.avg_response_time:.2f}s')
        
        if anomalies:
            print('\n=== Anomalies Detected ===')
            for anomaly in anomalies:
                print(f'- {anomaly.type}: {anomaly.description}')
        else:
            print('\n=== No Anomalies Detected ===')
        
        if recommendations:
            print('\n=== Optimization Recommendations ===')
            for rec in recommendations:
                print(f'- {rec.priority}: {rec.description}')
        else:
            print('\n=== No Optimization Recommendations ===')
        
        # Auto-scaling analysis
        auto_scaler = AutoScaler(str(project_root))
        scaling_plan = auto_scaler.analyze_scaling_needs(snapshot)
        
        print('\n=== Auto-Scaling Analysis ===')
        print(f'Recommended CPU Limit: {scaling_plan.cpu_limit}')
        print(f'Recommended Memory Limit: {scaling_plan.memory_limit}')
        print(f'Recommended Scale Factor: {scaling_plan.scale_factor}')
        
        if scaling_plan.should_scale:
            print('\n=== Scaling Recommendations ===')
            for action in scaling_plan.actions:
                print(f'- {action.type}: {action.description}')
        else:
            print('\n=== No Scaling Required ===')
        
        # Save report
        report = {
            'timestamp': snapshot.timestamp.isoformat(),
            'performance': {
                'cpu_usage': snapshot.cpu_usage,
                'memory_usage': snapshot.memory_usage,
                'disk_io': snapshot.disk_io,
                'network_io': snapshot.network_io,
                'active_connections': snapshot.active_connections,
                'queue_depth': snapshot.queue_depth,
                'avg_response_time': snapshot.avg_response_time
            },
            'anomalies': [{'type': a.type, 'description': a.description} for a in anomalies],
            'recommendations': [{'priority': r.priority, 'description': r.description} for r in recommendations],
            'scaling_plan': {
                'cpu_limit': scaling_plan.cpu_limit,
                'memory_limit': scaling_plan.memory_limit,
                'scale_factor': scaling_plan.scale_factor,
                'should_scale': scaling_plan.should_scale,
                'actions': [{'type': a.type, 'description': a.description} for a in scaling_plan.actions]
            }
        }
        
        with open('intelligent_optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print('\n=== Report Saved ===')
        print('Report saved to: intelligent_optimization_report.json')
        
    except Exception as e:
        print(f"Error during optimization analysis: {e}")
        print("Falling back to basic system analysis...")
        await basic_analysis()

if __name__ == "__main__":
    asyncio.run(run_optimization())
