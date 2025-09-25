"""
Monitoring and auto-scaling components for DSPy Agent.

Avoid importing optional submodules at package import time so tests can run in
minimal environments.
"""

try:
    from .auto_scaler import AutoScaler  # type: ignore
except Exception:  # pragma: no cover - optional
    AutoScaler = None  # type: ignore

try:
    from .performance_monitor import PerformanceMonitor  # type: ignore
except Exception:  # pragma: no cover - optional
    PerformanceMonitor = None  # type: ignore

__all__ = ['AutoScaler', 'PerformanceMonitor']
