"""
Monitoring and auto-scaling components for DSPy Agent.

Avoid importing optional submodules at package import time so tests can run in
minimal environments.
"""

try:
    from .auto_scaler import AutoScaler  # type: ignore
except Exception:  # pragma: no cover - optional
    AutoScaler = None  # type: ignore

# Avoid importing the numpy/scipy-heavy performance monitor at module import
# time. On sandboxed macOS runners the numpy import path can segfault
# (Accelerate polyfit check), which previously crashed any code that merely
# touched dspy_agent.monitor. Downstream callers that truly need the
# PerformanceMonitor can import dspy_agent.monitor.performance_monitor
# explicitly; the top-level package now exposes a placeholder to remain
# backwards compatible.
PerformanceMonitor = None  # type: ignore

__all__ = ['AutoScaler', 'PerformanceMonitor']
