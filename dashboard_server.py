#!/usr/bin/env python3
"""Compatibility launcher for the DSPy React monitoring dashboard."""

from enhanced_dashboard_server import (
    EnhancedDashboardHandler as DashboardHandler,
    start_enhanced_dashboard_server,
)


def start_dashboard_server(port: int = 8080) -> None:
    """Start the dashboard server (delegates to the enhanced handler)."""
    start_enhanced_dashboard_server(port)


if __name__ == "__main__":
    import sys

    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    start_dashboard_server(port_arg)
