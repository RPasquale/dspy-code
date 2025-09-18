"""Compatibility shim for legacy imports.

This module previously lived at the package root; tests and external tooling
still import ``dspy_agent.streaming_runtime``. Re-export the modern streaming
runtime here so those imports succeed.
"""

from .streaming.streaming_runtime import *  # noqa: F401,F403

