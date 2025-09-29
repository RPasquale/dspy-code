"""
Custom cache configuration to fix diskcache SQLite issues.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional


def get_safe_cache_directory() -> str:
    """Get a safe cache directory that avoids SQLite syntax errors."""
    # Check for explicit cache directory
    explicit = os.getenv("DSPY_CACHE_DIR")
    if explicit:
        return explicit
    
    # Try workspace-based cache
    workspace = os.getenv("DSPY_WORKSPACE", "/workspace")
    workspace_cache = os.path.join(workspace, ".dspy_cache")
    if os.path.exists(workspace) and os.access(workspace, os.W_OK):
        try:
            os.makedirs(workspace_cache, exist_ok=True)
            return workspace_cache
        except Exception:
            pass
    
    # Fall back to temp directory
    temp_cache = os.path.join(tempfile.gettempdir(), ".dspy_cache")
    try:
        os.makedirs(temp_cache, exist_ok=True)
        return temp_cache
    except Exception:
        return tempfile.gettempdir()


def configure_diskcache_environment():
    """Configure environment variables to fix diskcache SQLite issues."""
    # Disable problematic SQLite features
    os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_OPTIMIZATIONS", "1")
    os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_PRAGMA", "1")
    os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_WAL", "1")
    os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_JOURNAL", "1")
    
    # Set cache directory
    cache_dir = get_safe_cache_directory()
    os.environ.setdefault("DSPY_CACHE_DIR", cache_dir)
    
    return cache_dir


# Configure environment on import
configure_diskcache_environment()
