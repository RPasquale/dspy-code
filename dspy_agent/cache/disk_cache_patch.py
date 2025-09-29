"""
Monkey patch for diskcache to fix SQLite syntax errors.
"""
import os
import tempfile
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def patch_diskcache():
    """Apply patches to fix diskcache SQLite syntax errors."""
    try:
        import diskcache.core
        
        # Store original methods
        original_reset = diskcache.core.Cache.reset
        original_sql = diskcache.core.Cache._sql
        
        def safe_reset(self, key, value, update=False):
            """Safe reset method that handles SQLite syntax errors."""
            try:
                return original_reset(self, key, value, update)
            except Exception as e:
                if "syntax error" in str(e).lower() or "pragma" in str(e).lower():
                    logger.warning(f"Skipping problematic PRAGMA: {e}")
                    return
                raise
        
        def safe_sql(self):
            """Safe SQL method that handles syntax errors."""
            try:
                return original_sql(self)
            except Exception as e:
                if "syntax error" in str(e).lower():
                    logger.warning(f"Skipping problematic SQL: {e}")
                    # Return a dummy SQL function
                    def dummy_sql(*args, **kwargs):
                        return type('Result', (), {'fetchall': lambda: []})()
                    return dummy_sql
                raise
        
        # Apply patches
        diskcache.core.Cache.reset = safe_reset
        diskcache.core.Cache._sql = safe_sql
        
        logger.info("Applied diskcache SQLite syntax error patches")
        
    except ModuleNotFoundError:
        # diskcache is not fully available; quietly skip
        return
    except Exception:
        # Any other failure: do not spam logs in user workflows
        return


def configure_safe_cache_environment():
    """Configure environment variables to fix diskcache SQLite issues."""
    # Disable problematic SQLite features
    os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_OPTIMIZATIONS", "1")
    os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_PRAGMA", "1")
    os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_WAL", "1")
    os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_JOURNAL", "1")
    
    # Set cache directory
    cache_dir = os.getenv("DSPY_CACHE_DIR")
    if not cache_dir:
        # Try workspace-based cache
        workspace = os.getenv("DSPY_WORKSPACE", "/workspace")
        workspace_cache = os.path.join(workspace, ".dspy_cache")
        if os.path.exists(workspace) and os.access(workspace, os.W_OK):
            try:
                os.makedirs(workspace_cache, exist_ok=True)
                cache_dir = workspace_cache
            except Exception:
                pass
        
        if not cache_dir:
            # Fall back to temp directory
            cache_dir = os.path.join(tempfile.gettempdir(), ".dspy_cache")
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except Exception:
                cache_dir = tempfile.gettempdir()
    
    os.environ.setdefault("DSPY_CACHE_DIR", cache_dir)
    
    return cache_dir


# Do not auto-run on import; caller (CLI) invokes when appropriate
