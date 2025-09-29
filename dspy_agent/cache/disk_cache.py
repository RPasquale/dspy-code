"""
Custom diskcache wrapper to fix SQLite syntax errors.
"""
import os
import tempfile
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class SafeDiskCache:
    """A diskcache wrapper that handles SQLite syntax errors gracefully."""
    
    def __init__(self, directory: Optional[str] = None, **kwargs):
        self.directory = directory or self._get_safe_directory()
        self._cache = None
        self._init_cache(**kwargs)
    
    def _get_safe_directory(self) -> str:
        """Get a safe cache directory."""
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
    
    def _init_cache(self, **kwargs):
        """Initialize the cache with error handling."""
        try:
            # Set environment variables to disable problematic SQLite features
            os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_OPTIMIZATIONS", "1")
            os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_PRAGMA", "1")
            os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_WAL", "1")
            os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_JOURNAL", "1")
            
            # Try to import and create diskcache
            try:
                from diskcache import Cache
                self._cache = Cache(self.directory, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to create diskcache: {e}")
                # Fall back to a simple in-memory cache
                self._cache = {}
                
        except Exception as e:
            logger.error(f"Cache initialization failed: {e}")
            self._cache = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the cache."""
        if self._cache is None:
            return default
        
        try:
            if hasattr(self._cache, 'get'):
                return self._cache.get(key, default)
            else:
                return self._cache.get(key, default)
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, **kwargs) -> bool:
        """Set a value in the cache."""
        if self._cache is None:
            return False
        
        try:
            if hasattr(self._cache, 'set'):
                return self._cache.set(key, value, **kwargs)
            else:
                self._cache[key] = value
                return True
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        if self._cache is None:
            return False
        
        try:
            if hasattr(self._cache, 'delete'):
                return self._cache.delete(key)
            else:
                if key in self._cache:
                    del self._cache[key]
                    return True
                return False
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear the cache."""
        if self._cache is None:
            return False
        
        try:
            if hasattr(self._cache, 'clear'):
                self._cache.clear()
            else:
                self._cache.clear()
            return True
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")
            return False
    
    def close(self):
        """Close the cache."""
        if self._cache is not None and hasattr(self._cache, 'close'):
            try:
                self._cache.close()
            except Exception as e:
                logger.warning(f"Cache close failed: {e}")


# Global cache instance
_global_cache: Optional[SafeDiskCache] = None


def get_cache() -> SafeDiskCache:
    """Get the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SafeDiskCache()
    return _global_cache


def configure_cache_environment():
    """Configure environment variables to fix diskcache SQLite issues."""
    os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_OPTIMIZATIONS", "1")
    os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_PRAGMA", "1")
    os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_WAL", "1")
    os.environ.setdefault("DISKCACHE_DISABLE_SQLITE_JOURNAL", "1")
    
    # Set cache directory
    cache_dir = os.getenv("DSPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), ".dspy_cache"))
    os.environ.setdefault("DSPY_CACHE_DIR", cache_dir)
    
    return cache_dir


# Configure environment on import
configure_cache_environment()
