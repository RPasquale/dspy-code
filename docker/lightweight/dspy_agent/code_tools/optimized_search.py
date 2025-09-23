from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Set
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class SearchCache:
    """Cache for search results to improve performance."""
    pattern_hash: str
    workspace_hash: str
    timestamp: float
    results: List[Dict]
    total_files: int
    search_time: float


class OptimizedSearchEngine:
    """High-performance search engine with caching and parallel processing."""
    
    def __init__(self, cache_dir: Path = Path(".dspy_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "search_cache.json"
        self.cache: Dict[str, SearchCache] = {}
        self.cache_lock = threading.Lock()
        self.max_cache_age = 3600  # 1 hour
        self.max_results = 1000  # Limit results for performance
        self._load_cache()
    
    def _load_cache(self):
        """Load search cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for key, cache_data in data.items():
                        self.cache[key] = SearchCache(**cache_data)
        except Exception:
            pass
    
    def _save_cache(self):
        """Save search cache to disk."""
        try:
            with self.cache_lock:
                data = {k: asdict(v) for k, v in self.cache.items()}
                with open(self.cache_file, 'w') as f:
                    json.dump(data, f)
        except Exception:
            pass
    
    def _get_cache_key(self, pattern: str, workspace: Path, include_globs: Optional[Sequence[str]], exclude_globs: Optional[Sequence[str]]) -> str:
        """Generate cache key for search parameters."""
        key_data = {
            'pattern': pattern,
            'workspace': str(workspace),
            'include_globs': include_globs or [],
            'exclude_globs': exclude_globs or []
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _get_workspace_hash(self, workspace: Path) -> str:
        """Get hash of workspace to detect changes."""
        try:
            # Get modification time of key files
            key_files = list(workspace.glob("**/*.py"))[:100]  # Sample first 100 Python files
            mtimes = [f.stat().st_mtime for f in key_files if f.exists()]
            return hashlib.md5(str(sorted(mtimes)).encode()).hexdigest()
        except Exception:
            return str(time.time())
    
    def _is_cache_valid(self, cache_entry: SearchCache, workspace: Path) -> bool:
        """Check if cache entry is still valid."""
        if time.time() - cache_entry.timestamp > self.max_cache_age:
            return False
        
        # Check if workspace has changed
        current_hash = self._get_workspace_hash(workspace)
        return current_hash == cache_entry.workspace_hash
    
    def _iter_files_optimized(self, root: Path, include_globs: Sequence[str] | None = None, exclude_globs: Sequence[str] | None = None) -> Iterator[Path]:
        """Optimized file iteration with better filtering."""
        include_globs = include_globs or ["**/*"]
        exclude_globs = exclude_globs or [
            "**/.git/**", "**/.venv/**", "**/node_modules/**", "**/dist/**", "**/build/**",
            "**/.mypy_cache/**", "**/.pytest_cache/**", "**/__pycache__/**",
            "**/*.pyc", "**/*.pyo", "**/*.pyd", "**/.DS_Store", "**/Thumbs.db"
        ]
        
        seen: Set[Path] = set()
        file_count = 0
        max_files = 10000  # Limit files for performance
        
        for pat in include_globs:
            if file_count >= max_files:
                break
            try:
                for p in root.glob(pat):
                    if file_count >= max_files:
                        break
                    if p.is_file():
                        # Quick exclude check
                        skip = False
                        for ex in exclude_globs:
                            if p.match(ex):
                                skip = True
                                break
                        
                        if not skip and p not in seen:
                            seen.add(p)
                            file_count += 1
                            yield p
            except Exception:
                continue
    
    def _search_file_parallel(self, file_path: Path, pattern: str, regex: bool, encoding: str = "utf-8") -> List[Dict]:
        """Search a single file and return results."""
        try:
            # Skip large files
            if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                return []
            
            text = file_path.read_text(encoding=encoding, errors="ignore")
        except Exception:
            return []
        
        rx = re.compile(pattern) if regex else None
        results = []
        
        for i, line in enumerate(text.splitlines(), start=1):
            if len(results) >= 50:  # Limit results per file
                break
            try:
                ok = (rx.search(line) if rx else (pattern in line))
                if ok:
                    results.append({
                        'path': str(file_path),
                        'line_no': i,
                        'line': line.strip(),
                        'relative_path': str(file_path.relative_to(file_path.parents[1]) if len(file_path.parents) > 1 else file_path)
                    })
            except Exception:
                continue
        
        return results
    
    def search_text_optimized(
        self,
        root: Path,
        pattern: str,
        regex: bool = True,
        include_globs: Sequence[str] | None = None,
        exclude_globs: Sequence[str] | None = None,
        encoding: str = "utf-8",
        max_workers: int = 4,
        use_cache: bool = True
    ) -> List[Dict]:
        """Optimized text search with caching and parallel processing."""
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(pattern, root, include_globs, exclude_globs)
            with self.cache_lock:
                if cache_key in self.cache:
                    cache_entry = self.cache[cache_key]
                    if self._is_cache_valid(cache_entry, root):
                        return cache_entry.results[:self.max_results]
        
        start_time = time.time()
        all_results = []
        file_count = 0
        
        # Get files to search
        files_to_search = list(self._iter_files_optimized(root, include_globs, exclude_globs))
        
        # Parallel search
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file search tasks
            future_to_file = {
                executor.submit(self._search_file_parallel, file_path, pattern, regex, encoding): file_path
                for file_path in files_to_search
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                try:
                    results = future.result()
                    all_results.extend(results)
                    file_count += 1
                    
                    # Stop if we have enough results
                    if len(all_results) >= self.max_results:
                        break
                except Exception:
                    continue
        
        # Sort results by path and line number
        all_results.sort(key=lambda x: (x['path'], x['line_no']))
        
        # Limit results
        final_results = all_results[:self.max_results]
        
        # Cache results
        if use_cache:
            cache_entry = SearchCache(
                pattern_hash=hashlib.md5(pattern.encode()).hexdigest(),
                workspace_hash=self._get_workspace_hash(root),
                timestamp=time.time(),
                results=final_results,
                total_files=file_count,
                search_time=time.time() - start_time
            )
            
            with self.cache_lock:
                self.cache[cache_key] = cache_entry
                self._save_cache()
        
        return final_results
    
    def clear_cache(self):
        """Clear the search cache."""
        with self.cache_lock:
            self.cache.clear()
            if self.cache_file.exists():
                self.cache_file.unlink()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        with self.cache_lock:
            total_entries = len(self.cache)
            total_results = sum(len(entry.results) for entry in self.cache.values())
            avg_search_time = sum(entry.search_time for entry in self.cache.values()) / max(total_entries, 1)
            
            return {
                'total_entries': total_entries,
                'total_results': total_results,
                'avg_search_time': avg_search_time,
                'cache_size_mb': self.cache_file.stat().st_size / (1024 * 1024) if self.cache_file.exists() else 0
            }


# Global search engine instance
_search_engine = None

def get_search_engine() -> OptimizedSearchEngine:
    """Get the global search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = OptimizedSearchEngine()
    return _search_engine


def search_text_fast(
    root: Path,
    pattern: str,
    regex: bool = True,
    include_globs: Sequence[str] | None = None,
    exclude_globs: Sequence[str] | None = None,
    encoding: str = "utf-8",
    max_workers: int = 4,
    use_cache: bool = True
) -> List[Dict]:
    """Fast text search using the optimized search engine."""
    engine = get_search_engine()
    return engine.search_text_optimized(
        root, pattern, regex, include_globs, exclude_globs, encoding, max_workers, use_cache
    )


def clear_search_cache():
    """Clear the search cache."""
    engine = get_search_engine()
    engine.clear_cache()


def get_search_cache_stats() -> Dict:
    """Get search cache statistics."""
    engine = get_search_engine()
    return engine.get_cache_stats()
