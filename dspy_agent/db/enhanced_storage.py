"""
Enhanced storage layer with caching, optimization, and advanced querying for RedDB.

This module provides high-performance data access patterns, caching strategies,
and optimized queries for the DSPy agent's RedDB storage.
"""

from __future__ import annotations

import json
import time
import hashlib
import threading
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .factory import get_storage
from .data_models import (
    RedDBDataManager, EmbeddingVector, SignatureMetrics, VerifierMetrics,
    TrainingMetrics, ActionRecord, LogEntry, ContextState, PatchRecord,
    Environment, ActionType, AgentState, RetrievalEventRecord
)


@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata"""
    data: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_accessed: float = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > (self.timestamp + self.ttl)
    
    def touch(self) -> None:
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache:
    """Thread-safe LRU cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            return entry.data
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in cache"""
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            
            entry = CacheEntry(
                data=value,
                timestamp=time.time(),
                ttl=ttl
            )
            
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Evict oldest items if over capacity
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
    
    def invalidate(self, key: str) -> None:
        """Remove item from cache"""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_accesses = sum(entry.access_count for entry in self._cache.values())
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_accesses": total_accesses,
                "avg_accesses": total_accesses / len(self._cache) if self._cache else 0
            }


@dataclass
class QueryResult:
    """Query result with metadata"""
    data: List[Any]
    total_count: int
    execution_time: float
    cached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QueryBuilder:
    """Builder for complex queries on RedDB data"""
    
    def __init__(self, data_manager: 'EnhancedDataManager'):
        self.dm = data_manager
        self.filters: List[Callable[[Any], bool]] = []
        self.sort_key: Optional[Callable[[Any], Any]] = None
        self.sort_reverse: bool = False
        self.limit_count: Optional[int] = None
        self.offset_count: int = 0
    
    def filter(self, predicate: Callable[[Any], bool]) -> 'QueryBuilder':
        """Add a filter predicate"""
        self.filters.append(predicate)
        return self
    
    def filter_by_field(self, field: str, value: Any) -> 'QueryBuilder':
        """Filter by field value"""
        return self.filter(lambda item: getattr(item, field, None) == value)
    
    def filter_by_range(self, field: str, min_val: Any, max_val: Any) -> 'QueryBuilder':
        """Filter by field range"""
        return self.filter(lambda item: min_val <= getattr(item, field, 0) <= max_val)
    
    def filter_by_time_range(self, field: str, start_time: float, end_time: float) -> 'QueryBuilder':
        """Filter by timestamp range"""
        return self.filter_by_range(field, start_time, end_time)
    
    def sort_by(self, key: Callable[[Any], Any], reverse: bool = False) -> 'QueryBuilder':
        """Sort results"""
        self.sort_key = key
        self.sort_reverse = reverse
        return self
    
    def sort_by_field(self, field: str, reverse: bool = False) -> 'QueryBuilder':
        """Sort by field value"""
        return self.sort_by(lambda item: getattr(item, field, 0), reverse)
    
    def limit(self, count: int) -> 'QueryBuilder':
        """Limit number of results"""
        self.limit_count = count
        return self
    
    def offset(self, count: int) -> 'QueryBuilder':
        """Skip number of results"""
        self.offset_count = count
        return self
    
    def execute(self, data: List[Any]) -> QueryResult:
        """Execute the query on provided data"""
        start_time = time.time()
        
        # Apply filters
        filtered_data = data
        for filter_func in self.filters:
            filtered_data = [item for item in filtered_data if filter_func(item)]
        
        total_count = len(filtered_data)
        
        # Apply sorting
        if self.sort_key:
            filtered_data.sort(key=self.sort_key, reverse=self.sort_reverse)
        
        # Apply offset and limit
        end_idx = None if self.limit_count is None else self.offset_count + self.limit_count
        result_data = filtered_data[self.offset_count:end_idx]
        
        execution_time = time.time() - start_time
        
        return QueryResult(
            data=result_data,
            total_count=total_count,
            execution_time=execution_time
        )


class EnhancedDataManager(RedDBDataManager):
    """Enhanced data manager with caching and optimized queries"""
    
    def __init__(self, namespace: str = "dspy_agent", cache_size: int = 1000, cache_ttl: float = 3600):
        super().__init__(namespace)
        self.cache = LRUCache(max_size=cache_size, default_ttl=cache_ttl)
        self.query_cache = LRUCache(max_size=500, default_ttl=300)  # Shorter TTL for queries
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Start cache cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cache_cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _cache_cleanup_worker(self) -> None:
        """Background thread to clean up expired cache entries"""
        while True:
            try:
                time.sleep(300)  # Clean up every 5 minutes
                self.cache.cleanup_expired()
                self.query_cache.cleanup_expired()
            except Exception:
                pass
    
    def _get_cache_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key"""
        return f"{prefix}:{identifier}"
    
    def _get_query_cache_key(self, query_type: str, params: Dict[str, Any]) -> str:
        """Generate query cache key from parameters"""
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"query:{query_type}:{params_hash}"
    
    # Enhanced signature operations with caching
    def get_signature_metrics(self, signature_name: str) -> Optional[SignatureMetrics]:
        """Get signature metrics with caching"""
        cache_key = self._get_cache_key("signature", signature_name)
        
        # Try cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Fetch from storage
        metrics = super().get_signature_metrics(signature_name)
        if metrics:
            self.cache.put(cache_key, metrics)
        
        return metrics
    
    def store_signature_metrics(self, metrics: SignatureMetrics) -> None:
        """Store signature metrics and update cache"""
        super().store_signature_metrics(metrics)
        
        # Update cache
        cache_key = self._get_cache_key("signature", metrics.signature_name)
        self.cache.put(cache_key, metrics)
        
        # Invalidate related query cache
        self.query_cache.clear()
    
    def get_signature_performance_trend(self, signature_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance trend for a signature with caching"""
        cache_key = self._get_query_cache_key("signature_trend", {
            "signature_name": signature_name,
            "hours": hours
        })
        
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Fetch trend data
        cutoff = time.time() - (hours * 3600)
        trend_data = []
        
        for offset, data in self.storage.read("signature_metrics", count=1000):
            if (data.get('signature_name') == signature_name and 
                data.get('timestamp', 0) >= cutoff):
                trend_data.append(data)
        
        # Sort by timestamp
        trend_data.sort(key=lambda x: x.get('timestamp', 0))
        
        self.query_cache.put(cache_key, trend_data, ttl=300)  # 5 minute TTL
        return trend_data
    
    # Enhanced action recording with batch operations
    def record_actions_batch(self, actions: List[ActionRecord]) -> None:
        """Record multiple actions efficiently"""
        # Use thread pool for parallel storage
        futures = []
        
        for action in actions:
            future = self._executor.submit(self.record_action, action)
            futures.append(future)
        
        # Wait for all to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error recording action: {e}")
    
    def get_actions_by_type(self, action_type: ActionType, limit: int = 100) -> List[ActionRecord]:
        """Get actions filtered by type with caching"""
        cache_key = self._get_query_cache_key("actions_by_type", {
            "action_type": action_type.value,
            "limit": limit
        })
        
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Build query
        query = QueryBuilder(self)
        query.filter_by_field("action_type", action_type)
        query.sort_by_field("timestamp", reverse=True)
        query.limit(limit)
        
        # Get all recent actions and filter
        recent_actions = self.get_recent_actions(limit * 2)  # Get more to ensure we have enough after filtering
        result = query.execute(recent_actions)
        
        self.query_cache.put(cache_key, result.data, ttl=300)
        return result.data

    # Retrieval events ------------------------------------------------------------------
    def record_retrieval_event(self, event: RetrievalEventRecord) -> None:
        """Record retrieval event and update caches."""

        super().record_retrieval_event(event)
        cache_key = self._get_cache_key("retrieval", event.event_id)
        self.cache.put(cache_key, event)
        self.query_cache.clear()

    def get_recent_retrieval_events(self, limit: int = 50) -> List[RetrievalEventRecord]:
        cache_key = self._get_query_cache_key("retrieval_events", {"limit": limit})
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached
        events = super().get_recent_retrieval_events(limit)
        self.query_cache.put(cache_key, events, ttl=120)
        return events
    
    def get_actions_by_reward_range(self, min_reward: float, max_reward: float, limit: int = 100) -> List[ActionRecord]:
        """Get actions filtered by reward range"""
        cache_key = self._get_query_cache_key("actions_by_reward", {
            "min_reward": min_reward,
            "max_reward": max_reward,
            "limit": limit
        })
        
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached
        
        query = QueryBuilder(self)
        query.filter_by_range("reward", min_reward, max_reward)
        query.sort_by_field("timestamp", reverse=True)
        query.limit(limit)
        
        recent_actions = self.get_recent_actions(limit * 2)
        result = query.execute(recent_actions)
        
        self.query_cache.put(cache_key, result.data, ttl=300)
        return result.data
    
    # Advanced analytics methods
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        cache_key = self._get_query_cache_key("performance_summary", {"hours": hours})
        
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached
        
        cutoff = time.time() - (hours * 3600)
        
        # Gather metrics from multiple streams
        signature_data = []
        verifier_data = []
        action_data = []
        
        # Collect signature metrics
        for offset, data in self.storage.read("signature_metrics", count=1000):
            if data.get('timestamp', 0) >= cutoff:
                signature_data.append(data)
        
        # Collect verifier metrics
        for offset, data in self.storage.read("verifier_metrics", count=1000):
            if data.get('timestamp', 0) >= cutoff:
                verifier_data.append(data)
        
        # Collect action data
        for offset, data in self.storage.read("rl_actions", count=1000):
            if data.get('timestamp', 0) >= cutoff:
                action_data.append(data)
        
        # Calculate summary statistics
        summary = {
            "time_range_hours": hours,
            "signature_performance": {
                "avg_score": sum(d.get('performance_score', 0) for d in signature_data) / len(signature_data) if signature_data else 0,
                "avg_response_time": sum(d.get('avg_response_time', 0) for d in signature_data) / len(signature_data) if signature_data else 0,
                "total_signatures": len(set(d.get('signature_name') for d in signature_data))
            },
            "verifier_performance": {
                "avg_accuracy": sum(d.get('accuracy', 0) for d in verifier_data) / len(verifier_data) if verifier_data else 0,
                "total_checks": sum(d.get('checks_performed', 0) for d in verifier_data),
                "total_issues": sum(d.get('issues_found', 0) for d in verifier_data)
            },
            "action_performance": {
                "total_actions": len(action_data),
                "avg_reward": sum(d.get('reward', 0) for d in action_data) / len(action_data) if action_data else 0,
                "avg_confidence": sum(d.get('confidence', 0) for d in action_data) / len(action_data) if action_data else 0,
                "avg_execution_time": sum(d.get('execution_time', 0) for d in action_data) / len(action_data) if action_data else 0
            },
            "timestamp": time.time()
        }
        
        self.query_cache.put(cache_key, summary, ttl=600)  # 10 minute TTL
        return summary
    
    def get_learning_progress(self, sessions: int = 10) -> Dict[str, Any]:
        """Get learning progress over recent training sessions"""
        cache_key = self._get_query_cache_key("learning_progress", {"sessions": sessions})
        
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Get recent training history
        history = self.get_training_history(limit=sessions)
        
        if not history:
            return {"error": "No training history available"}
        
        # Calculate progress metrics
        training_accuracies = [h.training_accuracy for h in history]
        validation_accuracies = [h.validation_accuracy for h in history]
        losses = [h.loss for h in history]
        
        progress = {
            "sessions_analyzed": len(history),
            "training_accuracy": {
                "current": training_accuracies[-1] if training_accuracies else 0,
                "avg": sum(training_accuracies) / len(training_accuracies) if training_accuracies else 0,
                "trend": "improving" if len(training_accuracies) > 1 and training_accuracies[-1] > training_accuracies[0] else "declining"
            },
            "validation_accuracy": {
                "current": validation_accuracies[-1] if validation_accuracies else 0,
                "avg": sum(validation_accuracies) / len(validation_accuracies) if validation_accuracies else 0,
                "trend": "improving" if len(validation_accuracies) > 1 and validation_accuracies[-1] > validation_accuracies[0] else "declining"
            },
            "loss": {
                "current": losses[-1] if losses else 0,
                "avg": sum(losses) / len(losses) if losses else 0,
                "trend": "improving" if len(losses) > 1 and losses[-1] < losses[0] else "declining"
            },
            "timestamp": time.time()
        }
        
        self.query_cache.put(cache_key, progress, ttl=300)
        return progress
    
    # Embedding operations with similarity search
    def find_similar_embeddings(self, query_vector: List[float], top_k: int = 10, 
                               source_type: Optional[str] = None) -> List[Tuple[EmbeddingVector, float]]:
        """Find similar embeddings using cosine similarity"""
        # This would be more efficient with a proper vector database
        # For now, we'll do a linear search through stored embeddings
        
        cache_key = self._get_query_cache_key("similar_embeddings", {
            "query_hash": hashlib.md5(json.dumps(query_vector).encode()).hexdigest()[:16],
            "top_k": top_k,
            "source_type": source_type
        })
        
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # In a real implementation, you'd want to maintain an index of embeddings
        # For now, this is a placeholder that would need to be implemented
        # based on your specific embedding storage strategy
        
        similar_embeddings = []  # Placeholder
        
        self.query_cache.put(cache_key, similar_embeddings, ttl=600)
        return similar_embeddings
    
    # Cache management methods
    def clear_cache(self) -> None:
        """Clear all caches"""
        self.cache.clear()
        self.query_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "main_cache": self.cache.stats(),
            "query_cache": self.query_cache.stats()
        }
    
    def warm_cache(self, signatures: List[str] = None, verifiers: List[str] = None) -> None:
        """Pre-warm cache with commonly accessed data"""
        if signatures:
            for sig_name in signatures:
                self.get_signature_metrics(sig_name)
        
        if verifiers:
            for ver_name in verifiers:
                self.get_verifier_metrics(ver_name)
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Global enhanced data manager instance
_enhanced_data_manager: Optional[EnhancedDataManager] = None

def get_enhanced_data_manager() -> EnhancedDataManager:
    """Get the global enhanced data manager instance"""
    global _enhanced_data_manager
    if _enhanced_data_manager is None:
        _enhanced_data_manager = EnhancedDataManager()
    return _enhanced_data_manager


# Utility functions for common query patterns
def get_top_performing_signatures(limit: int = 10) -> List[SignatureMetrics]:
    """Get top performing signatures"""
    dm = get_enhanced_data_manager()
    all_signatures = dm.get_all_signature_metrics()
    
    query = QueryBuilder(dm)
    query.sort_by_field("performance_score", reverse=True)
    query.limit(limit)
    
    result = query.execute(all_signatures)
    return result.data


def get_recent_high_reward_actions(min_reward: float = 0.8, hours: int = 24) -> List[ActionRecord]:
    """Get recent actions with high rewards"""
    dm = get_enhanced_data_manager()
    cutoff = time.time() - (hours * 3600)
    
    query = QueryBuilder(dm)
    query.filter_by_range("reward", min_reward, 1.0)
    query.filter_by_range("timestamp", cutoff, time.time())
    query.sort_by_field("timestamp", reverse=True)
    
    recent_actions = dm.get_recent_actions(1000)
    result = query.execute(recent_actions)
    return result.data


def get_error_patterns(hours: int = 24) -> Dict[str, int]:
    """Analyze error patterns from logs"""
    dm = get_enhanced_data_manager()
    error_logs = dm.get_recent_logs(level="ERROR", limit=1000)
    
    cutoff = time.time() - (hours * 3600)
    recent_errors = [log for log in error_logs if log.timestamp >= cutoff]
    
    # Count error patterns by source
    error_patterns = defaultdict(int)
    for log in recent_errors:
        error_patterns[log.source] += 1
    
    return dict(error_patterns)
