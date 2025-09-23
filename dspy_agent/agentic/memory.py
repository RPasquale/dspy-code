from __future__ import annotations

"""Minimal knowledge-graph style memory for agentic RL signals."""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Mapping, Optional

from ..db import get_enhanced_data_manager, create_log_entry, create_retrieval_event, Environment

RETRIEVAL_LOG = '.dspy_agentic/retrieval.jsonl'


@dataclass
class AgentKnowledgeGraph:
    nodes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)

    def add_file_hint(self, task: str, file_hint: str, confidence: float) -> None:
        task_key = f"task::{task}"
        file_key = f"file::{file_hint}"
        self._touch(task_key)
        self._touch(file_key)
        edge = (task_key, file_key)
        data = self.edges.setdefault(edge, {"weight": 0.0, "count": 0.0})
        data["weight"] = max(data["weight"], float(confidence))
        data["count"] += 1.0

    def add_reference(self, source: str, target: str, weight: float = 0.0) -> None:
        edge = (source, target)
        self._touch(source)
        self._touch(target)
        data = self.edges.setdefault(edge, {"weight": 0.0, "count": 0.0})
        data["weight"] = max(data.get("weight", 0.0), float(weight))
        data["count"] += 1.0

    def _touch(self, node: str) -> None:
        self.nodes.setdefault(node, {"degree": 0.0})

    def summarise(self) -> Dict[str, float]:
        edge_weights = [v.get("weight", 0.0) for v in self.edges.values()]
        avg_weight = sum(edge_weights) / len(edge_weights) if edge_weights else 0.0
        fanout = {}
        for (src, _), data in self.edges.items():
            fanout[src] = fanout.get(src, 0.0) + data.get("count", 0.0)
        avg_fanout = sum(fanout.values()) / len(fanout) if fanout else 0.0
        return {
            "kg_nodes": float(len(self.nodes)),
            "kg_edges": float(len(self.edges)),
            "kg_avg_weight": avg_weight,
            "kg_avg_fanout": avg_fanout,
        }


def compute_retrieval_features(
    workspace: Path,
    patches: Iterable[Dict[str, object]],
    retrieval_events: Optional[Iterable[Dict[str, object]]] = None,
) -> List[float]:
    kg = AgentKnowledgeGraph()
    precision_samples = []
    coverage_nodes: set[str] = set()
    retrieval_scores: List[float] = []
    query_count = 0.0
    for rec in patches:
        task = str(rec.get('task') or rec.get('prompt_id') or 'task')
        hints = rec.get('file_candidates') or rec.get('file_hints') or ''
        metrics = rec.get('metrics') or {}
        try:
            pass_rate = float(metrics.get('pass_rate', 0.0))
        except Exception:
            pass_rate = 0.0
        candidates = []
        if isinstance(hints, str):
            candidates = [h.strip() for h in hints.split(',') if h.strip()]
        elif isinstance(hints, (list, tuple)):
            candidates = [str(h).strip() for h in hints if str(h).strip()]
        for hint in candidates:
                kg.add_file_hint(task, hint, pass_rate)
                coverage_nodes.add(hint)
        if candidates:
            precision_samples.append(pass_rate)
    if retrieval_events:
        for event in retrieval_events:
            try:
                query = str(event.get('query', ''))
            except Exception:
                query = ''
            hits = event.get('hits') or []
            if not query or not hits:
                continue
            query_key = f"query::{query}"
            for hit in hits:
                try:
                    path = str(hit.get('path') or '')
                except Exception:
                    path = ''
                if not path:
                    continue
                try:
                    score = float(hit.get('score', 0.0))
                except Exception:
                    score = 0.0
                kg.add_reference(query_key, f"file::{path}", weight=score)
                retrieval_scores.append(score)
                coverage_nodes.add(path)
            query_count += 1.0
    summary = kg.summarise()
    precision = sum(precision_samples) / len(precision_samples) if precision_samples else 0.0
    coverage = len(coverage_nodes)
    avg_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
    return [
        summary.get("kg_nodes", 0.0),
        summary.get("kg_edges", 0.0),
        summary.get("kg_avg_weight", 0.0),
        summary.get("kg_avg_fanout", 0.0),
        float(precision),
        float(coverage),
        float(avg_score),
        float(query_count),
    ]


def log_retrieval_event(workspace: Path, query: str, hits: Iterable[Mapping[str, object]], *, limit: int = 20) -> None:
    """Append a retrieval event for downstream agentic features."""

    try:
        ww = workspace.resolve()
    except Exception:
        ww = workspace
    log_path = ww / RETRIEVAL_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record_hits: List[Dict[str, object]] = []
    for idx, hit in enumerate(hits):
        if idx >= limit:
            break
        try:
            path = str(hit.get('path', '') or '')
        except Exception:
            path = ''
        if not path:
            continue
        entry = {
            'path': path,
            'score': float(hit.get('score', 0.0) or 0.0),
            'source': hit.get('source'),
        }
        record_hits.append(entry)
    record_id = str(uuid.uuid4())
    record = {
        'timestamp': time.time(),
        'event_id': record_id,
        'query': query,
        'hits': record_hits,
    }
    try:
        with log_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record) + '\n')
    except Exception:
        pass

    # Mirror to RedDB / enhanced manager
    try:
        dm = get_enhanced_data_manager()
        env_name = os.getenv('DSPY_ENVIRONMENT', 'development').upper()
        try:
            env = Environment[env_name]
        except KeyError:
            env = Environment.DEVELOPMENT
        
        # Create retrieval event using the utility function
        event = create_retrieval_event(
            workspace_path=str(ww),
            query=query,
            hits=record_hits,
            environment=env,
            event_id=record_id
        )
        
        # Record the event in RedDB
        dm.record_retrieval_event(event)
        
        # Also log the event
        log_entry = create_log_entry(
            level="INFO",
            source="retrieval",
            message="retrieval_event",
            context={"query": query, "hits": len(record_hits), "workspace": str(ww)},
            environment=env,
        )
        dm.log(log_entry)
        
    except Exception as e:
        # Log the error but don't fail the function
        import logging
        logging.warning(f"Failed to record retrieval event in RedDB: {e}")
        pass


def load_retrieval_events(workspace: Path, limit: int = 50) -> List[Dict[str, object]]:
    """Load retrieval events from both file and RedDB sources."""
    events: List[Dict[str, object]] = []
    
    # First try to load from RedDB
    try:
        dm = get_enhanced_data_manager()
        # Get recent retrieval events from RedDB
        reddb_events = dm.get_recent_retrieval_events(limit=limit)
        for event_record in reddb_events:
            # Convert RetrievalEventRecord to dict format
            if hasattr(event_record, 'to_dict'):
                events.append(event_record.to_dict())
            elif isinstance(event_record, dict):
                events.append(event_record)
    except Exception as e:
        import logging
        logging.warning(f"Failed to load retrieval events from RedDB: {e}")
    
    # If we don't have enough events from RedDB, try to load from file
    if len(events) < limit:
        log_path = workspace / RETRIEVAL_LOG
        if log_path.exists():
            try:
                lines = log_path.read_text(encoding='utf-8').splitlines()
                file_events: List[Dict[str, object]] = []
                for line in lines[-limit:]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        file_events.append(json.loads(line))
                    except Exception:
                        continue
                
                # Merge file events with RedDB events, avoiding duplicates
                existing_ids = {event.get('event_id') for event in events}
                for file_event in file_events:
                    if file_event.get('event_id') not in existing_ids:
                        events.append(file_event)
                        if len(events) >= limit:
                            break
                            
            except Exception as e:
                import logging
                logging.warning(f"Failed to load retrieval events from file: {e}")
    
    # Sort by timestamp and return the most recent events
    events.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    return events[:limit]


def query_retrieval_events(
    workspace: Path,
    query_filter: Optional[str] = None,
    min_score: Optional[float] = None,
    since_timestamp: Optional[float] = None,
    limit: int = 50
) -> List[Dict[str, object]]:
    """Query retrieval events with filtering capabilities."""
    events = load_retrieval_events(workspace, limit=limit * 2)  # Load more to filter
    
    # Apply filters
    filtered_events = []
    for event in events:
        # Filter by query text
        if query_filter and query_filter.lower() not in event.get('query', '').lower():
            continue
            
        # Filter by minimum score
        if min_score is not None:
            hits = event.get('hits', [])
            max_hit_score = max((hit.get('score', 0) for hit in hits), default=0)
            if max_hit_score < min_score:
                continue
                
        # Filter by timestamp
        if since_timestamp is not None and event.get('timestamp', 0) < since_timestamp:
            continue
            
        filtered_events.append(event)
        if len(filtered_events) >= limit:
            break
    
    return filtered_events


def get_retrieval_statistics(workspace: Path) -> Dict[str, float]:
    """Get statistics about retrieval events for the workspace."""
    events = load_retrieval_events(workspace, limit=1000)  # Load more for stats
    
    if not events:
        return {
            "total_events": 0.0,
            "avg_hits_per_query": 0.0,
            "avg_score": 0.0,
            "unique_queries": 0.0,
            "unique_files": 0.0,
        }
    
    total_events = len(events)
    total_hits = sum(len(event.get('hits', [])) for event in events)
    all_scores = []
    unique_queries = set()
    unique_files = set()
    
    for event in events:
        unique_queries.add(event.get('query', ''))
        for hit in event.get('hits', []):
            all_scores.append(hit.get('score', 0))
            unique_files.add(hit.get('path', ''))
    
    return {
        "total_events": float(total_events),
        "avg_hits_per_query": float(total_hits) / float(total_events) if total_events > 0 else 0.0,
        "avg_score": float(sum(all_scores)) / float(len(all_scores)) if all_scores else 0.0,
        "unique_queries": float(len(unique_queries)),
        "unique_files": float(len(unique_files)),
    }
