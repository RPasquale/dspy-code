from __future__ import annotations

"""Minimal knowledge-graph style memory for agentic RL signals."""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Mapping, Optional, Any

from ..db import get_enhanced_data_manager, create_log_entry, create_retrieval_event, Environment

RETRIEVAL_LOG = '.dspy_agentic/retrieval.jsonl'
GRAPH_MEMORY_FILE = '.dspy_agentic/graph_memory.json'


@dataclass
class AgentKnowledgeGraph:
    nodes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)

    def add_file_hint(self, task: str, file_hint: str, confidence: float) -> None:
        task_key = f"task::{task}"
        file_key = f"file::{file_hint}"
        self._touch(task_key, degree_increment=1.0)
        self._touch(file_key, degree_increment=1.0)
        edge = (task_key, file_key)
        data = self.edges.setdefault(edge, {"weight": 0.0, "count": 0.0})
        prev_count = float(data.get("count", 0.0))
        new_count = prev_count + 1.0
        prev_weight = float(data.get("weight", 0.0))
        # Exponential moving average to emphasise recent confidence
        alpha = min(1.0, 0.6 + (1.0 / max(new_count, 1.0)))
        updated_weight = (1 - alpha) * prev_weight + alpha * float(confidence)
        data["weight"] = max(0.0, min(1.0, updated_weight))
        data["count"] = new_count
        data["last_confidence"] = float(confidence)
        data["updated_at"] = time.time()

    def add_reference(self, source: str, target: str, weight: float = 0.0) -> None:
        edge = (source, target)
        self._touch(source, degree_increment=1.0)
        self._touch(target, degree_increment=1.0)
        data = self.edges.setdefault(edge, {"weight": 0.0, "count": 0.0})
        prev_count = float(data.get("count", 0.0))
        new_count = prev_count + 1.0
        prev_weight = float(data.get("weight", 0.0))
        alpha = min(1.0, 0.5 + (1.0 / max(new_count, 1.0)))
        updated_weight = (1 - alpha) * prev_weight + alpha * float(weight)
        data["weight"] = max(0.0, min(1.0, updated_weight))
        data["count"] = new_count
        data["last_score"] = float(weight)
        data["updated_at"] = time.time()

    def _touch(self, node: str, *, degree_increment: float = 0.0) -> None:
        rec = self.nodes.setdefault(node, {"degree": 0.0})
        rec["degree"] = float(rec.get("degree", 0.0)) + float(degree_increment)
        rec["last_seen"] = time.time()

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

    def top_targets(self, *, prefix: str, limit: int = 10) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for (_src, dst), data in self.edges.items():
            if not dst.startswith(prefix):
                continue
            weight = float(data.get("weight", 0.0))
            if weight <= 0.0:
                continue
            name = dst[len(prefix):]
            scores[name] = max(scores.get(name, 0.0), weight)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ordered[:limit]

    def to_payload(self) -> Dict[str, Any]:
        edges = []
        for (src, dst), data in self.edges.items():
            payload = dict(data)
            payload["src"] = src
            payload["dst"] = dst
            edges.append(payload)
        return {"nodes": self.nodes, "edges": edges}

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "AgentKnowledgeGraph":
        nodes = {}
        edges: Dict[Tuple[str, str], Dict[str, float]] = {}
        if isinstance(data.get("nodes"), dict):
            nodes = {str(k): dict(v) for k, v in data["nodes"].items()}  # type: ignore[arg-type]
        for raw in data.get("edges", []) or []:
            if not isinstance(raw, Mapping):
                continue
            src = str(raw.get("src") or "")
            dst = str(raw.get("dst") or "")
            if not src or not dst:
                continue
            payload = {k: v for k, v in raw.items() if k not in {"src", "dst"}}
            edges[(src, dst)] = payload  # type: ignore[assignment]
        return cls(nodes=nodes, edges=edges)


class GraphMemoryStore:
    """Disk-backed store for the agent knowledge graph with lazy loading."""

    def __init__(self, workspace: Path, *, filename: str = GRAPH_MEMORY_FILE) -> None:
        try:
            self.workspace = Path(workspace).resolve()
        except Exception:
            self.workspace = Path(workspace)
        self.path = self.workspace / filename
        self._graph: Optional[AgentKnowledgeGraph] = None
        self._meta: Dict[str, set[str]] = {
            "seen_events": set(),
            "seen_patches": set(),
        }

    def _ensure_loaded(self) -> AgentKnowledgeGraph:
        if self._graph is not None:
            return self._graph
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                payload = data.get("graph") if isinstance(data, Mapping) else None
                if not isinstance(payload, Mapping):
                    payload = data
                self._graph = AgentKnowledgeGraph.from_payload(payload)
                meta = data.get("meta") if isinstance(data, Mapping) else {}
                if isinstance(meta, Mapping):
                    seen_events = meta.get("seen_events", [])
                    seen_patches = meta.get("seen_patches", [])
                    self._meta["seen_events"] = {str(x) for x in seen_events if x}
                    self._meta["seen_patches"] = {str(x) for x in seen_patches if x}
            except Exception:
                self._graph = AgentKnowledgeGraph()
        else:
            self._graph = AgentKnowledgeGraph()
        return self._graph

    def graph(self) -> AgentKnowledgeGraph:
        return self._ensure_loaded()

    def flush(self) -> None:
        if self._graph is None:
            return
        payload = {
            "graph": self._graph.to_payload(),
            "updated_at": time.time(),
            "workspace": str(self.workspace),
            "meta": {
                "seen_events": sorted(self._meta.get("seen_events", [])),
                "seen_patches": sorted(self._meta.get("seen_patches", [])),
            },
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    def update_from_signals(
        self,
        *,
        patches: Iterable[Mapping[str, object]] = (),
        retrieval_events: Iterable[Mapping[str, object]] = (),
    ) -> AgentKnowledgeGraph:
        graph = self.graph()
        seen_events = self._meta.setdefault("seen_events", set())
        seen_patches = self._meta.setdefault("seen_patches", set())
        for rec in patches or []:
            task = str(rec.get("task") or rec.get("prompt_id") or rec.get("prompt_hash") or "task")
            patch_id = str(rec.get("patch_id") or rec.get("prompt_hash") or rec.get("task") or "")
            patch_key = patch_id or task
            if patch_key and patch_key in seen_patches:
                continue
            hints = rec.get("file_candidates") or rec.get("file_hints") or ""
            metrics = rec.get("metrics") or {}
            try:
                confidence = float(metrics.get("pass_rate", 0.0))
            except Exception:
                confidence = 0.0
            candidates: List[str] = []
            if isinstance(hints, str):
                candidates = [h.strip() for h in hints.split(",") if h.strip()]
            elif isinstance(hints, (list, tuple)):
                candidates = [str(h).strip() for h in hints if str(h).strip()]
            for hint in candidates:
                graph.add_file_hint(task, hint, confidence)
            if patch_key:
                seen_patches.add(patch_key)
        for event in retrieval_events or []:
            query = str(event.get("query") or "")
            if not query:
                continue
            event_id = str(event.get("event_id") or "")
            if not event_id:
                event_id = f"{query}::{event.get('timestamp', '')}"
            if event_id and event_id in seen_events:
                continue
            query_key = f"query::{query}"
            hits = event.get("hits") or []
            for hit in hits:
                try:
                    path = str(hit.get("path") or "")
                except Exception:
                    path = ""
                if not path:
                    continue
                try:
                    score = float(hit.get("score", 0.0))
                except Exception:
                    score = 0.0
                graph.add_reference(query_key, f"file::{path}", weight=score)
            if event_id:
                seen_events.add(event_id)
        self.flush()
        return graph

    def summary(self, *, limit: int = 8) -> Dict[str, Any]:
        graph = self.graph()
        top_files = graph.top_targets(prefix="file::", limit=limit)
        top_queries = graph.top_targets(prefix="query::", limit=limit)
        return {
            "top_files": [{"path": path, "confidence": score} for path, score in top_files],
            "top_queries": [{"query": query, "strength": score} for query, score in top_queries],
            "totals": graph.summarise(),
        }


def compute_retrieval_features(
    workspace: Path,
    patches: Iterable[Dict[str, object]],
    retrieval_events: Optional[Iterable[Dict[str, object]]] = None,
    *,
    update_memory: bool = True,
) -> List[float]:
    store = GraphMemoryStore(workspace)
    if update_memory:
        kg = store.update_from_signals(patches=patches, retrieval_events=retrieval_events or [])
    else:
        kg = store.graph()
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
                retrieval_scores.append(score)
                coverage_nodes.add(path)
            query_count += 1.0
    summary = kg.summarise()
    precision = sum(precision_samples) / len(precision_samples) if precision_samples else 0.0
    if not coverage_nodes:
        coverage_nodes = {name for name, _ in kg.top_targets(prefix="file::", limit=128)}
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
        float(query_count) if query_count else float(len(kg.top_targets(prefix="query::", limit=256))),
    ]


def summarize_graph_memory(workspace: Path, *, limit: int = 8) -> Dict[str, Any]:
    """Return a cached summary of the graph memory for UI/prompt consumption."""
    store = GraphMemoryStore(workspace)
    return store.summary(limit=limit)


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

    # Update persistent graph memory
    try:
        GraphMemoryStore(ww).update_from_signals(
            patches=[],
            retrieval_events=[record],
        )
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
