from __future__ import annotations

"""Minimal knowledge-graph style memory for agentic RL signals."""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Mapping, Optional

from ..db import get_enhanced_data_manager, create_log_entry, Environment, RetrievalEventRecord

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

    # Mirror to RedDB / enhanced manager when available
    try:
        from ..db import get_enhanced_data_manager, create_log_entry, Environment
        from ..db.data_models import RetrievalEventRecord

        dm = get_enhanced_data_manager()
        env_name = os.getenv('DSPY_ENVIRONMENT', 'development').upper()
        env = Environment[env_name] if env_name in Environment.__members__ else Environment.DEVELOPMENT
        event = RetrievalEventRecord(
            event_id=record_id,
            timestamp=record['timestamp'],
            workspace_path=str(ww),
            query=query,
            hits=record_hits,
            environment=env,
        )
        dm.record_retrieval_event(event)
        log_entry = create_log_entry(
            level="INFO",
            source="retrieval",
            message="retrieval_event",
            context={"query": query, "hits": len(record_hits), "workspace": str(ww)},
            environment=env,
        )
        dm.log(log_entry)
    except Exception:
        pass


def load_retrieval_events(workspace: Path, limit: int = 50) -> List[Dict[str, object]]:
    log_path = workspace / RETRIEVAL_LOG
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding='utf-8').splitlines()
    except Exception:
        return []
    events: List[Dict[str, object]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events
