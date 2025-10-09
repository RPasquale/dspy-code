"""Build a project activity graph from RedDB actions, issues, contributors, releases."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..db.redb_router import RedDBRouter
from ..db.enhanced_storage import EnhancedDataManager


@dataclass
class ActivityNode:
    id: str
    kind: str
    metadata: Dict[str, any]


@dataclass
class ActivityEdge:
    src: str
    dst: str
    kind: str
    metadata: Dict[str, any]


ACTION_EDGE_KIND = 'action_to_file'
ISSUE_EDGE_KIND = 'issue_to_action'
CONTRIBUTOR_EDGE_KIND = 'contributor_to_action'
RELEASE_EDGE_KIND = 'release_to_snapshot'


def _action_nodes(dm: EnhancedDataManager, limit: int = 500) -> Iterable[ActivityNode]:
    actions = dm.get_recent_actions(limit=limit)
    for action in actions:
        node_id = getattr(action, 'action_id', None)
        if not node_id:
            continue
        metadata = {
            'timestamp': getattr(action, 'timestamp', None),
            'reward': getattr(action, 'reward', None),
            'confidence': getattr(action, 'confidence', None),
            'environment': getattr(action, 'environment', None).value if getattr(action, 'environment', None) else None,
            'action_type': getattr(action, 'action_type', None).value if getattr(action, 'action_type', None) else None,
        }
        yield ActivityNode(id=f'action:{node_id}', kind='action', metadata=metadata)


def _action_edges(action: ActivityNode, action_record, workspace: Path) -> Iterable[ActivityEdge]:
    parameters = getattr(action_record, 'parameters', {}) if action_record else {}
    result = getattr(action_record, 'result', {}) if action_record else {}
    associated_files = set()
    for container in (parameters, result):
        if not isinstance(container, dict):
            continue
        for key in ('relative_path', 'path', 'doc_id', 'file'):
            val = container.get(key)
            if isinstance(val, str):
                associated_files.add(val.replace(str(workspace), '').lstrip('/'))
    for file_path in associated_files:
        yield ActivityEdge(
            src=action.id,
            dst=f'code:{file_path}',
            kind=ACTION_EDGE_KIND,
            metadata={
                'workspace': str(workspace),
            },
        )


def _issue_nodes(root: Path) -> Iterable[ActivityNode]:
    issue_file = root / 'project_issues.json'
    if not issue_file.exists():
        return []
    try:
        payload = json.loads(issue_file.read_text())
    except Exception:
        return []
    nodes = []
    for issue_id, data in payload.items() if isinstance(payload, dict) else []:
        nodes.append(ActivityNode(id=f'issue:{issue_id}', kind='issue', metadata=data if isinstance(data, dict) else {}))
    return nodes


def _release_nodes(root: Path) -> Iterable[ActivityNode]:
    releases_file = root / 'project_releases.json'
    if not releases_file.exists():
        return []
    try:
        payload = json.loads(releases_file.read_text())
    except Exception:
        return []
    nodes = []
    for rel_id, data in payload.items() if isinstance(payload, dict) else []:
        nodes.append(ActivityNode(id=f'release:{rel_id}', kind='release', metadata=data if isinstance(data, dict) else {}))
    return nodes


def build_project_activity_graph(
    namespace: str,
    *,
    workspace: Optional[str] = None,
    router: Optional[RedDBRouter] = None,
    dm: Optional[EnhancedDataManager] = None,
    action_limit: int = 500,
) -> Dict[str, List[Dict[str, any]]]:
    router = router or RedDBRouter()
    dm = dm or EnhancedDataManager(namespace=namespace)
    root = Path(workspace or os.getenv('DSPY_WORKSPACE') or '.').resolve()

    nodes: Dict[str, ActivityNode] = {}
    edges: List[ActivityEdge] = []

    for action_rec in dm.get_recent_actions(limit=action_limit):
        node = ActivityNode(
            id=f'action:{action_rec.action_id}',
            kind='action',
            metadata={
                'timestamp': action_rec.timestamp,
                'reward': action_rec.reward,
                'environment': getattr(action_rec.environment, 'value', None),
                'action_type': getattr(action_rec.action_type, 'value', None),
            },
        )
        nodes[node.id] = node
        edges.extend(_action_edges(node, action_rec, root))

        contributor = getattr(action_rec, 'parameters', {}).get('actor')
        if isinstance(contributor, str):
            c_id = f'contributor:{contributor}'
            nodes.setdefault(c_id, ActivityNode(id=c_id, kind='contributor', metadata={'name': contributor}))
            edges.append(ActivityEdge(src=c_id, dst=node.id, kind=CONTRIBUTOR_EDGE_KIND, metadata={}))

    for issue_node in _issue_nodes(root):
        nodes[issue_node.id] = issue_node
        linked_actions = issue_node.metadata.get('actions') if isinstance(issue_node.metadata, dict) else None
        for action_id in linked_actions or []:
            if f'action:{action_id}' in nodes:
                edges.append(ActivityEdge(src=issue_node.id, dst=f'action:{action_id}', kind=ISSUE_EDGE_KIND, metadata={}))

    for release_node in _release_nodes(root):
        nodes[release_node.id] = release_node
        snapshot = release_node.metadata.get('snapshot') if isinstance(release_node.metadata, dict) else None
        if snapshot:
            edges.append(ActivityEdge(src=release_node.id, dst=f'snapshot:{snapshot}', kind=RELEASE_EDGE_KIND, metadata={}))

    timestamp = time.time()
    graph_payload = {
        'generated_at': timestamp,
        'nodes': [asdict(node) for node in nodes.values()],
        'edges': [asdict(edge) for edge in edges],
    }
    router.st.put(router._k(namespace, 'project_graph', 'latest'), graph_payload)
    history_key = router._k(namespace, 'project_graph', 'history')
    history = router.st.get(history_key) or []
    history.append(timestamp)
    router.st.put(history_key, history[-20:])
    router.st.put(router._k(namespace, 'project_graph', 'snapshot', str(int(timestamp))), graph_payload)
    return graph_payload


__all__ = ['build_project_activity_graph']
