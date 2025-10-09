from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..db.redb_router import RedDBRouter
from ..db.enhanced_storage import EnhancedDataManager


@dataclass
class _NodeStats:
    visits: int = 0
    value: float = 0.0


def _collect_graph(router: RedDBRouter, namespace: str) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, Dict[str, Any]]]:
    adjacency: Dict[str, List[Tuple[str, float]]] = {}
    node_cache: Dict[str, Dict[str, Any]] = {}

    node_labels = router.st.get(f"{namespace}:graph:_node_labels") or []
    for label in node_labels:
        idx_key = router._k(namespace, "collection", f"graph::node::{label}", "_ids")
        ids = router.st.get(idx_key) or []
        for node_id in ids:
            rec = router.st.get(router._k(namespace, "graph", "node", label, node_id)) or {}
            node_cache[node_id] = rec

    edge_labels = router.st.get(f"{namespace}:graph:_edge_labels") or []
    for label in edge_labels:
        idx_key = router._k(namespace, "collection", f"graph::edge::{label}", "_ids")
        ids = router.st.get(idx_key) or []
        for edge_id in ids:
            rec = router.st.get(router._k(namespace, "graph", "edge", label, edge_id)) or {}
            src = rec.get('src')
            dst = rec.get('dst')
            if not src or not dst:
                continue
            props = rec.get('props') or {}
            weight = float(props.get('weight', 1.0))
            adjacency.setdefault(src, []).append((dst, weight))
    return adjacency, node_cache


def _collect_node_rewards(dm: EnhancedDataManager, limit: int = 1000) -> Dict[str, float]:
    rewards: Dict[str, float] = {}
    try:
        actions = dm.get_recent_actions(limit=limit)
    except Exception:
        return rewards
    for action in actions:
        reward = float(getattr(action, 'reward', 0.0) or 0.0)
        for container in (getattr(action, 'parameters', {}), getattr(action, 'result', {})):
            if not isinstance(container, dict):
                continue
            for key in ('relative_path', 'path', 'doc_id', 'file'):  # heuristic
                val = container.get(key)
                if isinstance(val, str):
                    rewards[val] = rewards.get(val, 0.0) + reward
    return rewards


def run_mcts_memory_refresh(
    namespace: str,
    *,
    iterations: int = 400,
    depth: int = 4,
    exploration: float = 1.2,
    reward_map: Optional[Dict[str, float]] = None,
    router: Optional[RedDBRouter] = None,
) -> Dict[str, float]:
    """Run a lightweight Monte Carlo search over the workspace graph and persist priorities."""

    router = router or RedDBRouter()
    adjacency, node_cache = _collect_graph(router, namespace)
    if not node_cache:
        return {}

    if reward_map is None:
        dm = EnhancedDataManager(namespace=namespace)
        reward_map = _collect_node_rewards(dm)

    stats: Dict[str, _NodeStats] = {node: _NodeStats() for node in node_cache}
    nodes = list(node_cache.keys())

    for _ in range(max(iterations, 1)):
        start = random.choice(nodes)
        path_nodes: List[str] = [start]
        total_reward = reward_map.get(start, 0.0)
        visits = 1
        current = start
        for depth_step in range(depth):
            neighbors = adjacency.get(current)
            if not neighbors:
                break
            weights = [1.0 / max(weight, 1e-6) for (_, weight) in neighbors]
            next_node = random.choices([n for n, _ in neighbors], weights=weights, k=1)[0]
            path_nodes.append(next_node)
            total_reward += reward_map.get(next_node, 0.0)
            visits += 1
            current = next_node
        avg_reward = total_reward / max(visits, 1)
        for node in path_nodes:
            stat = stats.setdefault(node, _NodeStats())
            stat.visits += 1
            stat.value += avg_reward

    priorities: Dict[str, float] = {}
    for node, stat in stats.items():
        if stat.visits == 0:
            continue
        priorities[node] = stat.value / stat.visits

    timestamp = time.time()
    for node_id, priority in priorities.items():
        rec = node_cache.get(node_id)
        if not rec:
            continue
        label = rec.get('label', 'code_file')
        updated = dict(rec)
        updated['mcts_priority'] = priority
        updated['mcts_updated_at'] = timestamp
        router.upsert_node(namespace, label, updated)

    return priorities
