"""Graph analytics helpers (centrality, clustering)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def pagerank(adjacency: Dict[str, List[Tuple[str, float]]], damping: float = 0.85, iterations: int = 30) -> Dict[str, float]:
    nodes = set(adjacency.keys())
    for neighbors in list(adjacency.values()):
        for dest, _ in neighbors:
            nodes.add(dest)
    for node in nodes:
        adjacency.setdefault(node, [])
    nodes = list(adjacency.keys())
    n = len(nodes)
    if n == 0:
        return {}
    rank = {node: 1.0 / n for node in nodes}
    inv = {node: [(src, weight) for src, outs in adjacency.items() for dest, weight in outs if dest == node] for node in nodes}
    for _ in range(iterations):
        new_rank = {}
        for node in nodes:
            inbound = inv.get(node, [])
            total = (1.0 - damping) / n
            for src, weight in inbound:
                out_weight = sum(w for _, w in adjacency[src]) or 1.0
                total += damping * (rank[src] * weight / out_weight)
            new_rank[node] = total
        rank = new_rank
    return rank


def connected_components(adjacency: Dict[str, List[Tuple[str, float]]]) -> Dict[str, int]:
    component_index = {}
    visited = set()
    component_id = 0

    for node in adjacency.keys():
        if node in visited:
            continue
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            component_index[curr] = component_id
            for neigh, _ in adjacency.get(curr, []):
                if neigh not in visited:
                    stack.append(neigh)
        component_id += 1
    return component_index


def mixed_language_neighbors(node_records: Dict[str, Dict[str, str]], adjacency: Dict[str, List[Tuple[str, float]]]) -> List[str]:
    mixed: List[str] = []
    for node, neighbors in adjacency.items():
        langs = {node_records.get(node, {}).get('language')}
        langs |= {node_records.get(n, {}).get('language') for n, _ in neighbors}
        langs = {lang for lang in langs if lang}
        if len(langs) > 1:
            mixed.append(node)
    return mixed
