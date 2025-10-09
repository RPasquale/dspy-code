"""Utility helpers for writing runtime and coverage data into RedDB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

from ..db.redb_router import RedDBRouter


def store_runtime_edge_counts(
    namespace: str,
    edge_counts: Mapping[str, float],
    *,
    router: RedDBRouter | None = None,
) -> None:
    router = router or RedDBRouter()
    key = router._k(namespace, 'runtime', 'edge_counts')
    payload = {str(k): float(v) for k, v in edge_counts.items()}
    router.st.put(key, payload)


def store_runtime_edge_counts_from_file(
    namespace: str,
    path: Path,
    *,
    router: RedDBRouter | None = None,
) -> None:
    content = json.loads(Path(path).read_text())
    edge_counts: Dict[str, float] = {}
    if isinstance(content, dict):
        for pair, val in content.items():
            edge_counts[str(pair)] = float(val)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                src = str(item.get('src') or item.get('from') or '')
                dst = str(item.get('dst') or item.get('to') or '')
                count = float(item.get('count') or 1)
                if src and dst:
                    edge_counts[f'{src}->{dst}'] = edge_counts.get(f'{src}->{dst}', 0.0) + count
    store_runtime_edge_counts(namespace, edge_counts, router=router)


def store_coverage_summary(
    namespace: str,
    coverage: Mapping[str, float],
    *,
    router: RedDBRouter | None = None,
) -> None:
    router = router or RedDBRouter()
    key = router._k(namespace, 'coverage', 'summary')
    router.st.put(key, {str(k): float(v) for k, v in coverage.items()})


def store_coverage_summary_from_file(
    namespace: str,
    path: Path,
    *,
    router: RedDBRouter | None = None,
) -> None:
    data = json.loads(Path(path).read_text())
    coverage: Dict[str, float] = {}
    files = data.get('files') if isinstance(data, dict) else None
    if isinstance(files, dict):
        for file_path, stats in files.items():
            if isinstance(stats, dict) and stats.get('lines') is not None:
                pct = stats['lines'].get('pct') if isinstance(stats['lines'], dict) else stats.get('lines')
                try:
                    coverage[file_path] = float(pct)
                except (TypeError, ValueError):
                    continue
    store_coverage_summary(namespace, coverage, router=router)


__all__ = [
    'store_runtime_edge_counts',
    'store_runtime_edge_counts_from_file',
    'store_coverage_summary',
    'store_coverage_summary_from_file',
]
