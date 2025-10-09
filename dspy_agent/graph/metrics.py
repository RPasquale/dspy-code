"""Graph metric collectors for enriching code graphs with real-world signals."""

from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from ..db.redb_router import RedDBRouter
except Exception:  # pragma: no cover - optional during limited imports
    RedDBRouter = None  # type: ignore


SUPPORTED_CODE_EXTS = {
    '.py', '.rs', '.go', '.ts', '.tsx', '.js', '.jsx', '.java', '.sh', '.bash', '.html', '.css'
}


@dataclass
class EdgeMetric:
    git_copresence: float = 0.0
    runtime_hits: float = 0.0
    coverage_overlap: float = 0.0
    last_observed: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _is_code_file(path: str) -> bool:
    return Path(path).suffix in SUPPORTED_CODE_EXTS


def _git(cmd: List[str], cwd: Path) -> str:
    try:
        result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=True)
        return result.stdout
    except Exception:
        return ""


def collect_git_cochange(root: Path, commit_limit: int = 200) -> Dict[Tuple[str, str], float]:
    if not (root / '.git').exists():
        return {}
    output = _git([
        'git', 'log', '--no-merges', f'--max-count={commit_limit}', '--pretty=format:%H', '--name-only'
    ], root)
    if not output:
        return {}
    pairs: Dict[Tuple[str, str], float] = defaultdict(float)
    current_files: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) == 40 and all(c in '0123456789abcdef' for c in line.lower()):
            files = [f for f in current_files if _is_code_file(f)]
            for i in range(len(files)):
                for j in range(len(files)):
                    if i == j:
                        continue
                    pairs[(files[i], files[j])] += 1.0
            current_files = set()
        else:
            current_files.add(line)
    return pairs


def collect_runtime_edges(
    root: Path,
    traces_dir: str = 'runtime_traces',
    *,
    router: 'RedDBRouter | None' = None,
    namespace: Optional[str] = None,
) -> Dict[Tuple[str, str], float]:
    base = root / traces_dir
    edge_counts: Dict[Tuple[str, str], float] = defaultdict(float)

    if router is not None and namespace:
        try:
            key = router._k(namespace, 'runtime', 'edge_counts')
            data = router.st.get(key)
            if isinstance(data, dict):
                for pair, val in data.items():
                    if isinstance(pair, str):
                        try:
                            src, dst = pair.split('->', 1)
                        except ValueError:
                            continue
                    elif isinstance(pair, (list, tuple)) and len(pair) == 2:
                        src, dst = pair
                    else:
                        continue
                    if _is_code_file(src) and _is_code_file(dst):
                        edge_counts[(src, dst)] += float(val or 0.0)
        except Exception:
            pass

    if not base.exists():
        return edge_counts

    for path in base.glob('**/*.json'):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        events = payload if isinstance(payload, list) else payload.get('events') or []
        for event in events:
            if not isinstance(event, dict):
                continue
            src = str(event.get('src') or event.get('from') or '')
            dst = str(event.get('dst') or event.get('to') or '')
            if not src or not dst:
                continue
            if _is_code_file(src) and _is_code_file(dst):
                edge_counts[(src, dst)] += float(event.get('count') or 1.0)
    return edge_counts


def collect_coverage_overlap(
    root: Path,
    report_path: str = 'coverage/coverage-summary.json',
    *,
    router: 'RedDBRouter | None' = None,
    namespace: Optional[str] = None,
) -> Dict[str, float]:
    if router is not None and namespace:
        try:
            key = router._k(namespace, 'coverage', 'summary')
            data = router.st.get(key)
            if isinstance(data, dict):
                overlap = {}
                for file_path, pct in data.items():
                    try:
                        overlap[file_path] = float(pct)
                    except (TypeError, ValueError):
                        continue
                if overlap:
                    return overlap
        except Exception:
            pass

    report = root / report_path
    if not report.exists():
        return {}
    try:
        data = json.loads(report.read_text())
    except Exception:
        return {}
    overlap: Dict[str, float] = {}
    for file_path, stats in data.get('files', {}).items():
        pct = stats.get('lines', {}).get('pct')
        if pct is None:
            continue
        overlap[file_path] = float(pct)
    return overlap


def collect_edge_metrics(
    root: Path,
    commit_limit: int = 200,
    *,
    router: 'RedDBRouter | None' = None,
    namespace: Optional[str] = None,
) -> Dict[Tuple[str, str], EdgeMetric]:
    root = root.resolve()
    git_pairs = collect_git_cochange(root, commit_limit=commit_limit)
    runtime_pairs = collect_runtime_edges(root, router=router, namespace=namespace)
    coverage = collect_coverage_overlap(root, router=router, namespace=namespace)

    metrics: Dict[Tuple[str, str], EdgeMetric] = {}
    now = time.time()
    keys = set(git_pairs.keys()) | set(runtime_pairs.keys())
    for src_dst in keys:
        src, dst = src_dst
        metric = EdgeMetric(
            git_copresence=git_pairs.get(src_dst, 0.0),
            runtime_hits=runtime_pairs.get(src_dst, 0.0),
            coverage_overlap=(coverage.get(src, 0.0) + coverage.get(dst, 0.0)) / 2.0,
            last_observed=now,
        )
        metrics[src_dst] = metric
    return metrics


def merge_edge_metrics(existing: Dict[str, Any], new: EdgeMetric, decay: float = 0.85) -> Dict[str, Any]:
    if not existing:
        existing = {}
    combined = {
        'git_copresence': existing.get('git_copresence', 0.0) * decay + new.git_copresence,
        'runtime_hits': existing.get('runtime_hits', 0.0) * decay + new.runtime_hits,
        'coverage_overlap': max(existing.get('coverage_overlap', 0.0) * decay, new.coverage_overlap),
        'last_observed': max(existing.get('last_observed', 0.0), new.last_observed),
        'updated_at': time.time(),
        'decayed_weight': existing.get('decayed_weight', 0.0) * decay + new.git_copresence + new.runtime_hits,
        'count': existing.get('count', 0) + 1,
    }
    return combined
