"""Utilities for enriching graph nodes with embeddings and operational metadata."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from ..embedding.embedder import Embedder
except Exception:  # optional dependency
    Embedder = None  # type: ignore


def compute_embedding(text: str) -> Optional[list[float]]:
    if Embedder is None:
        return None
    try:
        embedder = Embedder()
    except Exception:
        return None
    try:
        vec = embedder.embed_text(text)
        return list(vec)
    except Exception:
        return None


def load_codeowners(root: Path) -> Dict[str, str]:
    owners: Dict[str, str] = {}
    file_path = root / 'CODEOWNERS'
    if not file_path.exists():
        return owners
    for line in file_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pattern, owner = parts[0], parts[1]
        owners[pattern.lstrip('./')] = owner.lstrip('@')
    return owners


def match_owner(rel_path: str, owners: Dict[str, str]) -> Optional[str]:
    for pattern, owner in owners.items():
        if rel_path.endswith(pattern) or rel_path.startswith(pattern.rstrip('*')):
            return owner
    return None


def load_issue_metadata(root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    path = root / 'graph_issue_metadata.json'
    if not path.exists():
        return info
    try:
        info = json.loads(path.read_text())
    except Exception:
        info = {}
    return info


def build_node_metadata(
    root: Path,
    rel_path: str,
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    existing = existing or {}
    owners = existing.get('_codeowners_cache') or load_codeowners(root)
    existing['_codeowners_cache'] = owners
    owner = match_owner(rel_path, owners)

    issues = existing.get('_issues_cache')
    if issues is None:
        issues = load_issue_metadata(root)
        existing['_issues_cache'] = issues
    issue_data = issues.get(rel_path, {}) if isinstance(issues, dict) else {}

    metadata = {
        'owner': owner,
        'issues_open': issue_data.get('open', 0),
        'issues_last_updated': issue_data.get('updated_at'),
        'test_failures': issue_data.get('test_failures', 0),
        'slo': issue_data.get('slo'),
        'metadata_refreshed_at': time.time(),
    }
    return metadata
