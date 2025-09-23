from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, List


AGENT_CFG_FILENAME = ".dspy_agent.json"

_DEFAULTS: Dict[str, Any] = {
    "embeddings": {
        # Keep embeddings disabled by default so everything works offline
        "enabled": False,
        "backend": "auto",  # auto|hf|dspy
        "hf_model": "all-MiniLM-L6-v2",
        "dspy_model": "openai/text-embedding-3-small",
        # Optional per-path overrides: list of {pattern, enabled?, backend?, hf_model?, dspy_model?}
        "per_path": [],
    },
    "routing": {
        "rules": [
            {"when": r"(?i)neighbors\s+of\s+([\\w/\\\\.-]+\\.py)", "tool": "neighbors"},
            {"when": r"(?i)kafka|topics", "tool": "grep", "args": {"pattern": r"topics|topic|logs\\.ctx|agent\\."}},
        ]
    },
}


def load_agent_config(workspace: Path) -> Dict[str, Any]:
    """Load workspace-level agent config with defaults merged in."""
    ws = workspace.resolve()
    path = ws / AGENT_CFG_FILENAME
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    # Shallow-merge defaults
    out = json.loads(json.dumps(_DEFAULTS))  # deep copy
    for k, v in (data or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)  # type: ignore
        else:
            out[k] = v
    return out


def get_embedding_prefs(workspace: Path) -> Dict[str, Any]:
    cfg = load_agent_config(workspace)
    emb = cfg.get("embeddings") or {}
    if not isinstance(emb, dict):
        emb = {}
    enabled = bool(emb.get("enabled", False))
    backend = str(emb.get("backend", "auto")).strip().lower()
    hf_model = str(emb.get("hf_model", _DEFAULTS["embeddings"]["hf_model"]))
    dspy_model = str(emb.get("dspy_model", _DEFAULTS["embeddings"]["dspy_model"]))
    return {
        "enabled": enabled,
        "backend": backend,
        "hf_model": hf_model,
        "dspy_model": dspy_model,
        "per_path": list(emb.get("per_path", []) if isinstance(emb.get("per_path"), list) else []),
    }


def get_routing_rules(workspace: Path) -> List[Dict[str, Any]]:
    cfg = load_agent_config(workspace)
    rt = cfg.get("routing") or {}
    if not isinstance(rt, dict):
        return []
    rules = rt.get("rules") or []
    return list(rules) if isinstance(rules, list) else []


def resolve_embedding_mode_for_task(workspace: Path, task: str) -> Dict[str, Any] | None:
    """Best-effort: if task mentions a path and per_path has a matching rule, return it.

    Returns an embedding config override dict or None.
    """
    prefs = get_embedding_prefs(workspace)
    rules = prefs.get("per_path") or []
    if not isinstance(rules, list) or not task:
        return None
    import re as _re
    # Extract likely file/path tokens from task
    candidates: list[str] = []
    try:
        candidates += _re.findall(r"[\w./\\-]+\.py", task)
        candidates += _re.findall(r"(?:\./|/)?[\w./\\-]+", task)
    except Exception:
        pass
    text = task
    for rule in rules:
        pat = str(rule.get("pattern", "")).strip()
        if not pat:
            continue
        try:
            if _re.search(pat, text) or any(_re.search(pat, c or "") for c in candidates):
                return dict(rule)
        except Exception:
            # Fallback substring matching
            if (pat in text) or any(pat in (c or "") for c in candidates):
                return dict(rule)
    return None
