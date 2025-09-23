from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import re
from ..agent_config import get_routing_rules, get_embedding_prefs, resolve_embedding_mode_for_task


KEYWORDS = {
    "tests": ("run_tests", {}),
    "pytest": ("run_tests", {}),
    "unittest": ("run_tests", {}),
    "lint": ("lint", {}),
    "format": ("lint", {"fix": True}),
    "build": ("build", {}),
    "search": ("grep", {}),
    "grep": ("grep", {}),
    "find": ("grep", {}),
    "semantic": ("esearch", {}),
    "vector": ("vretr", {}),
    "embedding": ("emb-index", {}),
    "index": ("index", {}),
    "snapshot": ("codectx", {}),
    "context": ("context", {}),
    "neighbors": ("neighbors", {}),
    "imports": ("neighbors", {}),
    "calls": ("neighbors", {}),
    "compose": ("grep", {"pattern": "docker-compose.yml"}),
    "docker": ("grep", {"pattern": "docker-compose.yml"}),
    "rl": ("esearch", {"q": "rl policy bandit reward"}),
    "policy": ("esearch", {"q": "policy preference update"}),
    # Repo-specific helpers
    "infermesh": ("grep", {"pattern": r"infermesh|embed-worker|vectorizer|INFERMESH|4041|9101"}),
    "metrics": ("grep", {"pattern": r"metrics|/metrics|/health|health-check"}),
    "kafka": ("grep", {"pattern": r"topic|logs\\.ctx|agent\\.|deploy\\.events|bootstrap\.servers"}),
    "topics": ("grep", {"pattern": r"topics|topic|logs\\.ctx"}),
    "vectorized": ("grep", {"pattern": r"vectorized|vectorizer|embeddings"}),
    "pipeline": ("grep", {"pattern": r"pipeline|stack|compose"}),
}


def offline_route(task: str, ws: Path, has_logs: bool = False) -> Tuple[str, Dict[str, object]]:
    """Lightweight rule-based router for LM=none cases.

    Returns (tool, args) chosen by keyword heuristics.
    """
    q = (task or "").lower()
    # 1) Try workspace rules first
    try:
        rules = get_routing_rules(ws)
        for rule in rules:
            pattern = str(rule.get("when", ""))
            if not pattern:
                continue
            try:
                if re.search(pattern, q):
                    tool = str(rule.get("tool", "")).strip()
                    args = rule.get("args") or {}
                    if tool:
                        return tool, dict(args)
            except re.error:
                continue
    except Exception:
        pass
    # Special-case: Makefile targets
    if "makefile" in q and ("target" in q or "targets" in q):
        pat = r"^[A-Za-z0-9_.-]+:\s*$"
        return "grep", {"pattern": pat}
    # Special-case: neighbors of <file.py>
    m = re.search(r"neighbors\s+of\s+([\w/\\.-]+\.py)", q)
    if m:
        return "neighbors", {"file": m.group(1)}
    # Where is <symbol>
    m = re.search(r"where\s+is\s+([A-Za-z_][A-Za-z0-9_]*)", q)
    if m:
        sym = m.group(1)
        pat = rf"(def\s+{sym}\b|class\s+{sym}\b)"
        return "grep", {"pattern": pat}
    # Prefer docker compose file if task mentions docker and file exists
    if ("docker" in q or "compose" in q) and (ws / 'docker' / 'lightweight' / 'docker-compose.yml').exists():
        return "grep", {"pattern": "services:"}
    # Tests/lint/build first
    for kw in ("pytest", "tests", "unittest"):
        if kw in q:
            return "run_tests", {}
    if "lint" in q or "format" in q:
        return "lint", {"fix": False}
    if "build" in q:
        return "build", {}
    # Retrievals
    if any(w in q for w in ("semantic", "vector", "embedding")):
        # If embeddings disabled by config, prefer TF-IDF
        try:
            override = resolve_embedding_mode_for_task(ws, task) or {}
            emb = get_embedding_prefs(ws)
            enabled = bool(emb.get("enabled", False))
            if "enabled" in override:
                enabled = bool(override.get("enabled"))
            if not enabled:
                return "esearch", {"q": task}
        except Exception:
            pass
        return "vretr", {"k": 5}
    if any(w in q for w in ("search", "grep", "find")):
        return "grep", {"pattern": task}
    if "index" in q:
        return "index", {}
    # RL-related queries
    if any(w in q for w in ("rl", "bandit", "ppo", "reward")):
        return "esearch", {"q": task}
    # Fallback: show context/logs or code snapshot depending on availability
    return ("context" if has_logs else "codectx"), {}
