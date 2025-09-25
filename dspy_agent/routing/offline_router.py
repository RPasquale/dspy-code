from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import re
from ..agent_config import get_routing_rules, get_embedding_prefs, resolve_embedding_mode_for_task
from ..context import ContextManager
from ..db.data_models import ActionType, create_action_record, Environment
from ..db.enhanced_storage import get_enhanced_data_manager


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
    chosen: Tuple[str, Dict[str, object]] | None = None
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
                        chosen = (tool, dict(args))
                        break
            except re.error:
                continue
    except Exception:
        pass
    # Special-case: Makefile targets
    if chosen is None and ("makefile" in q and ("target" in q or "targets" in q)):
        pat = r"^[A-Za-z0-9_.-]+:\s*$"
        chosen = ("grep", {"pattern": pat})
    # Special-case: neighbors of <file.py>
    if chosen is None:
        m = re.search(r"neighbors\s+of\s+([\w/\\.-]+\.py)", q)
        if m:
            chosen = ("neighbors", {"file": m.group(1)})
    # Where is <symbol>
    if chosen is None:
        m = re.search(r"where\s+is\s+([A-Za-z_][A-Za-z0-9_]*)", q)
        if m:
            sym = m.group(1)
            pat = rf"(def\s+{sym}\b|class\s+{sym}\b)"
            chosen = ("grep", {"pattern": pat})
    # Prefer docker compose file if task mentions docker and file exists
    if chosen is None and (("docker" in q or "compose" in q) and (ws / 'docker' / 'lightweight' / 'docker-compose.yml').exists()):
        chosen = ("grep", {"pattern": "services:"})
    # Tests/lint/build first
    if chosen is None:
        for kw in ("pytest", "tests", "unittest"):
            if kw in q:
                chosen = ("run_tests", {})
                break
    if chosen is None and ("lint" in q or "format" in q):
        chosen = ("lint", {"fix": False})
    if chosen is None and ("build" in q):
        chosen = ("build", {})
    # Retrievals
    if chosen is None and any(w in q for w in ("semantic", "vector", "embedding")):
        # If embeddings disabled by config, prefer TF-IDF
        try:
            override = resolve_embedding_mode_for_task(ws, task) or {}
            emb = get_embedding_prefs(ws)
            enabled = bool(emb.get("enabled", False))
            if "enabled" in override:
                enabled = bool(override.get("enabled"))
            if not enabled:
                chosen = ("esearch", {"q": task})
            else:
                chosen = ("vretr", {"k": 5})
        except Exception:
            chosen = ("esearch", {"q": task})
    if chosen is None and any(w in q for w in ("search", "grep", "find")):
        chosen = ("grep", {"pattern": task})
    if chosen is None and ("index" in q):
        chosen = ("index", {})
    # RL-related queries
    if chosen is None and any(w in q for w in ("rl", "bandit", "ppo", "reward")):
        chosen = ("esearch", {"q": task})
    # Context-aware heuristics before final fallback
    try:
        cm = ContextManager(ws, ws / 'logs')
        feats = cm.agentic_features(max_items=10)
        stats = cm.stats_for_features()
        # If recent failures and logs exist, prefer context
        if chosen is None and has_logs and float(stats.get('recent_failure_rate', 0.0)) > 0.3:
            chosen = ("context", {})
        # If retrieval has signal, bias to esearch/vretr even without explicit keyword
        if chosen is None:
            # feats: [kg_nodes, kg_edges, kg_avg_weight, kg_avg_fanout, precision, coverage, avg_score, query_count]
            precision = float(feats[4]) if len(feats) > 4 else 0.0
            avg_score = float(feats[6]) if len(feats) > 6 else 0.0
            query_count = float(feats[7]) if len(feats) > 7 else 0.0
            generic = any(w in q for w in ("where", "how", "implement", "usage", "open", "init", "config"))
            if generic and (precision > 0.4 or (avg_score > 0.25 and query_count >= 2.0)):
                try:
                    override = resolve_embedding_mode_for_task(ws, task) or {}
                    emb = get_embedding_prefs(ws)
                    enabled = bool(emb.get("enabled", False))
                    if "enabled" in override:
                        enabled = bool(override.get("enabled"))
                    chosen = ("vretr", {"k": 5}) if enabled else ("esearch", {"q": task})
                except Exception:
                    chosen = ("esearch", {"q": task})
        # Fallback: show context/logs or code snapshot depending on availability
        if chosen is None:
            tool, args = ("context" if has_logs else "codectx"), {}
        else:
            tool, args = chosen
    except Exception:
        tool, args = ("context" if has_logs else "codectx"), {}

    # Log decision to RedDB for learning
    try:
        dm = get_enhanced_data_manager()
        before = {
            'workspace': str(ws),
            'task': task,
            'features': feats if 'feats' in locals() else [],
            'stats': stats if 'stats' in locals() else {},
            'has_logs': bool(has_logs),
        }
        after = {'tool': tool, 'args': dict(args)}
        rec = create_action_record(
            action_type=ActionType.TOOL_SELECTION,
            state_before=before,
            state_after=after,
            parameters={'mode': 'offline'},
            result={'selected': tool},
            reward=0.0,
            confidence=0.5,
            execution_time=0.0,
            environment=Environment.DEVELOPMENT,
        )
        dm.record_action(rec)  # type: ignore[attr-defined]
    except Exception:
        pass
    return tool, args
