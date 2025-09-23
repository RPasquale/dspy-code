from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..db import get_enhanced_data_manager, ActionRecord, ActionType
try:
    from ..mesh.core import MeshCoreClient
except Exception:
    MeshCoreClient = None  # type: ignore


def _norm_text(s: Any) -> str:
    t = (str(s or "").replace('\r', ' ').replace('\n', ' ').strip())
    # Squeeze spaces
    return ' '.join(t.split())


def _extract_prompt(a: ActionRecord) -> Optional[str]:
    cand = None
    P = a.parameters or {}
    SB = a.state_before or {}
    ST = a.state_after or {}
    R = a.result or {}
    # Common keys
    for obj in (P, R, SB, ST):
        for key in ("query", "prompt", "instruction", "user", "input", "task"):
            if isinstance(obj, dict) and obj.get(key):
                cand = obj.get(key)
                break
        if cand:
            break
    if not cand and isinstance(SB, dict) and SB.get('current_task'):
        cand = SB.get('current_task')
    if not cand:
        return None
    txt = _norm_text(cand)
    return txt if len(txt) >= 8 else None


def _extract_response(a: ActionRecord) -> str:
    R = a.result or {}
    P = a.parameters or {}
    keys = ("output", "message", "text", "stdout", "patch_content", "diff", "result")
    for obj in (R, P):
        if isinstance(obj, dict):
            for k in keys:
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v
    # Fallback to compact JSON of result
    try:
        return json.dumps({k: v for k, v in (R if isinstance(R, dict) else {}).items() if isinstance(v, (str, int, float))}, ensure_ascii=False)[:2000]
    except Exception:
        return ""


def mine_reddb_to_grpo(
    out_path: Path | str,
    *,
    hours: int = 24,
    limit_actions: int = 5000,
    include_types: Optional[List[str]] = None,
    min_k: int = 2,
    max_k: int = 6,
    mesh_topics: Optional[List[str]] = None,
    mesh_limit: int = 200,
) -> Path:
    """Mine recent actions from RedDB and produce a grouped preference JSONL for GRPO.

    - Groups by inferred user prompt
    - Each group contains candidates with reward and response text
    - Heuristically selects extremes when too many candidates exist per group
    - Enriches metadata with environment, signature_name, context_hash, and retrieval hits when available
    """
    dm = get_enhanced_data_manager()
    cutoff = time.time() - (hours * 3600)
    actions = dm.get_recent_actions(limit_actions)
    # Filter by time and, if requested, action types
    allow: Optional[set] = None
    if include_types:
        allow = set()
        for name in include_types:
            try:
                allow.add(ActionType(name))
            except Exception:
                # Try case-insensitive match
                for at in ActionType:
                    if at.value.lower() == str(name).lower():
                        allow.add(at)
                        break
    f_actions: List[ActionRecord] = []
    for a in actions:
        if a.timestamp < cutoff:
            continue
        if allow and a.action_type not in allow:
            continue
        if not isinstance(a.reward, (int, float)):
            continue
        f_actions.append(a)

    # Group by prompt
    groups: Dict[str, List[ActionRecord]] = defaultdict(list)
    for a in f_actions:
        p = _extract_prompt(a)
        if not p:
            continue
        groups[p].append(a)

    # Retrieve recent retrieval events (InferMesh integration) to enrich groups
    events = dm.get_recent_retrieval_events(limit=500)
    hits_by_query: Dict[str, Any] = {}
    for ev in events:
        q = _norm_text(ev.query)
        if not q:
            continue
        # Keep last seen hits for this query
        hits_by_query[q] = ev.hits

    # Optionally augment with Mesh Core tail topics
    mesh_added: Dict[str, int] = defaultdict(int)
    if mesh_topics and MeshCoreClient:
        try:
            cli = MeshCoreClient()
            for topic in mesh_topics:
                try:
                    tail = cli.tail(topic, limit=int(mesh_limit)) or {}
                    items = tail.get('items') if isinstance(tail, dict) else None
                    if not isinstance(items, list):
                        continue
                    for it in items:
                        # Heuristic extraction from generic message
                        pr = None
                        txt = None
                        rw = None
                        if isinstance(it, dict):
                            for k in ('prompt','query','input','task','user'):
                                v = it.get(k)
                                if isinstance(v, str) and len(v.strip()) >= 4:
                                    pr = _norm_text(v); break
                            for k in ('response','output','text','message','stdout'):
                                v = it.get(k)
                                if isinstance(v, str) and v.strip():
                                    txt = v; break
                            for k in ('reward','score','r'):
                                v = it.get(k)
                                try:
                                    rw = float(v)
                                except Exception:
                                    pass
                        if not pr or not txt or rw is None:
                            continue
                        # Attach to synthetic group
                        groups.setdefault(pr, [])  # type: ignore[arg-type]
                        # Append synthetic ActionRecord-like proxy (only used downstream for response extraction and metadata)
                        class _Proxy:
                            def __init__(self):
                                self.parameters = {'topic': topic}
                                self.result = {'output': txt}
                                self.state_before = {}
                                self.state_after = {}
                                self.reward = rw
                                self.action_id = None
                                self.action_type = ActionType('generic') if 'generic' in ActionType.__members__ else ActionType.other if hasattr(ActionType, 'other') else ActionType.plan
                                self.context_hash = None
                                self.environment = type('E', (), {'value': 'mesh'})
                                self.timestamp = time.time()
                        groups[pr].append(_Proxy())  # type: ignore[index]
                        mesh_added[pr] += 1
                except Exception:
                    continue
        except Exception:
            pass

    # Emit JSONL
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open('w', encoding='utf-8') as f:
        for prompt, items in groups.items():
            # Build candidates with reward
            cands = [
                {"text": _extract_response(a), "reward": float(a.reward), "_meta": {
                    "action_id": a.action_id,
                    "action_type": a.action_type.value,
                    "signature_name": (a.parameters or {}).get('signature_name') or (a.result or {}).get('signature_name'),
                    "context_hash": a.context_hash,
                    "environment": a.environment.value,
                    "timestamp": a.timestamp,
                }}
                for a in items
                if _extract_response(a)
            ]
            if len(cands) < min_k:
                continue

            # Select up to max_k with extremes and a few mids
            cands.sort(key=lambda x: x['reward'])
            if len(cands) > max_k:
                k = max_k
                take = max(2, min(4, k//2))
                lows = cands[:take//2]
                highs = cands[-(take - len(lows)):]
                mids = []
                remaining = k - len(lows) - len(highs)
                if remaining > 0:
                    step = max(1, (len(cands) - len(lows) - len(highs)) // remaining)
                    mid_slice = cands[len(lows):-len(highs)] if len(highs) > 0 else cands[len(lows):]
                    mids = mid_slice[::step][:remaining]
                cands = lows + mids + highs

            meta = {
                "group_size": len(items),
                "imesh_hits": hits_by_query.get(prompt) or [],
                "mesh": {
                    "context_available": any((a.context_hash for a in items)),
                    "nodes": {
                        "reddb": True,
                        "infermesh": bool(hits_by_query.get(prompt)),
                    },
                    "tail_candidates": int(mesh_added.get(prompt, 0))
                }
            }
            # Remove _meta from emitted candidates but keep as part of group meta snapshot
            group_cands = [{"text": c["text"], "reward": c["reward"]} for c in cands]
            record = {
                "prompt": prompt,
                "candidates": group_cands,
                "meta": meta,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    if written == 0:
        # If nothing written, create a minimal empty file so downstream sees valid JSONL
        out.write_text("")
    return out


def mine_reddb_hierarchical(
    out_dir: Path | str,
    *,
    hours: int = 24,
    limit_actions: int = 8000,
    min_k: int = 2,
    max_k: int = 6,
) -> Dict[str, Path]:
    """Produce multiple GRPO datasets for hierarchical training.

    Levels produced:
      - global: across all actions grouped by inferred prompt
      - signature: restricted to actions with a signature_name; grouped by (signature|prompt)
      - patch: actions related to patches/edits grouped by prompt
    """
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # Global dataset
    paths['global'] = mine_reddb_to_grpo(base / 'global.jsonl', hours=hours, limit_actions=limit_actions, include_types=None, min_k=min_k, max_k=max_k)

    # Signature-specific dataset: group by (signature|prompt)
    dm = get_enhanced_data_manager()
    cutoff = time.time() - (hours * 3600)
    acts = [a for a in dm.get_recent_actions(limit_actions) if a.timestamp >= cutoff]
    buckets: Dict[str, List[ActionRecord]] = defaultdict(list)
    for a in acts:
        sig = None
        for cont in (a.parameters, a.result, a.state_before, a.state_after):
            if isinstance(cont, dict) and isinstance(cont.get('signature_name'), str):
                sig = cont.get('signature_name'); break
        pr = _extract_prompt(a)
        if sig and pr:
            buckets[f"{sig}|{pr}"] .append(a)
    sig_out = base / 'signature.jsonl'
    with sig_out.open('w', encoding='utf-8') as f:
        for key, items in buckets.items():
            if len(items) < min_k:
                continue
            prompt = key.split('|', 1)[1] if '|' in key else key
            cands = [{"text": _extract_response(a), "reward": float(a.reward)} for a in items if _extract_response(a)]
            if len(cands) < min_k:
                continue
            cands.sort(key=lambda x: x['reward'])
            if len(cands) > max_k:
                cands = cands[:max_k//2] + cands[-(max_k - max_k//2):]
            f.write(json.dumps({"prompt": prompt, "candidates": cands, "meta": {"level": "signature"}}) + "\n")
    paths['signature'] = sig_out

    # Patch/edit dataset
    patch_types = ['patch_generation', 'code_edit']
    paths['patch'] = mine_reddb_to_grpo(base / 'patch.jsonl', hours=hours, limit_actions=limit_actions, include_types=patch_types, min_k=min_k, max_k=max_k)

    return paths
