from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..db import get_enhanced_data_manager, ActionRecord
from .mining import _extract_prompt


STOPWORDS = {
    'the','a','an','to','of','and','in','on','for','with','by','or','as','at','is','are','be','this','that','it','from','we','you','our','your','their','they','he','she','i','me','my','mine','yours','ours','theirs','code','fix','bug','test','tests'
}


def _tool_name_from_action(a: ActionRecord) -> Optional[str]:
    cand = None
    for obj in (a.parameters, a.result, a.state_before, a.state_after):
        if isinstance(obj, dict):
            for k in ('tool', 'action', 'tool_name', 'signature_name'):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    cand = v.strip()
                    break
        if cand:
            break
    if not cand:
        # Fallback to action type label
        try:
            cand = a.action_type.value
        except Exception:
            cand = None
    return cand


def _top_keywords(prompts: List[str], k: int = 4) -> List[str]:
    cnt = Counter()
    for p in prompts:
        toks = re.findall(r"[A-Za-z_]+", (p or "").lower())
        cnt.update([t for t in toks if t not in STOPWORDS and len(t) > 2])
    return [w for w, _ in cnt.most_common(k)]


def compute_policy_nudges(
    *,
    hours: int = 24,
    min_count: int = 5,
    top_k: int = 3,
    bottom_k: int = 2,
    min_avg_for_deny: float = 0.05,
) -> Dict[str, List[Dict[str, Optional[str]]]]:
    """Analyze recent actions to propose policy nudges.

    Returns a dict with two lists: 'prefer' and 'deny', each element shaped like
    { 'prefer_tool': str, 'deny_tool': str | None, 'regex': str | None } suitable for update_policy_with_feedback.
    """
    dm = get_enhanced_data_manager()
    cutoff = time.time() - (hours * 3600)
    acts = [a for a in dm.get_recent_actions(5000) if a.timestamp >= cutoff and isinstance(a.reward, (int, float))]

    # Aggregate per tool
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: { 'sum': 0.0, 'cnt': 0, 'prompts': [] })
    for a in acts:
        t = _tool_name_from_action(a)
        if not t:
            continue
        stats[t]['sum'] += float(a.reward)
        stats[t]['cnt'] += 1
        pr = _extract_prompt(a)
        if pr:
            stats[t]['prompts'].append(pr)

    items = []
    for t, s in stats.items():
        if s['cnt'] < min_count:
            continue
        avg = (s['sum'] / float(s['cnt'])) if s['cnt'] else 0.0
        items.append((t, avg, s['cnt'], s['prompts']))

    if not items:
        return { 'prefer': [], 'deny': [] }

    items.sort(key=lambda x: x[1])  # ascending by avg reward
    lows = items[:max(0, bottom_k)]
    highs = list(reversed(items))[:max(0, top_k)]

    prefer: List[Dict[str, Optional[str]]] = []
    deny: List[Dict[str, Optional[str]]] = []

    # Prefer: top_k by avg reward
    for t, avg, cnt, prompts in highs:
        kws = _top_keywords(prompts, k=4)
        rx = None
        if kws:
            rx = r"(?i)\b(" + "|".join(map(re.escape, kws[:min(3, len(kws))])) + r")\b"
        prefer.append({ 'prefer_tool': t, 'deny_tool': None, 'regex': rx })

    # Deny: bottom_k by avg reward (only if very low)
    for t, avg, cnt, prompts in lows:
        if avg <= float(min_avg_for_deny):
            kws = _top_keywords(prompts, k=3)
            rx = None
            if kws:
                rx = r"(?i)\b(" + "|".join(map(re.escape, kws[:min(2, len(kws))])) + r")\b"
            deny.append({ 'prefer_tool': None, 'deny_tool': t, 'regex': rx })

    return { 'prefer': prefer, 'deny': deny }

