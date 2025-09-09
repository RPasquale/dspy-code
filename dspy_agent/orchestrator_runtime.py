from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .log_reader import load_logs, extract_key_events
from .code_search import search_text, search_file as file_search, extract_context, python_extract_symbol
from .indexer import build_index, save_index, load_index, semantic_search
from .code_snapshot import build_code_snapshot


SAFE_TOOLS = {"context", "plan", "grep", "extract", "codectx", "index", "esearch"}


@dataclass
class EvalOutcome:
    score: float
    feedback: str
    evidence: str


def _json_arg(args: Any) -> Dict[str, Any]:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {}
    return {}


def evaluate_tool_choice(
    tool: str,
    args: Any,
    *,
    workspace: Path,
    logs_path: Optional[Path] = None,
    targets: Optional[List[str]] = None,
) -> EvalOutcome:
    """Execute a safe, non-destructive evaluation for a tool choice.

    Scoring heuristics:
    - grep: +1.0 if any hit; +extra if lines contain targets.
    - extract: +1.0 if a segment produced; +extra if contains targets.
    - context: +1.0 if key events non-empty; +extra if targets present.
    - codectx: +1.0 if snapshot summary non-empty; +extra if targets in snapshot.
    - index: builds index; +0.5 baseline.
    - esearch: +1.0 if any hit; +extra if hits contain targets.
    - plan: +1.0 if plan text contains actionable verbs / multiple steps.
    """
    t = (tool or "").lower().strip()
    argd = _json_arg(args)
    evidence = []
    score = 0.0
    fb = []
    targets = targets or []

    def cov(text: str, kws: List[str]) -> Tuple[float, List[str]]:
        missing = []
        textl = (text or "").lower()
        hit = 0
        for k in kws:
            if not k:
                continue
            if str(k).lower() in textl:
                hit += 1
            else:
                missing.append(k)
        denom = max(1, len([k for k in kws if k]))
        return hit / denom, missing

    if t not in SAFE_TOOLS:
        return EvalOutcome(score=0.0, feedback=f"Tool '{t}' is not evaluated in training; focus on safe tools.", evidence="")

    if t == "grep":
        pattern = argd.get("pattern") or argd.get("query") or ""
        globs = argd.get("globs") or ["**/*"]
        hits = search_text(workspace, pattern, regex=True, include_globs=globs)
        n = len(hits)
        evidence.append(f"grep hits={n}")
        if n > 0:
            score += 1.0
            text = "\n".join(h.line for h in hits[:50])
            c, miss = cov(text, targets)
            score += 0.5 * c
            if miss:
                fb.append("grep: Consider alternative keywords: " + ", ".join(miss))
        else:
            fb.append("grep returned 0 hits. Consider esearch or adjusting pattern.")

    elif t == "extract":
        file = argd.get("file")
        symbol = argd.get("symbol")
        regex = argd.get("regex")
        if not file:
            return EvalOutcome(0.0, "extract missing file arg", "")
        fpath = (workspace / file)
        seg = ""
        if symbol and fpath.suffix == ".py":
            res = python_extract_symbol(fpath, symbol)
            if res:
                _, _, seg = res
        elif regex:
            hits = file_search(fpath, regex, regex=True)
            if hits:
                text = fpath.read_text(errors="ignore")
                s, e, seg = extract_context(text, hits[0].line_no, before=3, after=3)
        if seg:
            score += 1.0
            evidence.append(f"extract segment_len={len(seg)}")
            c, miss = cov(seg, targets)
            score += 0.5 * c
            if miss:
                fb.append("extract: Segment misses: " + ", ".join(miss))
        else:
            fb.append("extract produced no segment. Provide --symbol (py) or --regex.")

    elif t == "context":
        log_target = logs_path or (workspace / "logs")
        bundle, count = load_logs([log_target])
        key = extract_key_events(bundle) if bundle else ""
        evidence.append(f"context logs_found={bool(bundle)} events_len={len(key)}")
        if key:
            score += 1.0
            c, miss = cov(key, targets)
            score += 0.5 * c
            if miss:
                fb.append("context: key events miss: " + ", ".join(miss))
        else:
            fb.append("context produced no events. Consider codectx if logs are missing.")

    elif t == "codectx":
        snap = build_code_snapshot(workspace)
        evidence.append(f"codectx snapshot_len={len(snap)}")
        if snap:
            score += 1.0
            c, miss = cov(snap, targets)
            score += 0.5 * c
            if miss:
                fb.append("codectx: summary miss: " + ", ".join(miss))
        else:
            fb.append("codectx empty snapshot.")

    elif t == "index":
        meta, items = build_index(workspace, smart=True)
        save_index(workspace, meta, items)
        score += 0.5
        evidence.append(f"index chunks={len(items)}")

    elif t == "esearch":
        q = argd.get("query") or argd.get("q") or ""
        try:
            meta, items = load_index(workspace)
        except FileNotFoundError:
            meta, items = build_index(workspace, smart=True); save_index(workspace, meta, items)
        hits = semantic_search(q, meta, items, top_k=5)
        evidence.append(f"esearch hits={len(hits)}")
        if hits:
            score += 1.0
            # Check coverage from snippets
            segs = []
            for _, it in hits:
                p = Path(it.path)
                try:
                    text = p.read_text(errors="ignore")
                    lines = text.splitlines()
                    s = max(1, it.start_line - 2)
                    e = min(len(lines), it.end_line + 2)
                    segs.append("\n".join(lines[s - 1 : e]))
                except Exception:
                    continue
            joined = "\n".join(segs)
            c, miss = cov(joined, targets)
            score += 0.5 * c
            if miss:
                fb.append("esearch: results miss: " + ", ".join(miss))
        else:
            fb.append("esearch found no results. Consider grep or building the index first.")

    elif t == "plan":
        # Heuristic: plan text should contain multiple steps-like lines
        plan_text = (argd.get("plan_text") or "")  # During training, we don't call LLM; rely on downstream plan execution elsewhere
        steps = [ln for ln in plan_text.splitlines() if ln.strip().startswith(('-','1','2','3'))]
        score += 1.0 if len(steps) >= 2 else 0.2
        evidence.append(f"plan steps_detected={len(steps)}")

    feedback = " \n".join(fb) if fb else "Looks good."
    return EvalOutcome(score=float(score), feedback=feedback, evidence="; ".join(evidence))

