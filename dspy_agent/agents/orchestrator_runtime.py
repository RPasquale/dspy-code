from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping

from ..streaming.log_reader import load_logs, extract_key_events
from ..code_tools.code_search import search_text, search_file as file_search, extract_context, python_extract_symbol
from ..embedding.indexer import build_index, save_index, load_index, semantic_search
from ..code_tools.code_snapshot import build_code_snapshot


class OrchestratorRuntime:
    """Minimal orchestrator runtime for tests and local workflows.

    Tracks simple tasks in-memory and provides create/execute/status methods.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(self, task: Dict[str, Any]) -> None:
        tid = str(task.get("id") or task.get("task_id") or "task")
        if not tid:
            raise ValueError("task id is required")
        self._tasks[tid] = {"status": "pending", "data": dict(task)}

    def execute_task(self, task_id: str) -> Dict[str, Any] | None:
        rec = self._tasks.get(task_id)
        if rec is None:
            return None
        rec["status"] = "running"
        # Trivial placeholder "execution" logic: snapshot or log extraction by type
        ttype = str(rec["data"].get("type") or "").lower()
        try:
            if ttype in {"code_analysis", "codectx"}:
                snap = build_code_snapshot(self.workspace)
                result = {"task": task_id, "summary_len": len(snap)}
            else:
                # Default to evaluating a safe tool choice if provided
                tool = rec["data"].get("tool") or "context"
                ev = evaluate_tool_choice(str(tool), rec["data"], workspace=self.workspace)
                result = {"task": task_id, "score": ev.score, "feedback": ev.feedback}
        except Exception:
            result = {"task": task_id, "error": True}
        rec["status"] = "completed"
        rec["result"] = result
        return result

    def get_task_status(self, task_id: str) -> str:
        rec = self._tasks.get(task_id)
        if not rec:
            return "failed"
        return str(rec.get("status") or "pending")


SAFE_TOOLS = {
    "context", "plan", "grep", "extract", "codectx", "index", "esearch",
    "knowledge", "vretr", "intel", "edit", "patch", "run_tests", "lint", "build"
}


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
    result_metrics: Optional[Mapping[str, Any]] = None,
    result_info: Optional[Mapping[str, Any]] = None,
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
    metrics = dict(result_metrics or {})
    info = dict(result_info or {})
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

    elif t == "vretr":
        # Vector retrieval over local embedding index
        q = argd.get('query') or argd.get('q') or ''
        k = int(argd.get('k', 5))
        try:
            from .embeddings_index import load_emb_index, emb_search as _emb_search, embed_query as _embed_query
            import os as _os
            items = load_emb_index(workspace)
            # Prefer InferMesh when configured
            try:
                if _os.getenv('INFERMESH_URL'):
                    from ..embedding.infermesh import InferMeshEmbedder as _IME  # type: ignore
                    _base = (_os.getenv('INFERMESH_URL') or 'http://infermesh:9000').strip()
                    _model = (argd.get('model') or _os.getenv('EMBED_MODEL') or 'sentence-transformers/all-MiniLM-L6-v2')
                    _embedder = _IME(_base, _model, api_key=_os.getenv('INFERMESH_API_KEY'))
                else:
                    import dspy as _dspy
                    _model = argd.get('model') or 'openai/text-embedding-3-small'
                    _embedder = _dspy.Embeddings(model=_model)
            except Exception:
                import dspy as _dspy
                _model = argd.get('model') or 'openai/text-embedding-3-small'
                _embedder = _dspy.Embeddings(model=_model)
            qv = _embed_query(_embedder, q)
            hits = _emb_search(qv, items, top_k=k)
        except Exception:
            hits = []
        evidence.append(f"vretr hits={len(hits)}")
        if hits:
            score += 1.0
            # Load snippets and check coverage of targets
            segs = []
            for score_i, it in hits:
                p = Path(it.path)
                try:
                    text = p.read_text(errors='ignore')
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
                fb.append("vretr: results miss: " + ", ".join(miss))
        else:
            fb.append("vretr found no results. Consider building embeddings index first.")

    elif t == "edit":
        patch_text = argd.get("patch") or argd.get("diff") or ""
        if not patch_text and metrics.get("patch"):
            patch_text = str(metrics.get("patch", ""))
        patch_text = patch_text or ""
        lines = [ln for ln in patch_text.splitlines() if ln.strip()]
        evidence.append(f"edit patch_lines={len(lines)}")
        if patch_text.strip():
            score += 0.8
            if len(lines) <= 200:
                score += 0.2
        else:
            fb.append("edit produced no patch content.")

    elif t == "patch":
        applied = bool(metrics.get("applied"))
        raw_pr = metrics.get("pass_rate", metrics.get("tests_pass_rate", 0.0))
        try:
            pass_rate = float(raw_pr)
        except Exception:
            pass_rate = 0.0
        raw_blast = metrics.get("blast_radius", 0.0)
        try:
            blast_radius = float(raw_blast)
        except Exception:
            blast_radius = 0.0
        evidence.append(f"patch applied={applied} pass_rate={pass_rate:.2f} blast_radius={blast_radius:.1f}")
        if applied:
            score += 1.0 + 0.5 * max(0.0, min(1.0, pass_rate))
            if pass_rate >= 1.0:
                score += 0.3
            if blast_radius > 400:
                score -= 0.3
        else:
            fb.append("patch was not applied or failed verification.")
        if metrics.get("reverted"):
            fb.append("patch reverted after tests/lint failures.")
        if info.get("error"):
            fb.append(f"patch error: {info['error']}")

    elif t == "run_tests":
        total = metrics.get("tests_total")
        passed = metrics.get("tests_passed")
        failed = metrics.get("tests_failed")
        try:
            pass_rate = float(metrics.get("pass_rate", 0.0))
        except Exception:
            pass_rate = 0.0
        evidence.append(f"run_tests total={total} passed={passed} failed={failed} pr={pass_rate:.2f}")
        if total is None:
            score += 0.3
            fb.append("run_tests metrics unavailable; ensure toolchain detected.")
        else:
            score += 0.4
            if total and total > 0:
                score += 0.6 * max(0.0, min(1.0, pass_rate))
                if failed:
                    fb.append(f"run_tests: {failed} failure(s).")
                elif passed == total:
                    score += 0.2

    elif t == "lint":
        try:
            issues = int(metrics.get("lint_issues", 0) or 0)
        except Exception:
            issues = 0
        lint_ok = metrics.get("lint_ok")
        lint_ok = bool(lint_ok) if lint_ok is not None else (issues == 0)
        evidence.append(f"lint ok={lint_ok} issues={issues}")
        score += 0.4
        if lint_ok:
            score += 0.6
        else:
            fb.append(f"lint reported {issues} issue(s).")

    elif t == "build":
        build_ok = bool(metrics.get("build_ok"))
        evidence.append(f"build ok={build_ok}")
        score += 0.4
        if build_ok:
            score += 0.6
        else:
            fb.append("build command failed.")

    elif t == "intel":
        # Compose knowledge + vretr given a natural-language query
        q = argd.get('query') or argd.get('q') or ''
        k = int(argd.get('k', 5))
        # Gather knowledge matches
        kn_files = []
        try:
            from ..db.factory import get_storage as _get_storage
            st = _get_storage()
            if st is not None:
                graph = st.get('code:graph') if hasattr(st, 'get') else None  # type: ignore
                if isinstance(graph, dict):
                    files = graph.get('files', []) or []
                    # Heuristic: if query includes CamelCase -> class, snake_case -> function, else treat tokens as possible paths
                    import re as _re
                    classes = _re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', q)
                    funcs = _re.findall(r'\b[a-z_][a-z0-9_]+\b', q)
                    for rec in files:
                        pth = rec.get('path','')
                        hit = False
                        if any(c in (rec.get('classes') or []) for c in classes):
                            hit = True
                        if any(fn in (rec.get('functions') or []) for fn in funcs):
                            hit = True
                        if any(tok in pth for tok in funcs[:3]):
                            hit = True
                        if hit:
                            kn_files.append(rec)
        except Exception:
            pass
        # Gather vector retrieval hits
        vretr_hits = []
        try:
            from .embeddings_index import load_emb_index, emb_search as _emb_search, embed_query as _embed_query
            import os as _os
            items = load_emb_index(workspace)
            try:
                if _os.getenv('INFERMESH_URL'):
                    from ..embedding.infermesh import InferMeshEmbedder as _IME  # type: ignore
                    _base = (_os.getenv('INFERMESH_URL') or 'http://infermesh:9000').strip()
                    _model = (argd.get('model') or _os.getenv('EMBED_MODEL') or 'sentence-transformers/all-MiniLM-L6-v2')
                    _embedder = _IME(_base, _model, api_key=_os.getenv('INFERMESH_API_KEY'))
                else:
                    import dspy as _dspy
                    _model = argd.get('model') or 'openai/text-embedding-3-small'
                    _embedder = _dspy.Embeddings(model=_model)
            except Exception:
                import dspy as _dspy
                _model = argd.get('model') or 'openai/text-embedding-3-small'
                _embedder = _dspy.Embeddings(model=_model)
            qv = _embed_query(_embedder, q)
            vretr_hits = _emb_search(qv, items, top_k=k)
        except Exception:
            vretr_hits = []
        # Score: +1 if any knowledge match, +1 if any vretr hit, coverage over joined text
        if kn_files:
            score += 1.0
        if vretr_hits:
            score += 1.0
        text = "\n".join(
            [rec.get('path','')+" " + " ".join((rec.get('classes') or []) + (rec.get('functions') or [])) for rec in kn_files[:50]]
        )
        segs = []
        for _, it in vretr_hits:
            try:
                p = Path(it.path); txt=p.read_text(errors='ignore'); lines=txt.splitlines(); s=max(1,it.start_line-2); e=min(len(lines),it.end_line+2); segs.append("\n".join(lines[s-1:e]))
            except Exception:
                continue
        text += ("\n"+"\n".join(segs))
        c, miss = cov(text, targets)
        score += 0.5 * c
        evidence.append(f"intel kn={len(kn_files)} vretr={len(vretr_hits)} cov={c:.2f}")
        if miss:
            fb.append("intel: missing targets: " + ", ".join(miss))

    elif t == "plan":
        # Heuristic: plan text should contain multiple steps-like lines
        plan_text = (argd.get("plan_text") or "")  # During training, we don't call LLM; rely on downstream plan execution elsewhere
        steps = [ln for ln in plan_text.splitlines() if ln.strip().startswith(('-','1','2','3'))]
        score += 1.0 if len(steps) >= 2 else 0.2
        evidence.append(f"plan steps_detected={len(steps)}")

    elif t == "knowledge":
        # Query code knowledge graph stored in KV
        try:
            from ..db.factory import get_storage as _get_storage
            st = _get_storage()
        except Exception:
            st = None
        found = []
        if st is not None:
            graph = st.get('code:graph') if hasattr(st, 'get') else None  # type: ignore
            if isinstance(graph, dict):
                files = graph.get('files', []) or []
                f = argd.get('file'); imp = argd.get('import'); cls = argd.get('class'); fn = argd.get('function')
                for rec in files:
                    if f and f not in rec.get('path',''):
                        continue
                    if imp and imp not in (rec.get('imports') or []):
                        continue
                    if cls and cls not in (rec.get('classes') or []):
                        continue
                    if fn and fn not in (rec.get('functions') or []):
                        continue
                    found.append(rec)
        n = len(found)
        evidence.append(f"knowledge matches={n}")
        if n>0:
            score += 1.0
            # coverage by targets over concatenated names
            text = "\n".join([rec.get('path','')+" "+" ".join((rec.get('classes') or [])+(rec.get('functions') or [])) for rec in found[:50]])
            c, miss = cov(text, targets)
            score += 0.5 * c
            if miss:
                fb.append("knowledge: consider refining query / add imports or symbols: " + ", ".join(miss))
        else:
            fb.append("knowledge found no matches. Provide file/import/class/function.")

    feedback = " \n".join(fb) if fb else "Looks good."
    return EvalOutcome(score=float(score), feedback=feedback, evidence="; ".join(evidence))
