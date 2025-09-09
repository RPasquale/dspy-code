from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from datetime import datetime

import dspy


def _read_jsonl(path: Path) -> List[dict]:
    data: List[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_examples_for_module(module: str, jsonl_path: Path) -> List[dspy.Example]:
    """Load dataset examples.

    Schemas (per-line JSON):
    - module=context:
        {"task": str, "logs_preview": str, "context_keywords": [str], "key_points_keywords": [str]}

    - module=task:
        {"task": str, "context": str, "plan_keywords": [str], "commands_keywords": [str]}

    - module=code:
        {"snapshot": str, "ask": str, "keywords": [str]}
    """
    raw = _read_jsonl(jsonl_path)
    out: List[dspy.Example] = []
    for r in raw:
        if module == "context":
            ex = dspy.Example(
                task=r.get("task", ""),
                logs_preview=r.get("logs_preview", ""),
                context_keywords=r.get("context_keywords", []),
                key_points_keywords=r.get("key_points_keywords", []),
            )
        elif module == "task":
            ex = dspy.Example(
                task=r.get("task", ""),
                context=r.get("context", ""),
                plan_keywords=r.get("plan_keywords", []),
                commands_keywords=r.get("commands_keywords", []),
            )
        elif module == "code":
            ex = dspy.Example(
                snapshot=r.get("snapshot", ""),
                ask=r.get("ask", "Summarize this code snapshot."),
                keywords=r.get("keywords", []),
            )
        else:
            raise ValueError(f"Unknown module: {module}")
        out.append(ex)
    return out


def _coverage(text: str, kws: List[str]) -> Tuple[float, List[str]]:
    missing = []
    t = (text or "").lower()
    hit = 0
    for k in kws:
        if not k:
            continue
        if str(k).lower() in t:
            hit += 1
        else:
            missing.append(k)
    denom = max(1, len([k for k in kws if k]))
    return hit / denom, missing


def metric_for_module(module: str) -> Callable:
    """Return a GEPA metric(gold, pred, trace, pred_name, pred_trace) function."""

    def metric_context(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
        ctx = getattr(pred, "context", "") or ""
        points = getattr(pred, "key_points", "") or ""
        cov1, miss1 = _coverage(ctx, getattr(gold, "context_keywords", []) or [])
        cov2, miss2 = _coverage(points, getattr(gold, "key_points_keywords", []) or [])
        score = 0.5 * cov1 + 0.5 * cov2
        fb = []
        if miss1:
            fb.append("Context missing: " + ", ".join(miss1))
        if miss2:
            fb.append("Key points missing: " + ", ".join(miss2))
        if not fb:
            fb.append("Good coverage. Keep it concise and ordered.")
        return {"score": float(score), "feedback": " \n".join(fb)}

    def metric_task(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
        plan = getattr(pred, "plan", "") or ""
        cmds = getattr(pred, "commands", "") or ""
        cov1, miss1 = _coverage(plan, getattr(gold, "plan_keywords", []) or [])
        cov2, miss2 = _coverage(cmds, getattr(gold, "commands_keywords", []) or [])
        score = 0.6 * cov1 + 0.4 * cov2
        fb = []
        if miss1:
            fb.append("Plan missing: " + ", ".join(miss1))
        if miss2:
            fb.append("Commands missing: " + ", ".join(miss2))
        if not fb:
            fb.append("Plan looks solid. Ensure commands are safe/non-destructive.")
        return {"score": float(score), "feedback": " \n".join(fb)}

    def metric_code(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
        summ = getattr(pred, "summary", "") or ""
        bullets = getattr(pred, "bullets", "") or ""
        kws = getattr(gold, "keywords", []) or []
        cov1, miss1 = _coverage(summ + "\n" + bullets, kws)
        score = cov1
        fb = []
        if miss1:
            fb.append("Mention these components: " + ", ".join(miss1))
        else:
            fb.append("Good coverage. Keep bullets scannable.")
        return {"score": float(score), "feedback": " \n".join(fb)}

    return {"context": metric_context, "task": metric_task, "code": metric_code}[module]


def make_logging_metric(module: str, progress_path: Optional[str]) -> Callable:
    base_metric = metric_for_module(module)

    def logging_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
        res = base_metric(gold, pred, trace=trace, pred_name=pred_name, pred_trace=pred_trace)
        if progress_path:
            try:
                p = Path(progress_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                rec = {
                    "ts": datetime.utcnow().isoformat(),
                    "module": module,
                    "split": getattr(gold, "split", "train"),
                    "score": float(res.get("score", 0.0)),
                }
                with p.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass
        return res

    return logging_metric


def student_module(module: str) -> dspy.Module:
    from .skills.context_builder import ContextBuilder
    from .skills.task_agent import TaskAgent
    from .skills.code_context import CodeContext

    if module == "context":
        return ContextBuilder()
    if module == "task":
        return TaskAgent()
    if module == "code":
        return CodeContext()
    raise ValueError(f"Unknown module: {module}")


def run_gepa(
    module: str,
    train_jsonl: Path,
    *,
    auto: Optional[str] = None,
    max_full_evals: Optional[int] = None,
    max_metric_calls: Optional[int] = None,
    reflection_lm: Optional[dspy.LM] = None,
    log_dir: Optional[str] = None,
    track_stats: bool = True,
    progress_path: Optional[str] = None,
) -> dspy.Module:
    trainset = load_examples_for_module(module, train_jsonl)
    for ex in trainset:
        setattr(ex, "split", "train")
    metric = make_logging_metric(module, progress_path)

    if not (auto or max_full_evals or max_metric_calls):
        auto = "light"

    gepa = dspy.GEPA(
        metric=metric,
        auto=auto,
        max_full_evals=max_full_evals,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        log_dir=log_dir,
        track_stats=track_stats,
    )
    student = student_module(module)
    optimized = gepa.compile(student, trainset=trainset, valset=trainset)
    return optimized


def run_gepa_with_val(
    module: str,
    train_jsonl: Path,
    val_jsonl: Path,
    *,
    auto: Optional[str] = None,
    max_full_evals: Optional[int] = None,
    max_metric_calls: Optional[int] = None,
    reflection_lm: Optional[dspy.LM] = None,
    log_dir: Optional[str] = None,
    track_stats: bool = True,
    progress_path: Optional[str] = None,
) -> dspy.Module:
    trainset = load_examples_for_module(module, train_jsonl)
    valset = load_examples_for_module(module, val_jsonl)
    for ex in trainset:
        setattr(ex, "split", "train")
    for ex in valset:
        setattr(ex, "split", "val")
    metric = make_logging_metric(module, progress_path)

    if not (auto or max_full_evals or max_metric_calls):
        auto = "light"

    gepa = dspy.GEPA(
        metric=metric,
        auto=auto,
        max_full_evals=max_full_evals,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        log_dir=log_dir,
        track_stats=track_stats,
    )
    student = student_module(module)
    optimized = gepa.compile(student, trainset=trainset, valset=valset)
    return optimized


def evaluate_on_set(module: str, prog: dspy.Module, jsonl_path: Path) -> Dict[str, float]:
    """Evaluate a trained module on a given set using the module-specific metric."""
    metric = metric_for_module(module)
    examples = load_examples_for_module(module, jsonl_path)
    scores: List[float] = []
    for ex in examples:
        try:
            pred = prog(**ex.inputs())  # type: ignore
            res = metric(ex, pred)
            scores.append(float(res.get("score", 0.0)))
        except Exception:
            scores.append(0.0)
    avg = sum(scores) / max(1, len(scores))
    return {"n": float(len(scores)), "avg_score": float(avg)}
