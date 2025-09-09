from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy

from .skills.orchestrator import Orchestrator
from .orchestrator_runtime import evaluate_tool_choice
from datetime import datetime


def _read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _state_text(workspace: Path, logs: Optional[Path]) -> str:
    has_logs = bool(logs and Path(logs).exists())
    return f"workspace={workspace} logs={logs or (workspace/'logs')} logs_exist={has_logs}"


def load_orchestrator_trainset(jsonl_path: Path) -> List[dspy.Example]:
    """Schema per line JSON:
    {
      "query": str,
      "workspace": str,  # path
      "logs": str | null, # path
      "targets": [str]    # optional: keywords to hit
    }
    """
    data = _read_jsonl(jsonl_path)
    examples: List[dspy.Example] = []
    for r in data:
        q = r.get("query", "")
        ws = Path(r.get("workspace", ".")).resolve()
        logs = r.get("logs")
        lp = Path(logs).resolve() if logs else None
        st = _state_text(ws, lp)
        ex = dspy.Example(query=q, state=st, workspace=str(ws), logs=str(lp) if lp else "", targets=r.get("targets", []))
        examples.append(ex)
    return examples


def gepa_metric_for_orchestrator(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
    """Evaluate chosen tool by executing a safe evaluation and providing feedback."""
    tool = getattr(pred, "tool", "")
    args_json = getattr(pred, "args_json", "{}")
    ws = Path(getattr(gold, "workspace", "."))
    logs = Path(getattr(gold, "logs", "")).resolve() if getattr(gold, "logs", "") else None
    targets = getattr(gold, "targets", []) or []

    outcome = evaluate_tool_choice(tool, args_json, workspace=ws, logs_path=logs, targets=targets)
    # Return score + textual feedback
    fb = f"{outcome.feedback} | evidence: {outcome.evidence}"
    return {"score": float(outcome.score), "feedback": fb}


def make_logging_metric_orchestrator(progress_path: Optional[str]):
    def metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
        res = gepa_metric_for_orchestrator(gold, pred, trace=trace, pred_name=pred_name, pred_trace=pred_trace)
        if progress_path:
            try:
                p = Path(progress_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                rec = {
                    "ts": datetime.utcnow().isoformat(),
                    "module": "orchestrator",
                    "split": getattr(gold, "split", "train"),
                    "score": float(res.get("score", 0.0)),
                }
                with p.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass
        return res

    return metric


def run_gepa_orchestrator(train_jsonl: Path, *, auto: Optional[str] = "light", reflection_lm: Optional[dspy.LM] = None, log_dir: Optional[str] = None, track_stats: bool = True, progress_path: Optional[str] = None) -> dspy.Module:
    trainset = load_orchestrator_trainset(train_jsonl)
    for ex in trainset:
        setattr(ex, "split", "train")
    if not auto:
        auto = "light"
    metric = make_logging_metric_orchestrator(progress_path)
    gepa = dspy.GEPA(metric=metric, auto=auto, reflection_lm=reflection_lm, log_dir=str(log_dir) if log_dir else None, track_stats=track_stats)
    student = Orchestrator()
    optimized = gepa.compile(student, trainset=trainset, valset=trainset)
    return optimized


def run_gepa_orchestrator_with_val(train_jsonl: Path, val_jsonl: Path, *, auto: Optional[str] = "light", reflection_lm: Optional[dspy.LM] = None, log_dir: Optional[str] = None, track_stats: bool = True, progress_path: Optional[str] = None) -> dspy.Module:
    trainset = load_orchestrator_trainset(train_jsonl)
    valset = load_orchestrator_trainset(val_jsonl)
    for ex in trainset:
        setattr(ex, "split", "train")
    for ex in valset:
        setattr(ex, "split", "val")
    if not auto:
        auto = "light"
    metric = make_logging_metric_orchestrator(progress_path)
    gepa = dspy.GEPA(metric=metric, auto=auto, reflection_lm=reflection_lm, log_dir=str(log_dir) if log_dir else None, track_stats=track_stats)
    student = Orchestrator()
    optimized = gepa.compile(student, trainset=trainset, valset=valset)
    return optimized


def evaluate_orchestrator(prog: dspy.Module, jsonl_path: Path) -> Dict[str, float]:
    examples = load_orchestrator_trainset(jsonl_path)
    scores: List[float] = []
    for ex in examples:
        try:
            pred = prog(query=ex.query, state=ex.state)  # type: ignore
            res = gepa_metric_for_orchestrator(ex, pred)
            scores.append(float(res.get("score", 0.0)))
        except Exception:
            scores.append(0.0)
    avg = sum(scores) / max(1, len(scores))
    return {"n": float(len(scores)), "avg_score": float(avg)}
