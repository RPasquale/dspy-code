from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import json

import dspy

from ..skills.code_edit import CodeEdit
from ..code_tools.code_eval import EvalConfig, evaluate_patch


def _read_jsonl(path: Path) -> List[dict]:
    data: List[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_codegen_set(jsonl_path: Path) -> List[dspy.Example]:
    """Dataset schema per line:
    {"task": str, "context": str, "file_hints": str}
    """
    raw = _read_jsonl(jsonl_path)
    out: List[dspy.Example] = []
    for r in raw:
        ex = dspy.Example(task=r.get("task", ""), context=r.get("context", ""), file_hints=r.get("file_hints", ""))
        ex = ex.with_inputs("task", "context")
        out.append(ex)
    return out


def make_codegen_metric(workspace: Path, test_cmd: Optional[str], type_cmd: Optional[str], lint_cmd: Optional[str]):
    def metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
        patch = getattr(pred, "patch", "") or ""
        if not patch.strip():
            return 0.0
        cfg = EvalConfig(workspace=workspace, test_cmd=test_cmd, type_cmd=type_cmd, lint_cmd=lint_cmd)
        res = evaluate_patch(patch, cfg)
        return res.score
    return metric


def run_gepa_codegen(
    train_jsonl: Path,
    *,
    workspace: Path,
    test_cmd: Optional[str],
    type_cmd: Optional[str],
    lint_cmd: Optional[str],
    auto: Optional[str] = "light",
    reflection_lm: Optional[dspy.LM] = None,
    log_dir: Optional[str] = None,
    track_stats: bool = True,
) -> dspy.Module:
    trainset = load_codegen_set(train_jsonl)
    metric = make_codegen_metric(workspace, test_cmd, type_cmd, lint_cmd)
    student = CodeEdit(use_cot=True)
    gepa = dspy.GEPA(metric=metric, auto=auto, reflection_lm=reflection_lm, log_dir=log_dir, track_stats=track_stats)
    optimized = gepa.compile(student, trainset=trainset, valset=trainset)
    return optimized

