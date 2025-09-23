from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Type
import json

import dspy

# Local signatures
from ..skills.code_context import CodeContextSig
from ..skills.code_context_rag import CodeContextRAGSig
from ..skills.task_agent import PlanTaskSig
from ..skills.file_locator import FileLocatorSig
from ..skills.test_planner import TestPlannerSig
from ..skills.patch_verifier import PatchVerifierSig
from ..db import get_enhanced_data_manager, SignatureMetrics
from datetime import datetime


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _to_examples(rows: List[Dict[str, Any]], sig: Type[dspy.Signature]) -> List[dspy.Example]:
    exs: List[dspy.Example] = []
    # Build an example per row; assume keys align to signature fields
    in_fields = [n for n, f in sig.__fields__.items() if getattr(f, 'io', None) == 'I']
    out_fields = [n for n, f in sig.__fields__.items() if getattr(f, 'io', None) == 'O']
    for r in rows:
        inputs = {k: r.get(k, '') for k in in_fields}
        outputs = {k: r.get(k, '') for k in out_fields}
        exs.append(dspy.Example(**inputs).with_outputs(*out_fields, **outputs))
    return exs


def _resolve_signature(name: str) -> Type[dspy.Signature]:
    key = name.strip().lower()
    mapping: Dict[str, Type[dspy.Signature]] = {
        'codectx': CodeContextSig,
        'codectx_rag': CodeContextRAGSig,
        'task': PlanTaskSig,
        'file_locator': FileLocatorSig,
        'test_planner': TestPlannerSig,
        'patch_verifier': PatchVerifierSig,
    }
    if key not in mapping:
        raise ValueError(f"Unknown module for teleprompt: {name}. Choose from: {', '.join(mapping)}")
    return mapping[key]


def _default_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """Coarse metric: reward non-empty key outputs. Avoid overfitting to length."""
    def _sigmoid_len(x: str, k: float = 80.0) -> float:
        n = float(len((x or '').strip()))
        return 1.0 / (1.0 + dspy.math.exp(-(n - k) / 20.0))
    score = 0.0
    # Count non-empty outputs on prediction
    keys = [k for k in dir(pred) if not k.startswith('_')]
    for k in keys:
        v = getattr(pred, k, None)
        if isinstance(v, str) and v.strip():
            score += 0.25 * _sigmoid_len(v)
    return max(0.0, min(1.0, score))


def run_teleprompt(
    module: str,
    train_jsonl: Path,
    *,
    val_jsonl: Optional[Path] = None,
    method: str = 'bootstrap',
    shots: int = 8,
    reflection_lm: Optional[dspy.LM] = None,
    log_dir: Optional[Path] = None,
) -> dspy.Module:
    sig = _resolve_signature(module)
    train_rows = _load_jsonl(train_jsonl)
    trainset = _to_examples(train_rows, sig)
    valset = None
    if val_jsonl and val_jsonl.exists():
        val_rows = _load_jsonl(val_jsonl)
        valset = _to_examples(val_rows, sig)

    # Student: Predict on the signature (CoT can be enabled after teleprompting if desired)
    class Student(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(sig)
        def forward(self, **kwargs):
            return self.predict(**kwargs)

    student = Student()

    # Teleprompter selection
    method_l = (method or 'bootstrap').strip().lower()
    if method_l in {'bootstrap', 'bootstrapicl', 'bootstrap_icl'}:
        try:
            from dspy.teleprompt import BootstrapFewShotWithRandomSearch as BootstrapICL
        except Exception:
            from dspy.teleprompt import Bootstrap as BootstrapICL  # type: ignore
        tele = BootstrapICL(metric=_default_metric, max_labeled_demos=int(shots))
    elif method_l in {'mipro', 'miprov2'}:
        try:
            from dspy.teleprompt import MIPROv2
        except Exception:
            raise RuntimeError("MIPROv2 teleprompter not available in this DSPy version.")
        tele = MIPROv2(metric=_default_metric, num_candidates=int(shots))
    else:
        raise ValueError("method must be one of: bootstrap, mipro")

    optimized = tele.compile(student, trainset=trainset, valset=valset or trainset)
    return optimized


def evaluate_program_on_jsonl(sig: Type[dspy.Signature], prog: dspy.Module, jsonl_path: Path) -> Dict[str, float]:
    rows = _load_jsonl(jsonl_path)
    in_fields = [n for n, f in sig.__fields__.items() if getattr(f, 'io', None) == 'I']
    scores: List[float] = []
    for r in rows:
        try:
            kwargs = {k: r.get(k, '') for k in in_fields}
            pred = prog(**kwargs)
            score = _default_metric(dspy.Example(**kwargs), pred)
            scores.append(float(score))
        except Exception:
            scores.append(0.0)
    avg = sum(scores) / max(1, len(scores))
    return {"n": float(len(scores)), "avg_score": float(avg)}


def run_teleprompt_suite(
    modules: List[str],
    methods: List[str],
    dataset_dir: Path,
    *,
    shots: int = 8,
    reflection_lm: Optional[dspy.LM] = None,
    log_dir: Optional[Path] = None,
    save_best_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    dm = get_enhanced_data_manager()
    results: Dict[str, Any] = {"experiments": []}
    ts = datetime.utcnow().isoformat()
    if save_best_dir:
        try:
            save_best_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    for mod in modules:
        sig = _resolve_signature(mod)
        sig_name = sig.__name__
        # Expect dataset files named {mod}_train.jsonl etc.
        train_jsonl = dataset_dir / f"{mod}_train.jsonl"
        val_jsonl = dataset_dir / f"{mod}_val.jsonl"
        test_jsonl = dataset_dir / f"{mod}_test.jsonl"
        if not train_jsonl.exists():
            # Fallback to generic train.jsonl
            train_jsonl = dataset_dir / "train.jsonl"
        valset = val_jsonl if val_jsonl.exists() else (dataset_dir / "val.jsonl")
        testset = test_jsonl if test_jsonl.exists() else (dataset_dir / "test.jsonl")
        best_score = -1.0
        best_prog = None
        best_meta: Dict[str, Any] = {}
        for m in methods:
            try:
                prog = run_teleprompt(
                    module=mod,
                    train_jsonl=train_jsonl,
                    val_jsonl=valset if valset.exists() else None,
                    method=m,
                    shots=shots,
                    reflection_lm=reflection_lm,
                    log_dir=log_dir,
                )
            except Exception as e:
                results["experiments"].append({
                    "module": mod, "signature": sig_name, "method": m, "error": str(e), "timestamp": ts
                })
                continue
            metrics_val = {"n": 0.0, "avg_score": 0.0}
            metrics_test = {"n": 0.0, "avg_score": 0.0}
            if valset.exists():
                metrics_val = evaluate_program_on_jsonl(sig, prog, valset)
            if testset.exists():
                metrics_test = evaluate_program_on_jsonl(sig, prog, testset)
            avg = metrics_test.get("avg_score") or metrics_val.get("avg_score") or 0.0
            meta = {
                "module": mod,
                "signature": sig_name,
                "method": m,
                "shots": int(shots),
                "val": metrics_val,
                "test": metrics_test,
                "timestamp": ts,
            }
            results["experiments"].append(meta)
            # Save best program JSON-ish representation
            if save_best_dir:
                try:
                    outp = save_best_dir / f"tele_{mod}_{m}_{int(time.time())}.json"
                    with outp.open('w') as f:
                        json.dump({"program": str(prog), **meta}, f, indent=2)
                except Exception:
                    pass
            # Update signature metrics/optimization history
            try:
                current = dm.get_signature_metrics(sig_name)
                if current is None:
                    current = SignatureMetrics(
                        signature_name=sig_name,
                        performance_score=avg * 100.0,
                        success_rate=avg * 100.0,
                        avg_response_time=2.0,
                        memory_usage="n/a",
                        iterations=1,
                        last_updated=datetime.utcnow().isoformat(),
                        signature_type="analysis",
                        active=True,
                        optimization_history=[],
                    )
                current.performance_score = max(current.performance_score, avg * 100.0)
                current.success_rate = max(current.success_rate, avg * 100.0)
                current.iterations = int(current.iterations or 0) + 1
                current.last_updated = datetime.utcnow().isoformat()
                current.optimization_history.append({
                    "timestamp": time.time(),
                    "type": "teleprompt",
                    "method": m,
                    "shots": int(shots),
                    "val": metrics_val,
                    "test": metrics_test,
                })
                dm.store_signature_metrics(current)
            except Exception:
                pass
            if float(avg) > float(best_score):
                best_score = float(avg)
                best_prog = prog
                best_meta = meta
        results.setdefault("best", {})[mod] = {"score": best_score, **best_meta}
    return results
