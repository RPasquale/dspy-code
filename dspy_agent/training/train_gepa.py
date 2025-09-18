from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from datetime import datetime

import dspy

from ..db import get_enhanced_data_manager, TrainingMetrics, Environment, create_log_entry


def _read_jsonl(path: Path) -> List[dict]:
    data: List[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_examples_for_module(module: str, jsonl_path: Path, *, code_summary: Optional[str] = None) -> List[dspy.Example]:
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
            ex = ex.with_inputs("task", "logs_preview")
        elif module == "task":
            ex = dspy.Example(
                task=r.get("task", ""),
                context=(r.get("context", "") + (f"\n\nCode Summary:\n{code_summary}" if code_summary else "")),
                plan_keywords=r.get("plan_keywords", []),
                commands_keywords=r.get("commands_keywords", []),
            )
            ex = ex.with_inputs("task", "context")
        elif module == "code":
            ex = dspy.Example(
                snapshot=r.get("snapshot", ""),
                ask=r.get("ask", "Summarize this code snapshot."),
                keywords=r.get("keywords", []),
            )
            ex = ex.with_inputs("snapshot", "ask")
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
        # Heuristic penalty for dangerous commands
        dangerous = [
            "rm -rf", "git reset --hard", "mkfs", ":> /", "dd if=", "sudo ",
            "curl ", " | sh", " | bash", "chmod -R 777 /", "chown -R /",
        ]
        if any(tok in cmds.lower() for tok in dangerous):
            score = max(0.0, score - 0.2)
        fb = []
        if miss1:
            fb.append("Plan missing: " + ", ".join(miss1))
        if miss2:
            fb.append("Commands missing: " + ", ".join(miss2))
        if any(tok in cmds.lower() for tok in dangerous):
            fb.append("Avoid destructive commands; prefer safe, reviewable steps.")
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


def _score_of(res) -> float:
    try:
        if isinstance(res, (int, float)):
            return float(res)
        if isinstance(res, dict):
            return float(res.get("score", 0.0))
    except Exception:
        pass
    return 0.0


def make_logging_metric(module: str, progress_path: Optional[str], session_id: Optional[str] = None) -> Callable:
    base_metric = metric_for_module(module)
    data_manager = get_enhanced_data_manager()

    def logging_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
        res = base_metric(gold, pred, trace=trace, pred_name=pred_name, pred_trace=pred_trace)
        score = _score_of(res)
        
        # Store in RedDB
        try:
            training_metrics = TrainingMetrics(
                session_id=session_id or f"{module}_gepa_{int(time.time())}",
                timestamp=time.time(),
                epoch=getattr(pred, 'epoch', 1),
                training_accuracy=score,
                validation_accuracy=score,
                loss=1.0 - score,  # Convert score to loss
                learning_rate=0.001,  # Default learning rate
                batch_size=1,
                model_type="gepa",
                environment=Environment.DEVELOPMENT,
                hyperparameters={
                    "module": module,
                    "split": getattr(gold, "split", "train"),
                    "auto_mode": "light"
                },
                convergence_metrics={
                    "feedback": res.get("feedback", "") if isinstance(res, dict) else "",
                    "score": score,
                    "execution_time": time.time()
                }
            )
            data_manager.store_training_metrics(training_metrics)
            
            # Log the training step
            log_entry = create_log_entry(
                level="INFO",
                source=f"training.{module}",
                message=f"Training step completed with score: {score:.3f}",
                context={
                    "session_id": training_metrics.session_id,
                    "module": module,
                    "score": score,
                    "split": getattr(gold, "split", "train"),
                    "feedback": res.get("feedback", "") if isinstance(res, dict) else ""
                },
                environment=Environment.DEVELOPMENT
            )
            data_manager.log(log_entry)
            
        except Exception as e:
            # Fallback logging if RedDB fails
            error_log = create_log_entry(
                level="ERROR",
                source=f"training.{module}",
                message=f"Failed to store training metrics in RedDB: {str(e)}",
                context={"error": str(e), "module": module},
                environment=Environment.DEVELOPMENT
            )
            try:
                data_manager.log(error_log)
            except:
                pass  # Silent fallback
        
        # Maintain file logging for backward compatibility
        if progress_path:
            try:
                p = Path(progress_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                rec = {
                    "ts": datetime.utcnow().isoformat(),
                    "module": module,
                    "split": getattr(gold, "split", "train"),
                    "score": score,
                }
                with p.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass
        
        # Return a pure float score to be robust across DSPy versions
        return score

    return logging_metric


def student_module(module: str) -> dspy.Module:
    from ..skills.context_builder import ContextBuilder
    from ..skills.task_agent import TaskAgent
    from ..skills.code_context import CodeContext

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
    code_summary: Optional[str] = None,
) -> dspy.Module:
    # Generate unique session ID
    session_id = f"{module}_gepa_{int(time.time())}"
    data_manager = get_enhanced_data_manager()
    
    # Log training session start
    start_log = create_log_entry(
        level="INFO",
        source=f"training.{module}",
        message=f"Starting GEPA {module} training session: {session_id}",
        context={
            "session_id": session_id,
            "module": module,
            "train_jsonl": str(train_jsonl),
            "auto": auto,
            "max_full_evals": max_full_evals,
            "max_metric_calls": max_metric_calls,
            "track_stats": track_stats
        },
        environment=Environment.DEVELOPMENT
    )
    data_manager.log(start_log)
    
    trainset = load_examples_for_module(module, train_jsonl, code_summary=code_summary)
    for ex in trainset:
        setattr(ex, "split", "train")
    
    # Log dataset info
    dataset_log = create_log_entry(
        level="INFO",
        source=f"training.{module}",
        message=f"Loaded training dataset with {len(trainset)} examples",
        context={
            "session_id": session_id,
            "module": module,
            "dataset_size": len(trainset)
        },
        environment=Environment.DEVELOPMENT
    )
    data_manager.log(dataset_log)
    
    metric = make_logging_metric(module, progress_path, session_id)

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
    
    # Time the compilation
    compile_start = time.time()
    optimized = gepa.compile(student, trainset=trainset, valset=trainset)
    compile_time = time.time() - compile_start
    
    # Log training completion
    completion_log = create_log_entry(
        level="INFO",
        source=f"training.{module}",
        message=f"Completed GEPA {module} training session: {session_id}",
        context={
            "session_id": session_id,
            "module": module,
            "compile_time": compile_time,
            "dataset_size": len(trainset),
            "auto_mode": auto
        },
        environment=Environment.DEVELOPMENT
    )
    data_manager.log(completion_log)
    
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
    code_summary: Optional[str] = None,
) -> dspy.Module:
    trainset = load_examples_for_module(module, train_jsonl, code_summary=code_summary)
    valset = load_examples_for_module(module, val_jsonl, code_summary=code_summary)
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
