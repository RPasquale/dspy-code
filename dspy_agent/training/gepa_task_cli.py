from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import dspy  # type: ignore

from .train_gepa import (
    evaluate_on_set,
    run_gepa,
    run_gepa_with_val,
)

try:
    from dspy_agent.cli import _maybe_configure_lm, _record_gepa_outcome  # type: ignore
except ImportError:  # pragma: no cover - fallback when CLI helpers unavailable
    _maybe_configure_lm = None  # type: ignore
    _record_gepa_outcome = None  # type: ignore


def _configure_lm(
    use_ollama: bool,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    workspace: Optional[Path],
) -> Optional[dspy.LM]:
    if _maybe_configure_lm is None:
        return None
    return _maybe_configure_lm(True, use_ollama, model, base_url, api_key, workspace=workspace)


def _dump_result_marker(payload: dict) -> None:
    marker = json.dumps(payload, ensure_ascii=False)
    # Runner looks for this marker inside stdout to attach structured metadata.
    print(f"__DSPY_RESULT__:{marker}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute a GEPA optimisation pass and emit structured results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--module", required=True, choices=["context", "task", "code"], help="GEPA module to optimise")
    parser.add_argument("--train-jsonl", required=True, type=Path, help="Training dataset JSONL")
    parser.add_argument("--val-jsonl", type=Path, help="Optional validation dataset JSONL")
    parser.add_argument("--test-jsonl", type=Path, help="Optional test dataset JSONL")
    parser.add_argument("--dataset-dir", type=Path, help="Directory containing <module>_train.jsonl etc.")
    parser.add_argument("--workspace", type=Path, help="Workspace root for saving outcomes (defaults to CWD)")
    parser.add_argument("--auto", default="light", help="GEPA auto budget label")
    parser.add_argument("--max-full-evals", type=int, help="Override GEPA max_full_evals")
    parser.add_argument("--max-metric-calls", type=int, help="Override GEPA max_metric_calls")
    parser.add_argument("--log-dir", type=Path, help="Directory for GEPA logs/progress")
    parser.add_argument("--progress-path", type=Path, help="Progress JSONL path (defaults under log-dir)")
    parser.add_argument("--output-json", required=True, type=Path, help="Where to write structured result JSON")
    parser.add_argument("--use-ollama", action="store_true", default=False, help="Use Ollama for reflection LM")
    parser.add_argument("--model", help="Reflection model name for GEPA")
    parser.add_argument("--base-url", help="LLM base URL")
    parser.add_argument("--api-key", help="LLM API key")
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Soft timeout (for metadata only)")
    args = parser.parse_args(argv)

    module = args.module.lower()
    workspace = (args.workspace or Path.cwd()).resolve()

    train_jsonl = args.train_jsonl
    val_jsonl = args.val_jsonl
    test_jsonl = args.test_jsonl

    if args.dataset_dir:
        ds = args.dataset_dir
        if not train_jsonl.exists():
            cand = ds / f"{module}_train.jsonl"
            if cand.exists():
                train_jsonl = cand
        if not val_jsonl and (ds / f"{module}_val.jsonl").exists():
            val_jsonl = ds / f"{module}_val.jsonl"
        if not test_jsonl and (ds / f"{module}_test.jsonl").exists():
            test_jsonl = ds / f"{module}_test.jsonl"

    if not train_jsonl.exists():
        raise SystemExit(f"Training file does not exist: {train_jsonl}")

    log_dir = (args.log_dir or workspace / f".gepa_{module}").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    progress_path = (args.progress_path or (log_dir / "progress.jsonl")).resolve()
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    lm = _configure_lm(
        use_ollama=args.use_ollama,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        workspace=workspace,
    )

    auto = args.auto
    if not auto:
        auto = None

    optimized = None
    try:
        if val_jsonl and val_jsonl.exists():
            optimized = run_gepa_with_val(
                module=module,
                train_jsonl=train_jsonl,
                val_jsonl=val_jsonl,
                auto=auto,
                max_full_evals=args.max_full_evals,
                max_metric_calls=args.max_metric_calls,
                reflection_lm=lm,
                log_dir=str(log_dir),
                track_stats=True,
                progress_path=str(progress_path),
            )
        else:
            optimized = run_gepa(
                module=module,
                train_jsonl=train_jsonl,
                auto=auto,
                max_full_evals=args.max_full_evals,
                max_metric_calls=args.max_metric_calls,
                reflection_lm=lm,
                log_dir=str(log_dir),
                track_stats=True,
                progress_path=str(progress_path),
            )
    except Exception as exc:  # pragma: no cover - best-effort resilience
        output = {
            "module": module,
            "success": False,
            "error": str(exc),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2))
        _dump_result_marker(output)
        raise

    test_metrics = {}
    if test_jsonl and test_jsonl.exists():
        try:
            test_metrics = evaluate_on_set(module, optimized, test_jsonl)
        except Exception as exc:  # pragma: no cover
            test_metrics = {"error": str(exc)}

    if _record_gepa_outcome is not None:
        try:
            progress_obj = progress_path if progress_path.exists() else None
            _record_gepa_outcome(module, optimized, workspace, progress_obj)
        except Exception:  # pragma: no cover
            pass

    output = {
        "module": module,
        "success": True,
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl) if val_jsonl else None,
        "test_jsonl": str(test_jsonl) if test_jsonl else None,
        "auto": auto,
        "log_dir": str(log_dir),
        "progress_path": str(progress_path),
        "workspace": str(workspace),
        "test_metrics": test_metrics,
        "timeout_seconds": args.timeout_seconds,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    _dump_result_marker(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
