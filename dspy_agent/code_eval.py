from __future__ import annotations

import os
import shutil
import subprocess as sp
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .patcher import apply_unified_patch


@dataclass
class EvalConfig:
    workspace: Path
    test_cmd: Optional[str] = None
    type_cmd: Optional[str] = None
    lint_cmd: Optional[str] = None
    timeout_sec: int = 180


@dataclass
class EvalResult:
    score: float
    feedback: str
    passed: bool
    tests_ok: bool
    type_ok: bool
    lint_ok: bool


def _run(cmd: str, cwd: Path, timeout: int) -> Tuple[bool, str]:
    try:
        proc = sp.run(cmd, cwd=str(cwd), shell=True, stdout=sp.PIPE, stderr=sp.STDOUT, timeout=timeout, text=True)
        ok = (proc.returncode == 0)
        return ok, proc.stdout
    except Exception as e:
        return False, str(e)


def evaluate_patch(patch: str, cfg: EvalConfig) -> EvalResult:
    """Apply a patch in a temp copy of the workspace and run test/type/lint commands.

    Returns a composite score in [0,1] with feedback.
    """
    ws = cfg.workspace.resolve()
    tmp = Path(tempfile.mkdtemp(prefix="dspy_eval_"))
    try:
        # Copy workspace shallowly (best-effort)
        for item in ws.iterdir():
            if item.name in {".git", ".venv", "venv", "env"}:
                continue
            dst = tmp / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        ok, msg = apply_unified_patch(patch, tmp)
        if not ok:
            return EvalResult(score=0.0, feedback=f"patch failed: {msg}", passed=False, tests_ok=False, type_ok=False, lint_ok=False)
        # Choose defaults if not provided
        test_cmd = cfg.test_cmd
        if test_cmd is None:
            test_cmd = "pytest -q" if (tmp / "tests").exists() else None
        type_cmd = cfg.type_cmd or "python -m compileall -q ."
        lint_cmd = cfg.lint_cmd  # optional

        tests_ok = True
        type_ok = True
        lint_ok = True
        fb_parts = []
        if test_cmd:
            tests_ok, out = _run(test_cmd, tmp, cfg.timeout_sec)
            fb_parts.append(f"tests: {'ok' if tests_ok else 'fail'}\n{out[-2000:]}" if out else "tests: no output")
        # Always run type check fallback
        type_ok, out = _run(type_cmd, tmp, cfg.timeout_sec)
        fb_parts.append(f"types: {'ok' if type_ok else 'fail'}\n{out[-1000:]}" if out else "types: no output")
        if lint_cmd:
            lint_ok, out = _run(lint_cmd, tmp, cfg.timeout_sec)
            fb_parts.append(f"lint: {'ok' if lint_ok else 'fail'}\n{out[-1000:]}" if out else "lint: no output")

        # Score: tests 0.6, types 0.25, lint 0.15 (if provided)
        w_tests = 0.6 if test_cmd else 0.0
        w_type = 0.25
        w_lint = 0.15 if lint_cmd else 0.0
        denom = max(1e-6, w_tests + w_type + w_lint)
        score = (w_tests * (1.0 if tests_ok else 0.0) + w_type * (1.0 if type_ok else 0.0) + w_lint * (1.0 if lint_ok else 0.0)) / denom
        fb = "\n\n".join(fb_parts)
        return EvalResult(score=float(score), feedback=fb, passed=tests_ok and type_ok and (lint_ok or not lint_cmd), tests_ok=tests_ok, type_ok=type_ok, lint_ok=lint_ok)
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

