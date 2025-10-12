from __future__ import annotations

from typing import Any, List

from ..rl.rlkit import AgentResult, VerifierProtocol


class LintCleanVerifier:
    kind = "lint_clean"

    def __call__(self, result: AgentResult) -> float:
        exit_code = result.metrics.get("lint_exit_code")
        if exit_code is not None:
            try:
                return 1.0 if int(exit_code) == 0 else 0.0
            except Exception:
                return 0.0
        lint_ok = result.metrics.get("lint_ok")
        if lint_ok is not None:
            return 1.0 if bool(lint_ok) else 0.0
        issues = result.metrics.get("lint_issues")
        if issues is not None:
            try:
                return 1.0 if int(issues) == 0 else 0.0
            except Exception:
                return 0.0
        return 0.0


class TestsPassedVerifier:
    kind = "tests_passed"

    def __call__(self, result: AgentResult) -> float:
        exit_code = result.metrics.get("tests_exit_code")
        if exit_code is not None:
            try:
                return 1.0 if int(exit_code) == 0 else 0.0
            except Exception:
                return 0.0
        passed = result.metrics.get("tests_passed")
        total = result.metrics.get("tests_total")
        try:
            passed_val = float(passed or 0.0)
            total_val = float(total or 0.0)
        except Exception:
            return 0.0
        return (passed_val / total_val) if total_val > 0 else 0.0


class DiffSizeVerifier:
    kind = "diff_size"

    def __call__(self, result: AgentResult) -> float:
        blast = result.metrics.get("blast_radius")
        if blast is None:
            return 0.0
        try:
            value = float(blast)
        except Exception:
            return 0.0
        if value <= 0:
            return 1.0
        if value >= 400:
            return 0.0
        return max(0.0, 1.0 - (value / 400.0))


def get_code_quality_verifiers() -> List[VerifierProtocol]:
    return [LintCleanVerifier(), TestsPassedVerifier(), DiffSizeVerifier()]

