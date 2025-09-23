from __future__ import annotations

"""
Custom verifiers for DSPy RL reward shaping.

Each verifier exposes:
  - kind: a short key used in reward aggregation and analytics
  - __call__(result: AgentResult) -> float: returns a scalar score

These verifiers read from AgentResult.metrics, which the toolchain populates
with fields such as: pass_rate, lint_issues, blast_radius, quality_policy,
tests_total, tests_passed, etc. See dspy_agent.rl.rlkit.ToolchainExecutor.
"""

from typing import Any, Mapping


class _BaseVerifier:
    kind = "base"
    def __call__(self, result) -> float:  # pragma: no cover
        raise NotImplementedError


class PassRateImprovementVerifier(_BaseVerifier):
    kind = "pass_rate_improve"
    def __call__(self, result) -> float:
        m: Mapping[str, Any] = result.metrics
        pr = float(m.get("pass_rate", 0.0) or 0.0)
        # Optionally compare to baseline if present
        base = float(m.get("baseline_pass_rate", 0.0) or 0.0)
        return pr - base


class LowBlastRadiusVerifier(_BaseVerifier):
    kind = "low_blast"
    def __call__(self, result) -> float:
        m: Mapping[str, Any] = result.metrics
        br = float(m.get("blast_radius", 0.0) or 0.0)
        # Invert: smaller blast radius should yield higher score
        return -br


class QualityPolicyVerifier(_BaseVerifier):
    kind = "quality_policy"
    def __call__(self, result) -> float:
        # Score adherence to policy (0..1)
        m: Mapping[str, Any] = result.metrics
        return float(m.get("quality_policy", 1.0) or 0.0)


class LintIssuesPenaltyVerifier(_BaseVerifier):
    kind = "lint_penalty"
    def __call__(self, result) -> float:
        m: Mapping[str, Any] = result.metrics
        issues = int(m.get("lint_issues", 0) or 0)
        return -float(issues)


def get_verifiers():
    """Return a list of custom verifiers.

    You can point RL to this module (dspy_agent.verifiers.custom) to use them.
    """
    return [
        PassRateImprovementVerifier(),
        LowBlastRadiusVerifier(),
        QualityPolicyVerifier(),
        LintIssuesPenaltyVerifier(),
    ]

