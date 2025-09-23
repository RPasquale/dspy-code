from __future__ import annotations

from typing import Optional

try:
    import dspy  # type: ignore
except Exception:  # pragma: no cover
    dspy = None  # type: ignore


class PlanTaskSig((dspy.Signature if dspy is not None else object)):
    """Given a task and context, propose a short, safe plan and exact commands.

    Constraints:
    - Non-destructive by default; call out any risky steps explicitly.
    - Prefer concrete, copy-pasteable shell commands aligned to the plan.
    - Keep it concise (<= 10 steps) and include assumptions/risks.
    """

    task: str = (dspy.InputField() if dspy is not None else None)  # type: ignore
    context: str = (dspy.InputField() if dspy is not None else None)  # type: ignore

    plan: str = (dspy.OutputField(desc="Ordered, concise steps to proceed") if dspy is not None else "")  # type: ignore
    commands: str = (dspy.OutputField(desc="Copy-pasteable shell commands") if dspy is not None else "")  # type: ignore
    assumptions: str = (dspy.OutputField(desc="Assumptions or prerequisites", default="") if dspy is not None else "")  # type: ignore
    risks: str = (dspy.OutputField(desc="Potential risks and mitigations", default="") if dspy is not None else "")  # type: ignore
    rationale: str = (dspy.OutputField(desc="Why this plan/commands", default="") if dspy is not None else "")  # type: ignore


class TaskAgent((dspy.Module if dspy is not None else object)):
    def __init__(self, use_cot: Optional[bool] = None):
        if dspy is not None:
            super().__init__()
            self.fast = dspy.Predict(PlanTaskSig)
            cot = getattr(dspy, 'ChainOfThought', None)
            self.slow = (cot or dspy.Predict)(PlanTaskSig)
        else:
            class _Predict:
                def __init__(self, *a, **k): pass
                def __call__(self, **kw):
                    from types import SimpleNamespace
                    return SimpleNamespace(plan="", commands="", assumptions="", risks="", rationale="")
            self.fast = _Predict()
            self.slow = _Predict()
        self.use_cot = use_cot

    def forward(self, task: str, context: str):
        if self.use_cot is True:
            return self.slow(task=task, context=context)
        pred = self.fast(task=task, context=context)
        if self.use_cot is False:
            return pred
        # Low-signal: short/empty plan or missing commands
        plan = (getattr(pred, 'plan', '') or '').strip()
        commands = (getattr(pred, 'commands', '') or '').strip()
        low_signal = (len(plan.splitlines()) < 2) or (not commands)
        return pred if not low_signal else self.slow(task=task, context=context)
