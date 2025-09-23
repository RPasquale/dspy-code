from __future__ import annotations

from typing import Optional

import dspy


class PlanTaskSig(dspy.Signature):
    """Given a task and context, propose a short, safe plan and exact commands.

    Constraints:
    - Non-destructive by default; call out any risky steps explicitly.
    - Prefer concrete, copy-pasteable shell commands aligned to the plan.
    - Keep it concise (<= 10 steps) and include assumptions/risks.
    """

    task: str = dspy.InputField()
    context: str = dspy.InputField()

    plan: str = dspy.OutputField(desc="Ordered, concise steps to proceed")
    commands: str = dspy.OutputField(desc="Copy-pasteable shell commands")
    assumptions: str = dspy.OutputField(desc="Assumptions or prerequisites", default="")
    risks: str = dspy.OutputField(desc="Potential risks and mitigations", default="")
    rationale: str = dspy.OutputField(desc="Why this plan/commands", default="")


class TaskAgent(dspy.Module):
    def __init__(self, use_cot: Optional[bool] = None):
        super().__init__()
        self.fast = dspy.Predict(PlanTaskSig)
        self.slow = dspy.ChainOfThought(PlanTaskSig)
        self.use_cot = use_cot

    def forward(self, task: str, context: str):
        if self.use_cot is True:
            return self.slow(task=task, context=context)
        pred = self.fast(task=task, context=context)
        if self.use_cot is False:
            return pred
        plan = (getattr(pred, 'plan', '') or '').strip()
        commands = (getattr(pred, 'commands', '') or '').strip()
        low_signal = (len(plan.splitlines()) < 2) or (not commands)
        return pred if not low_signal else self.slow(task=task, context=context)
