from __future__ import annotations

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


class TaskAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.Predict(PlanTaskSig)

    def forward(self, task: str, context: str):
        return self.plan(task=task, context=context)
