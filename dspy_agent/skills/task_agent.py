from __future__ import annotations

import dspy


class PlanTaskSig(dspy.Signature):
    """Given a task and context, propose a brief plan and safe commands.

    Prefer non-destructive steps. Include exact shell commands when relevant.
    Keep it concise (<= 10 steps).
    """

    task: str = dspy.InputField()
    context: str = dspy.InputField()

    plan: str = dspy.OutputField(desc="Short, ordered steps to proceed")
    commands: str = dspy.OutputField(desc="Suggested shell commands if any")


class TaskAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.Predict(PlanTaskSig)

    def forward(self, task: str, context: str):
        return self.plan(task=task, context=context)

