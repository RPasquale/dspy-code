from __future__ import annotations

import dspy


class BuildContextSig(dspy.Signature):
    """Summarize logs into actionable context for a coding task.

    Extract key errors, likely causes, and concrete hints. Keep summary tight, then
    provide bullets and any missing info that would unblock next steps.
    """

    task: str = dspy.InputField()
    logs_preview: str = dspy.InputField()

    context: str = dspy.OutputField(desc="Concise context and key findings")
    key_points: str = dspy.OutputField(desc="Bulleted list of errors and clues")
    missing_info: str = dspy.OutputField(desc="What info would help next", default="")
    next_steps: str = dspy.OutputField(desc="Immediate next steps", default="")


class ContextBuilder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(BuildContextSig)

    def forward(self, task: str, logs_preview: str):
        return self.predict(task=task, logs_preview=logs_preview)
