from __future__ import annotations

import dspy


class BuildContextSig(dspy.Signature):
    """Summarize logs into actionable context for a coding task.

    Given logs and a task, extract key errors, likely causes, and relevant hints.
    Output a concise context paragraph plus bullet points.
    """

    task: str = dspy.InputField()
    logs_preview: str = dspy.InputField()

    context: str = dspy.OutputField(desc="Concise context and key findings")
    key_points: str = dspy.OutputField(desc="Bulleted list of errors and clues")


class ContextBuilder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(BuildContextSig)

    def forward(self, task: str, logs_preview: str):
        return self.predict(task=task, logs_preview=logs_preview)

