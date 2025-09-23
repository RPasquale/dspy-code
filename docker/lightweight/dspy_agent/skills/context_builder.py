from __future__ import annotations

from typing import Optional

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
    rationale: str = dspy.OutputField(desc="Brief reasoning behind the summary", default="")


class ContextBuilder(dspy.Module):
    def __init__(self, use_cot: Optional[bool] = None):
        super().__init__()
        self.fast = dspy.Predict(BuildContextSig)
        self.slow = dspy.ChainOfThought(BuildContextSig)
        self.use_cot = use_cot

    def forward(self, task: str, logs_preview: str):
        if self.use_cot is True:
            pred = self.slow(task=task, logs_preview=logs_preview)
        else:
            pred = self.fast(task=task, logs_preview=logs_preview)
        ctx = getattr(pred, 'context', '') or ''
        kp = getattr(pred, 'key_points', '') or ''
        low_signal = (not ctx.strip()) and (not kp.strip())
        if low_signal and self.use_cot is not True:
            pred = self.slow(task=task, logs_preview=logs_preview)
        return pred
