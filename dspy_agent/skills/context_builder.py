from __future__ import annotations

import dspy
import re
from typing import Tuple, Optional


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
        try:
            if self.use_cot is True:
                pred = self.slow(task=task, logs_preview=logs_preview)
            else:
                pred = self.fast(task=task, logs_preview=logs_preview)
            # Validate outputs; if empty, synthesize a safe fallback.
            ctx = getattr(pred, 'context', '') or ''
            kp = getattr(pred, 'key_points', '') or ''
            mi = getattr(pred, 'missing_info', '') or ''
            ns = getattr(pred, 'next_steps', '') or ''
            low_signal = (not ctx.strip()) and (not kp.strip())
            if low_signal and self.use_cot is not True:
                # Escalate to CoT once
                pred = self.slow(task=task, logs_preview=logs_preview)
                ctx = getattr(pred, 'context', '') or ''
                kp = getattr(pred, 'key_points', '') or ''
                mi = getattr(pred, 'missing_info', '') or ''
                ns = getattr(pred, 'next_steps', '') or ''
                low_signal = (not ctx.strip()) and (not kp.strip())
            if low_signal:
                f_ctx, f_kp, f_mi, f_ns = self._fallback(task, logs_preview)
                return type(pred)(context=f_ctx, key_points=f_kp, missing_info=f_mi, next_steps=f_ns)
            return pred
        except Exception:
            f_ctx, f_kp, f_mi, f_ns = self._fallback(task, logs_preview)
            # Return a simple namespace-compatible object with expected fields.
            class _Fallback:
                def __init__(self, context: str, key_points: str, missing_info: str, next_steps: str):
                    self.context = context
                    self.key_points = key_points
                    self.missing_info = missing_info
                    self.next_steps = next_steps
                    self.rationale = ""
            return _Fallback(f_ctx, f_kp, f_mi, f_ns)

    # Heuristic fallback builder -----------------------------------------
    def _fallback(self, task: str, logs: str) -> Tuple[str, str, str, str]:
        text = logs or ''
        # Extract last error-like lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        tail = lines[-50:]
        errors = [ln for ln in tail if re.search(r"(error|exception|traceback|failed|timeout)", ln, re.I)]
        hints = []
        for ln in errors[:5]:
            hints.append(f"- {ln}")
        if not hints and tail:
            hints = [f"- {tail[-1]}"]
        key_points = "\n".join(hints) if hints else "- No explicit errors found in recent logs"
        context = f"Task: {task}\nRecent symptoms suggest failures around the lines below. Investigate the first occurrence and neighboring code."
        missing = "- Need exact stack trace and offending file paths\n- Share last commands run and their outputs"
        next_steps = (
            "- Reproduce the error locally with the same command\n"
            "- Open the referenced files and inspect around failing lines\n"
            "- Add minimal logging or assertions where crash occurs\n"
            "- Rerun tests to confirm the fix"
        )
        return context, key_points, missing, next_steps
