from __future__ import annotations

import dspy
from typing import Optional


class ControllerSig(dspy.Signature):
    """Chain-of-Thought controller for global agent behavior and routing (lightweight)."""

    query: str = dspy.InputField(desc="User query/instruction")
    state: str = dspy.InputField(desc="Environment snapshot: workspace, logs, code summary, indexes")
    memory: str = dspy.InputField(default="", desc="Optional session memory context")
    preferences: str = dspy.InputField(default="", desc="Optional policy preferences and constraints")
    metrics: str = dspy.InputField(default="", desc="Optional performance metrics snapshot")

    plan: str = dspy.OutputField(desc="Coarse plan for the turn")
    mode: str = dspy.OutputField(desc="Performance profile and knob settings")
    tool: str = dspy.OutputField(desc="Chosen tool (safe default if unsure)")
    args_json: str = dspy.OutputField(desc="JSON dict of tool args as string")
    guardrails: str = dspy.OutputField(desc="Execute-time guardrails and checks")
    rationale: str = dspy.OutputField(desc="Reasoning for plan + tool choice")
    next_steps: str = dspy.OutputField(default="", desc="Optional follow-on suggestions")


class Controller(dspy.Module):
    def __init__(self, use_cot: Optional[bool] = True):
        super().__init__()
        self.fast = dspy.Predict(ControllerSig)
        self.slow = dspy.ChainOfThought(ControllerSig)
        self.use_cot = True if use_cot is None else bool(use_cot)

    def forward(self, query: str, state: str, memory: str = "", preferences: str = "", metrics: str = ""):
        if self.use_cot:
            return self.slow(query=query, state=state, memory=memory, preferences=preferences, metrics=metrics)
        return self.fast(query=query, state=state, memory=memory, preferences=preferences, metrics=metrics)

