from __future__ import annotations

import dspy
from typing import Optional


class ControllerSig(dspy.Signature):
    """Chain-of-Thought controller for global agent behavior and routing.

    Inputs:
    - query: user's natural instruction
    - state: compact environment snapshot (workspace, logs summary, code summary, indexes)
    - memory: optional session memory/context
    - preferences: optional policy preferences / guardrails
    - metrics: optional recent performance metrics

    Outputs:
    - plan: high-level plan for this turn
    - mode: performance profile and knobs (e.g., profile=fast|balanced|maxquality; beam_k=N; speculative=true|false; draft_model=name)
    - tool: next tool to execute (safe default if unsure)
    - args_json: JSON object with arguments for the tool (stringified)
    - guardrails: constraints/safety checks to respect this step
    - rationale: brief reasoning for plan + tool choice
    - next_steps: optional suggestions for follow-on steps
    """

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
        # Default to CoT for global control; allow Predict for speed if forced
        self.fast = dspy.Predict(ControllerSig)
        self.slow = dspy.ChainOfThought(ControllerSig)
        self.use_cot = True if use_cot is None else bool(use_cot)

    def forward(
        self,
        query: str,
        state: str,
        memory: str = "",
        preferences: str = "",
        metrics: str = "",
    ):
        if self.use_cot:
            return self.slow(query=query, state=state, memory=memory, preferences=preferences, metrics=metrics)
        return self.fast(query=query, state=state, memory=memory, preferences=preferences, metrics=metrics)

