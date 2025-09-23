from __future__ import annotations

from typing import Optional

import dspy


class PreferenceRewriterSig(dspy.Signature):
    """Rewrite orchestrator input state with policy preferences.

    Inputs: user query, current state/context, compact policy text (from YAML).
    Outputs: adjusted state instructions and suggested tool preferences.
    """

    query: str = dspy.InputField(desc="User's request")
    state: str = dspy.InputField(desc="Current state/context string")
    policy_text: str = dspy.InputField(desc="Compact do/don'ts from policy")

    adjusted_state: str = dspy.OutputField(desc="State with embedded guidance and guardrails")
    prefer_tools: str = dspy.OutputField(desc="Comma-separated tool names to prefer", default="")
    blocked_tools: str = dspy.OutputField(desc="Comma-separated tool names to avoid", default="")
    rationale: str = dspy.OutputField(desc="Why these preferences apply", default="")


class PreferenceRewriter(dspy.Module):
    def __init__(self, use_cot: Optional[bool] = None):
        super().__init__()
        self.fast = dspy.Predict(PreferenceRewriterSig)
        self.slow = dspy.ChainOfThought(PreferenceRewriterSig)
        self.use_cot = use_cot

    def forward(self, query: str, state: str, policy_text: str):
        if self.use_cot is True:
            return self.slow(query=query, state=state, policy_text=policy_text)
        pred = self.fast(query=query, state=state, policy_text=policy_text)
        if self.use_cot is False:
            return pred
        adj = (getattr(pred, 'adjusted_state', '') or '').strip()
        low_signal = len(adj) < 20
        return pred if not low_signal else self.slow(query=query, state=state, policy_text=policy_text)
