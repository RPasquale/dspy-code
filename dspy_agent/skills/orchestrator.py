from __future__ import annotations

import dspy
from types import SimpleNamespace


TOOLS = [
    "context", "plan", "grep", "extract", "tree", "ls",
    "codectx", "index", "esearch", "emb_index", "emb_search", "knowledge", "vretr", "intel",
    "open", "watch", "sg", "patch", "diff", "git_status",
    "git_add", "git_commit"
]


class OrchestrateToolSig(dspy.Signature):
    """Choose the best CLI tool and arguments for the user's intent.

    Tools: context, plan, grep, extract, tree, ls, codectx, index, esearch, emb_index, emb_search, knowledge, vretr, intel, open, watch, sg, patch, diff, git_status, git_add, git_commit
    Return JSON in args_json with the arguments for that tool.
    Keep choices safe and non-destructive unless explicitly requested by the user.
    """

    query: str = dspy.InputField(desc="User's natural-language request")
    state: str = dspy.InputField(desc="Short environment summary: workspace, logs, last extract, indexes available")

    tool: str = dspy.OutputField(desc=f"One of: {', '.join(TOOLS)} (choose conservatively)")
    args_json: str = dspy.OutputField(desc="JSON object of arguments; omit unknown fields")
    rationale: str = dspy.OutputField(desc="Brief reasoning for the tool choice")


class Orchestrator(dspy.Module):
    def __init__(self, use_cot: bool = True):
        super().__init__()
        # Use CoT to justify routing when available
        self.predict = dspy.ChainOfThought(OrchestrateToolSig) if use_cot else dspy.Predict(OrchestrateToolSig)

    def forward(self, query: str, state: str):
        # Predict tool selection with error handling and validation
        try:
            pred = self.predict(query=query, state=state)
        except Exception as e:
            # Sensible default fallback when routing fails
            return SimpleNamespace(
                tool="plan",
                args_json="{}",
                rationale=f"Fallback to 'plan' due to prediction error: {e}"
            )

        tool = (getattr(pred, "tool", None) or "").strip()
        if tool not in TOOLS:
            raise ValueError(f"Predicted tool '{tool}' is not in supported TOOLS: {', '.join(TOOLS)}")

        return pred
