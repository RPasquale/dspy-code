from __future__ import annotations

import dspy


TOOLS = [
    "context", "plan", "grep", "extract", "tree", "ls",
    "codectx", "index", "esearch", "emb-index", "emb-search",
    "open", "watch", "sg", "patch", "diff", "git_status",
    "git_add", "git_commit"
]


class OrchestrateToolSig(dspy.Signature):
    """Choose the best CLI tool and arguments for the user's intent.

    Tools: context, plan, grep, extract, tree, ls, codectx, index, esearch, emb-index, emb-search, open, watch, sg, patch, diff, git_status, git_add, git_commit
    Return JSON in args_json with the arguments for that tool.
    Keep choices safe and non-destructive unless explicitly requested by the user.
    """

    query: str = dspy.InputField(desc="User's natural-language request")
    state: str = dspy.InputField(desc="Short environment summary: workspace, logs, last extract, indexes available")

    tool: str = dspy.OutputField(desc=f"One of: {', '.join(TOOLS)}")
    args_json: str = dspy.OutputField(desc="A JSON object mapping argument names to values")


class Orchestrator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(OrchestrateToolSig)

    def forward(self, query: str, state: str):
        return self.predict(query=query, state=state)

