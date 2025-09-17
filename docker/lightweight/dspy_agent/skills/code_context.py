from __future__ import annotations

import dspy


class CodeContextSig(dspy.Signature):
    """Summarize a code snapshot for understanding and planning.

    Return a tight summary and bullets listing components, important functions/classes,
    likely entry points, and risky areas relevant to the ask.
    """

    snapshot: str = dspy.InputField()
    ask: str = dspy.InputField()
    code_graph: str = dspy.InputField(desc="Optional code graph summary", default="")

    summary: str = dspy.OutputField(desc="Concise codebase summary")
    bullets: str = dspy.OutputField(desc="Bulleted key parts, files, and hints")
    entry_points: str = dspy.OutputField(desc="Likely files/functions to touch", default="")
    risk_areas: str = dspy.OutputField(desc="Places changes can be risky", default="")


class CodeContext(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CodeContextSig)

    def forward(self, snapshot: str, ask: str = "Summarize this code snapshot.", code_graph: str = ""):
        return self.predict(snapshot=snapshot, ask=ask, code_graph=code_graph)
