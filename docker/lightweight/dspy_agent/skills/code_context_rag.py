from __future__ import annotations

import dspy


class CodeContextRAGSig(dspy.Signature):
    """Summarize code with retrieval context to surface hot spots and citations.

    Accepts retrieved snippets and references (paths/anchors). Produces a concise
    summary, bullets, hot files/functions relevant to the ask, and a citations field
    that maps insights to references when appropriate.
    """

    snapshot: str = dspy.InputField()
    ask: str = dspy.InputField()
    code_graph: str = dspy.InputField(desc="Optional code graph summary", default="")
    retrieved_snippets: str = dspy.InputField(desc="Concatenated retrieved code/docs", default="")
    references: str = dspy.InputField(desc="JSON list of {path, lines, anchor}", default="")

    summary: str = dspy.OutputField(desc="Concise summary with retrieval context")
    bullets: str = dspy.OutputField(desc="Bulleted key parts and hints")
    entry_points: str = dspy.OutputField(desc="Likely files/functions to touch")
    risk_areas: str = dspy.OutputField(desc="Places changes are risky")
    hot_spots: str = dspy.OutputField(desc="Top hot files/functions tied to ask")
    citations: str = dspy.OutputField(desc="Map insights to references (JSON or text)")


class CodeContextRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CodeContextRAGSig)

    def forward(
        self,
        snapshot: str,
        ask: str = "Summarize this code snapshot.",
        code_graph: str = "",
        retrieved_snippets: str = "",
        references: str = "",
    ):
        return self.predict(
            snapshot=snapshot,
            ask=ask,
            code_graph=code_graph,
            retrieved_snippets=retrieved_snippets,
            references=references,
        )

