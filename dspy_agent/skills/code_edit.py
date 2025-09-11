from __future__ import annotations

import dspy


class CodeEditSig(dspy.Signature):
    """Propose a minimal, safe code change as a unified diff.

    Inputs include the task, relevant context (logs/errors), optional code graph summary,
    and optional file hints. Output a unified diff patch and a brief rationale.
    """

    task: str = dspy.InputField()
    context: str = dspy.InputField()
    code_graph: str = dspy.InputField(desc="Optional code graph summary", default="")
    file_hints: str = dspy.InputField(desc="Optional hints: files/modules to touch", default="")

    patch: str = dspy.OutputField(desc="Unified diff patch (git-style)")
    rationale: str = dspy.OutputField(desc="Brief reasoning and safety checks")


class CodeEdit(dspy.Module):
    def __init__(self, use_cot: bool = True):
        super().__init__()
        # Prefer Chain-of-Thought for code edits; fallback to Predict when unavailable
        self.predict = dspy.ChainOfThought(CodeEditSig) if use_cot else dspy.Predict(CodeEditSig)

    def forward(self, task: str, context: str, code_graph: str = "", file_hints: str = ""):
        return self.predict(task=task, context=context, code_graph=code_graph, file_hints=file_hints)

