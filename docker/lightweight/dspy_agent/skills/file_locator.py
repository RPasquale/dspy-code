from __future__ import annotations

import dspy


class FileLocatorSig(dspy.Signature):
    """Identify likely files/modules to modify for a given task and context.

    Return a JSON list of candidate objects with fields: path, score (0-1), reason.
    Keep candidates focused and explain selection criteria in notes.
    """

    task: str = dspy.InputField()
    context: str = dspy.InputField()
    code_graph: str = dspy.InputField(desc="Optional code graph summary", default="")

    file_candidates: str = dspy.OutputField(desc="JSON list: [{path, score, reason}]")
    confidence: str = dspy.OutputField(desc="Confidence 0-1 as string")
    notes: str = dspy.OutputField(desc="Selection criteria and assumptions")


class FileLocator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(FileLocatorSig)

    def forward(self, task: str, context: str, code_graph: str = ""):
        return self.predict(task=task, context=context, code_graph=code_graph)

