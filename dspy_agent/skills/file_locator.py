from __future__ import annotations

from typing import Optional

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
    rationale: str = dspy.OutputField(desc="Brief reasoning behind choices", default="")


class FileLocator(dspy.Module):
    def __init__(self, use_cot: Optional[bool] = None, *, beam_k: int = 1):
        super().__init__()
        self.fast = dspy.Predict(FileLocatorSig)
        self.slow = dspy.ChainOfThought(FileLocatorSig)
        self.use_cot = use_cot
        self.beam_k = max(1, int(beam_k))

    def forward(self, task: str, context: str, code_graph: str = ""):
        if self.use_cot is True:
            return self.slow(task=task, context=context, code_graph=code_graph)
        pred = self.fast(task=task, context=context, code_graph=code_graph)
        if self.use_cot is False:
            return pred
        cands = (getattr(pred, 'file_candidates', '') or '').strip()
        low_signal = not cands or cands in {'[]', '{}'} or 'path' not in cands
        if low_signal:
            pred = self.slow(task=task, context=context, code_graph=code_graph)
        # Beam select best by candidate count and confidence
        def _score(o) -> float:
            txt = (getattr(o, 'file_candidates', '') or '').strip()
            try:
                import json
                arr = json.loads(txt) if txt else []
                n = len(arr) if isinstance(arr, list) else 0
            except Exception:
                n = txt.count('path')
            conf = 0.0
            try:
                conf = float(getattr(o, 'confidence', '0') or '0')
            except Exception:
                conf = 0.0
            return conf * 2.0 + float(n)
        best = pred; best_score = _score(pred)
        if self.beam_k > 1:
            proposer = self.slow if (self.use_cot is True or low_signal) else self.fast
            for _ in range(self.beam_k - 1):
                cand = proposer(task=task, context=context, code_graph=code_graph)
                sc = _score(cand)
                if sc > best_score:
                    best, best_score = cand, sc
        return best
