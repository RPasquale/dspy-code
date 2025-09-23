from __future__ import annotations

from typing import Optional

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
    rationale: str = dspy.OutputField(desc="Brief reasoning behind the summary", default="")


class CodeContext(dspy.Module):
    def __init__(self, use_cot: Optional[bool] = None, *, beam_k: int = 1):
        super().__init__()
        # Light gating: fast Predict first; escalate to CoT if low-signal
        self.fast = dspy.Predict(CodeContextSig)
        self.slow = dspy.ChainOfThought(CodeContextSig)
        self.use_cot = use_cot
        self.beam_k = max(1, int(beam_k))

    def forward(self, snapshot: str, ask: str = "Summarize this code snapshot.", code_graph: str = ""):
        if self.use_cot is True:
            return self.slow(snapshot=snapshot, ask=ask, code_graph=code_graph)
        # Try fast path
        pred = self.fast(snapshot=snapshot, ask=ask, code_graph=code_graph)
        if self.use_cot is False:
            return pred
        # Escalate when low-signal
        summary = (getattr(pred, 'summary', '') or '').strip()
        bullets = (getattr(pred, 'bullets', '') or '').strip()
        entry = (getattr(pred, 'entry_points', '') or '').strip()
        risk = (getattr(pred, 'risk_areas', '') or '').strip()
        low_signal = (len(summary) < 40 and len(bullets) < 20) or (not entry and not risk)
        if low_signal:
            pred = self.slow(snapshot=snapshot, ask=ask, code_graph=code_graph)

        # Beam select best of K
        def _score(o) -> float:
            s = len(((getattr(o, 'summary', '') or '')).strip())
            ep = len(((getattr(o, 'entry_points', '') or '')).strip())
            rk = len(((getattr(o, 'risk_areas', '') or '')).strip())
            return s + 0.5 * ep + 0.5 * rk
        best = pred; best_score = _score(pred)
        if self.beam_k > 1:
            proposer = self.slow if (self.use_cot is True or low_signal) else self.fast
            for _ in range(self.beam_k - 1):
                cand = proposer(snapshot=snapshot, ask=ask, code_graph=code_graph)
                sc = _score(cand)
                if sc > best_score:
                    best, best_score = cand, sc
        return best
