from __future__ import annotations

from typing import Optional

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
    rationale: str = dspy.OutputField(desc="Brief reasoning behind selections", default="")


class CodeContextRAG(dspy.Module):
    def __init__(self, use_cot: Optional[bool] = None, *, beam_k: int = 1):
        super().__init__()
        self.fast = dspy.Predict(CodeContextRAGSig)
        self.slow = dspy.ChainOfThought(CodeContextRAGSig)
        self.use_cot = use_cot
        self.beam_k = max(1, int(beam_k))

    def forward(
        self,
        snapshot: str,
        ask: str = "Summarize this code snapshot.",
        code_graph: str = "",
        retrieved_snippets: str = "",
        references: str = "",
    ):
        if self.use_cot is True:
            return self.slow(
                snapshot=snapshot,
                ask=ask,
                code_graph=code_graph,
                retrieved_snippets=retrieved_snippets,
                references=references,
            )
        pred = self.fast(
            snapshot=snapshot,
            ask=ask,
            code_graph=code_graph,
            retrieved_snippets=retrieved_snippets,
            references=references,
        )
        if self.use_cot is False:
            return pred
        summary = (getattr(pred, 'summary', '') or '').strip()
        bullets = (getattr(pred, 'bullets', '') or '').strip()
        hot = (getattr(pred, 'hot_spots', '') or '').strip()
        low_signal = (len(summary) < 40 and len(bullets) < 20) or not hot
        if low_signal:
            pred = self.slow(
                snapshot=snapshot,
                ask=ask,
                code_graph=code_graph,
                retrieved_snippets=retrieved_snippets,
                references=references,
            )
        # Beam select
        def _score(o) -> float:
            s = len(((getattr(o, 'summary', '') or '')).strip())
            hs = len(((getattr(o, 'hot_spots', '') or '')).strip())
            ep = len(((getattr(o, 'entry_points', '') or '')).strip())
            return s + 0.7 * hs + 0.5 * ep
        best = pred; best_score = _score(pred)
        if self.beam_k > 1:
            proposer = self.slow if (self.use_cot is True or low_signal) else self.fast
            for _ in range(self.beam_k - 1):
                cand = proposer(
                    snapshot=snapshot,
                    ask=ask,
                    code_graph=code_graph,
                    retrieved_snippets=retrieved_snippets,
                    references=references,
                )
                sc = _score(cand)
                if sc > best_score:
                    best, best_score = cand, sc
        return best
