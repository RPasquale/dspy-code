from __future__ import annotations

from typing import Iterable, Optional

import dspy


class GraphMemorySummarySig(dspy.Signature):
    """Summarize graph-RAG + memory context into actionable verifier targets."""

    query: str = dspy.InputField(desc="User or agent objective driving retrieval")
    vector_focus: str = dspy.InputField(desc="Vector/semantic hit overview", default="")
    document_focus: str = dspy.InputField(desc="Document or keyword matches", default="")
    graph_focus: str = dspy.InputField(desc="Neighbor graph / edge highlights", default="")
    mcts_focus: str = dspy.InputField(desc="Top graph nodes ranked by MCTS", default="")
    memory_focus: str = dspy.InputField(desc="Memory statistics and retrieval metrics", default="")
    fewshot_guidance: str = dspy.InputField(desc="Optional inline few-shot exemplars", default="")

    summary: str = dspy.OutputField(desc="Concise synthesis of the knowledge state")
    recommended_paths: str = dspy.OutputField(desc="Ranked paths/files to inspect next")
    verifier_targets: str = dspy.OutputField(desc="Verifier or reward signals to pursue")
    followups: str = dspy.OutputField(desc="Next retrieval or action suggestions")
    rationale: str = dspy.OutputField(desc="Reasoning that links signals to actions", default="")


class GraphMemorySummary(dspy.Module):
    """LLM helper for Graph-memory fusion with optional few-shot demos."""

    def __init__(self, use_cot: Optional[bool] = None, *, beam_k: int = 1, demos: Optional[Iterable[dspy.Example]] = None) -> None:
        super().__init__()
        self.fast = dspy.Predict(GraphMemorySummarySig)
        self.slow = dspy.ChainOfThought(GraphMemorySummarySig)
        self.use_cot = use_cot
        self.beam_k = max(1, int(beam_k))
        self.demos = list(demos or [])

    def forward(
        self,
        query: str,
        *,
        vector_focus: str = "",
        document_focus: str = "",
        graph_focus: str = "",
        mcts_focus: str = "",
        memory_focus: str = "",
    ):
        fewshot_guidance = self._format_demos()
        predictor = self._select_predictor()
        pred = predictor(
            query=query,
            vector_focus=vector_focus,
            document_focus=document_focus,
            graph_focus=graph_focus,
            mcts_focus=mcts_focus,
            memory_focus=memory_focus,
            fewshot_guidance=fewshot_guidance,
        )
        best = pred
        if self.beam_k > 1:
            best, _ = self._beam_select(
                query,
                vector_focus,
                document_focus,
                graph_focus,
                mcts_focus,
                memory_focus,
                fewshot_guidance,
                predictor,
            )
        return best

    # ------------------------------------------------------------------
    def _format_demos(self) -> str:
        if not self.demos:
            return ""
        rows = []
        for demo in self.demos:
            try:
                rows.append(
                    "Query: {query}\nSummary: {summary}\nPaths: {paths}\nSignals: {signals}\n".format(
                        query=str(getattr(demo, 'query', '') or demo.get('query', '')),
                        summary=str(getattr(demo, 'summary', '') or demo.get('summary', '')),
                        paths=str(getattr(demo, 'recommended_paths', '') or demo.get('recommended_paths', '')),
                        signals=str(getattr(demo, 'verifier_targets', '') or demo.get('verifier_targets', '')),
                    ).strip()
                )
            except Exception:
                continue
        return "\n\n".join(rows)

    def _select_predictor(self):
        if self.use_cot is True:
            return self.slow
        if self.use_cot is False:
            return self.fast
        return self.fast

    def _beam_select(
        self,
        query: str,
        vector_focus: str,
        document_focus: str,
        graph_focus: str,
        mcts_focus: str,
        memory_focus: str,
        fewshot_guidance: str,
        predictor,
    ):
        best = predictor(
            query=query,
            vector_focus=vector_focus,
            document_focus=document_focus,
            graph_focus=graph_focus,
            mcts_focus=mcts_focus,
            memory_focus=memory_focus,
            fewshot_guidance=fewshot_guidance,
        )
        best_score = self._score(best)
        for _ in range(self.beam_k - 1):
            cand = predictor(
                query=query,
                vector_focus=vector_focus,
                document_focus=document_focus,
                graph_focus=graph_focus,
                mcts_focus=mcts_focus,
                memory_focus=memory_focus,
                fewshot_guidance=fewshot_guidance,
            )
            cand_score = self._score(cand)
            if cand_score > best_score:
                best, best_score = cand, cand_score
        return best, best_score

    @staticmethod
    def _score(pred) -> float:
        def _length(attr: str) -> int:
            return len(str(getattr(pred, attr, '')).strip())
        return float(_length('summary') + 0.6 * _length('recommended_paths') + 0.6 * _length('verifier_targets'))


__all__ = [
    'GraphMemorySummarySig',
    'GraphMemorySummary',
]
