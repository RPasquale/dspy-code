from __future__ import annotations

import dspy


class MultiHeadSummarySig(dspy.Signature):
    """
    Summarize multi-head retrieval results into a clear, helpful answer.

    Inputs describe the query and compact overviews of each head.
    Outputs include a human-readable answer plus next steps.
    """

    query = dspy.InputField(desc="original user query")
    vector_overview = dspy.InputField(desc="top semantic matches (vector)")
    document_overview = dspy.InputField(desc="top document hits (keyword)")
    table_overview = dspy.InputField(desc="table rows summary")
    graph_overview = dspy.InputField(desc="graph neighbors/relations summary")

    answer = dspy.OutputField(desc="concise answer to the query")
    summary = dspy.OutputField(desc="short summary of supporting evidence")
    next_steps = dspy.OutputField(desc="concrete next steps or follow-ups")


class SummarizeResults(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m = dspy.Predict(MultiHeadSummarySig)

    def forward(self, query: str, vector_overview: str, document_overview: str, table_overview: str, graph_overview: str):
        return self.m(
            query=query,
            vector_overview=vector_overview,
            document_overview=document_overview,
            table_overview=table_overview,
            graph_overview=graph_overview,
        )

