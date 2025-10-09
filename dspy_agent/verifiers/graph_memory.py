from __future__ import annotations

from typing import Mapping

from ..rl.rlkit import AgentResult, VerifierProtocol


def _metric(result: AgentResult, *names: str) -> float:
    metrics = result.metrics
    for name in names:
        if name in metrics:
            try:
                return float(metrics.get(name, 0.0) or 0.0)
            except Exception:
                continue
    return 0.0


class GraphSignalVerifier:
    kind = "graph_signal"

    def __call__(self, result: AgentResult) -> float:
        return _metric(result, 'quality_graph', 'graph')


class GraphPrefetchVerifier:
    kind = "graph_prefetch"

    def __call__(self, result: AgentResult) -> float:
        return _metric(result, 'quality_graph_prefetch', 'graph_prefetch')


class GraphMctsAlignmentVerifier:
    kind = "graph_mcts_alignment"

    def __call__(self, result: AgentResult) -> float:
        score = _metric(result, 'quality_mcts_alignment', 'graph_mcts_overlap')
        priority = _metric(result, 'graph_mcts_priority')
        return max(0.0, min(1.0, 0.7 * score + 0.3 * priority))


class MemoryPrecisionVerifier:
    kind = "memory_precision"

    def __call__(self, result: AgentResult) -> float:
        return _metric(result, 'memory_precision')


class MemoryCoverageVerifier:
    kind = "memory_coverage"

    def __call__(self, result: AgentResult) -> float:
        coverage = _metric(result, 'memory_coverage')
        query_rate = _metric(result, 'memory_query_rate')
        return max(0.0, min(1.0, 0.6 * coverage + 0.4 * query_rate))


def get_graph_memory_verifiers() -> list[VerifierProtocol]:
    return [
        GraphSignalVerifier(),
        GraphPrefetchVerifier(),
        GraphMctsAlignmentVerifier(),
        MemoryPrecisionVerifier(),
        MemoryCoverageVerifier(),
    ]


__all__ = [
    'GraphSignalVerifier',
    'GraphPrefetchVerifier',
    'GraphMctsAlignmentVerifier',
    'MemoryPrecisionVerifier',
    'MemoryCoverageVerifier',
    'get_graph_memory_verifiers',
]
