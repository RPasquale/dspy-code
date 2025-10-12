from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence

import dspy

from ..agentic import load_retrieval_events, summarize_graph_memory
from ..context.context_manager import ContextManager
from ..rl.rlkit import AgentResult, RewardConfig, aggregate_reward, get_verifiers
from ..signatures.graph_memory import GraphMemorySummary, GraphMemorySummarySig
from .data_rag import DataRAG, DataRAGResult, _overview_document, _overview_graph, _overview_table, _overview_vector

DEFAULT_GRAPH_MEMORY_DEMOS: List[Dict[str, Any]] = [
    {
        'query': 'Investigate flaky pipeline tests',
        'summary': 'Focus on pipeline orchestration modules and recent flaky test logs.',
        'recommended_paths': 'tests/pipeline/test_pipeline_runner.py\nci/pipeline.yml',
        'verifier_targets': 'graph_mcts_alignment, memory_precision',
        'followups': 'Run pytest -k pipeline --maxfail=1',
    },
    {
        'query': 'Boost retrieval coverage around orchestrator',
        'summary': 'Review orchestrator runtime and refresh retrieval cache hits.',
        'recommended_paths': 'dspy_agent/agents/orchestrator_runtime.py\nlogs/orchestrator/*.jsonl',
        'verifier_targets': 'graph_signal, graph_prefetch, memory_coverage',
        'followups': 'Run mcts refresh and sync vector index',
    },
]


@dataclass
class GraphMemoryReport:
    query: str
    rag: DataRAGResult
    fusion_score: float
    metrics: Dict[str, float]
    reward: float
    reward_breakdown: Dict[str, float]
    summary: str
    recommended_paths: List[str]
    verifier_targets: List[str]
    followups: List[str]
    rationale: str
    context: Dict[str, Any]
    namespace: str
    generated_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'namespace': self.namespace,
            'query': self.query,
            'fusion_score': self.fusion_score,
            'metrics': self.metrics,
            'reward': self.reward,
            'reward_breakdown': self.reward_breakdown,
            'summary': self.summary,
            'recommended_paths': self.recommended_paths,
            'verifier_targets': self.verifier_targets,
            'followups': self.followups,
            'rationale': self.rationale,
            'rag': {
                'mode': self.rag.mode,
                'answer': self.rag.answer,
                'context': self.rag.context,
            },
            'context': self.context,
            'generated_at': self.generated_at,
        }


class GraphMemoryExplorer:
    """Coordinator that fuses DataRAG, memory stats, and verifier scoring."""

    def __init__(
        self,
        *,
        namespace: str = "default",
        workspace: Optional[str] = None,
        top_k: int = 8,
        reward_weights: Optional[Mapping[str, float]] = None,
        summary_module: Optional[GraphMemorySummary] = None,
    ) -> None:
        self.namespace = namespace
        self.workspace = Path(workspace or Path.cwd())
        self.top_k = max(1, int(top_k))
        self.data_rag = DataRAG(namespace=namespace, workspace=str(self.workspace))
        if summary_module is not None:
            self.summary_module = summary_module
        else:
            demos = load_graph_memory_demos()
            self.summary_module = GraphMemorySummary(use_cot=None, beam_k=2, demos=demos or None)
        self.reward_weights = dict(reward_weights or {
            'quality_vec': 0.6,
            'quality_doc': 0.4,
            'quality_graph': 0.8,
            'quality_graph_prefetch': 0.6,
            'quality_mcts_alignment': 1.0,
            'graph_mcts_priority': 0.6,
            'graph_seed_ratio': 0.3,
            'memory_precision': 0.5,
            'memory_coverage': 0.4,
            'memory_query_rate': 0.3,
        })

    def build_report(self, query: str, *, collection: Optional[str] = None) -> GraphMemoryReport:
        rag_result = self.data_rag(query, top_k=self.top_k, collection=collection, use_lm=False)
        fusion = rag_result.context.get('fusion', {})
        fusion_features = fusion.get('features', {})
        graph_prefetch = rag_result.context.get('graph_prefetch', {})
        mcts_top = rag_result.context.get('mcts_top', [])
        graph_metrics = rag_result.context.get('graph_metrics', {})
        vector_focus = _overview_vector(rag_result.context.get('vector', {}))
        document_focus = _overview_document(rag_result.context.get('document', {}))
        table_focus = _overview_table(rag_result.context.get('table', {}))
        graph_focus = _overview_graph(rag_result.context.get('graph', {}))
        mcts_focus = self._format_mcts(mcts_top)
        memory_focus, memory_metrics = self._memory_focus()
        merged_metrics = self._merge_metrics(fusion_features, graph_metrics, memory_metrics)
        reward, reward_breakdown = self._score_metrics(query, merged_metrics)
        summary_payload = self._summarize(
            query,
            vector_focus,
            document_focus,
            graph_focus,
            mcts_focus,
            memory_focus,
        )
        report_context = {
            'vector_focus': vector_focus,
            'document_focus': document_focus,
            'table_focus': table_focus,
            'graph_focus': graph_focus,
            'mcts_focus': mcts_focus,
            'memory_focus': memory_focus,
            'graph_prefetch': graph_prefetch,
            'graph_metrics': graph_metrics,
            'mcts_top': mcts_top,
            'fusion': fusion,
        }
        paths = _split_lines(summary_payload.recommended_paths)
        verifier_targets = _split_lines(summary_payload.verifier_targets)
        followups = _split_lines(summary_payload.followups)
        return GraphMemoryReport(
            query=query,
            rag=rag_result,
            fusion_score=float(fusion.get('score', 0.0)),
            metrics=merged_metrics,
            reward=reward,
            reward_breakdown=reward_breakdown,
            summary=summary_payload.summary,
            recommended_paths=paths,
            verifier_targets=verifier_targets,
            followups=followups,
            rationale=getattr(summary_payload, 'rationale', ''),
            context=report_context,
            namespace=self.namespace,
            generated_at=time.time(),
        )

    # ------------------------------------------------------------------
    def _summarize(
        self,
        query: str,
        vector_focus: str,
        document_focus: str,
        graph_focus: str,
        mcts_focus: str,
        memory_focus: str,
    ) -> GraphMemorySummarySig:
        try:
            return self.summary_module(
                query=query,
                vector_focus=vector_focus,
                document_focus=document_focus,
                graph_focus=graph_focus,
                mcts_focus=mcts_focus,
                memory_focus=memory_focus,
            )
        except Exception:
            return SimpleNamespace(
                summary=f"Graph memory summary for '{query}'",
                recommended_paths=graph_focus,
                verifier_targets=mcts_focus,
                followups=memory_focus,
                rationale="LLM summary unavailable",
            )

    def _memory_focus(self) -> tuple[str, Dict[str, float]]:
        ctx = ContextManager(self.workspace, self.workspace / 'logs')
        feats = ctx.agentic_features(max_items=12)
        retrievals = load_retrieval_events(self.workspace, limit=8)
        focus_lines = [
            f"KG nodes: {feats[0]:.0f}",
            f"KG edges: {feats[1]:.0f}",
            f"Precision: {feats[4]:.2f}",
            f"Coverage: {feats[5]:.0f}",
            f"Avg score: {feats[6]:.2f}",
            f"Queries: {feats[7]:.0f}",
        ]
        for ev in retrievals[:3]:
            query = str(ev.get('query', ''))
            hits = len(ev.get('hits', []) or [])
            focus_lines.append(f"Retrieval: {query[:60]} ({hits} hits)")
        try:
            gm_summary = summarize_graph_memory(self.workspace, limit=5)
        except Exception:
            gm_summary = {}
        top_files = gm_summary.get('top_files') if isinstance(gm_summary, dict) else None
        if top_files:
            for entry in top_files[:3]:
                focus_lines.append(f"Graph focus: {entry.get('path','')} ({float(entry.get('confidence',0.0)):.2f})")
        metrics = {
            'memory_precision': float(feats[4]) if len(feats) > 4 else 0.0,
            'memory_coverage': min(1.0, float(feats[5]) / 50.0) if len(feats) > 5 else 0.0,
            'memory_avg_score': float(feats[6]) if len(feats) > 6 else 0.0,
            'memory_query_rate': min(1.0, float(feats[7]) / 10.0) if len(feats) > 7 else 0.0,
        }
        return "\n".join(focus_lines), metrics

    def _score_metrics(self, query: str, metrics: Mapping[str, float]) -> tuple[float, Dict[str, float]]:
        verifiers = get_verifiers()
        reward_cfg = RewardConfig(weights=self.reward_weights, penalty_kinds=[], scales={})
        total, _, detail = aggregate_reward(
            AgentResult(metrics=metrics, info={'signature_name': 'graph_memory', 'query': query}),
            verifiers,
            reward_cfg,
        )
        return float(total), detail

    def _merge_metrics(
        self,
        fusion_features: Mapping[str, Any],
        graph_metrics: Mapping[str, Any],
        memory_metrics: Mapping[str, float],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for key, value in fusion_features.items():
            try:
                metrics[str(key)] = float(value)
            except Exception:
                continue
        for key, value in graph_metrics.items():
            norm_key = f"graph_{key}"
            try:
                metrics[norm_key] = float(value)
            except Exception:
                continue
        metrics.update(memory_metrics)
        return metrics

    @staticmethod
    def _format_mcts(nodes: Sequence[Mapping[str, Any]]) -> str:
        if not nodes:
            return "(no mcts priorities)"
        lines = []
        for idx, node in enumerate(nodes[:10]):
            ident = str(node.get('relative_path') or node.get('id') or node.get('path') or '?')
            priority = float(node.get('priority', 0.0))
            rank = f"#{idx + 1}".rjust(3)
            lines.append(f"{rank} {ident} (p={priority:.3f})")
        return "\n".join(lines)


def _split_lines(value: str) -> List[str]:
    parts = []
    for line in (value or "").splitlines():
        cleaned = line.strip().lstrip('-').strip()
        if cleaned:
            parts.append(cleaned)
    return parts


__all__ = [
    'GraphMemoryExplorer',
    'GraphMemoryReport',
    'load_graph_memory_demos',
]


def load_graph_memory_demos(path: Optional[Path] = None) -> List:
    """Load optional few-shot demos for the graph memory signature."""

    demo_path = path
    items: List[Dict[str, Any]] = []
    if demo_path is None:
        demo_path = Path(__file__).resolve().parents[1] / 'resources' / 'graph_memory_demos.jsonl'
    try:
        if demo_path.exists():
            for line in demo_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                items.append(payload)
    except Exception:
        items = []
    if not items:
        items = list(DEFAULT_GRAPH_MEMORY_DEMOS)
    demos: List[Any] = []
    for payload in items:
        try:
            demos.append(dspy.Example(**payload))
        except Exception:
            demos.append(payload)
    return demos
