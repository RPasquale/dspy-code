from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
import re
import time
from pathlib import Path

from ..db.redb_router import RedDBRouter, QueryRequest
from ..dbkit import RedDBStorage
from ..rl.rlkit import AgentResult, RewardConfig, aggregate_reward, get_verifiers
from ..rl.rl_helpers import load_effective_rl_config_dict
from ..llm import configure_lm, temporary_lm
from ..signatures.data_summary import SummarizeResults
from ..agents.knowledge import build_code_graph
from ..graph.metrics import collect_edge_metrics, merge_edge_metrics
from ..graph.node_metadata import build_node_metadata, compute_embedding
from ..graph.analytics import pagerank, connected_components, mixed_language_neighbors


@dataclass
class DataRAGResult:
    mode: str
    answer: str
    context: Dict[str, Any]


class DataRAG:
    """Modular RAG over vector/doc/table/graph using RedDBRouter.

    Heuristics:
    - Try vector first for semantic matches; fallback to doc when empty.
    - Table mode when SELECT/WHERE patterns or table= provided.
    - Graph mode when 'neighbor' or graph param provided.
    Returns a compact answer with raw context for agent post-processing.
    """

    def __init__(self, namespace: str = "default", workspace: Optional[str] = None) -> None:
        st = RedDBStorage(url=os.getenv("REDDB_URL", None), namespace=os.getenv("REDDB_NAMESPACE", namespace))
        self.router = RedDBRouter(storage=st)
        self.namespace = namespace
        self.workspace = workspace or os.getenv("DSPY_WORKSPACE") or os.getcwd()
        self._graph_ready = False

    def __call__(self, text: str, *, top_k: int = 5, collection: Optional[str] = None, use_lm: bool = True) -> DataRAGResult:
        # Run multi-head retrieval with learned fusion
        fused, ctx = self._multi_head(text=text, top_k=top_k, collection=collection)
        # Optional DSPy/Ollama summarization
        answer = None
        if use_lm:
            lm = configure_lm(provider="ollama", model_name=os.getenv("OLLAMA_MODEL", None))
            if lm is not None:
                try:
                    vec_ov = _overview_vector(ctx.get('vector') or {})
                    doc_ov = _overview_document(ctx.get('document') or {})
                    tab_ov = _overview_table(ctx.get('table') or {})
                    gra_ov = _overview_graph(ctx.get('graph') or {})
                    with temporary_lm(lm):
                        s = SummarizeResults()
                        pred = s(query=text, vector_overview=vec_ov, document_overview=doc_ov, table_overview=tab_ov, graph_overview=gra_ov)
                    answer = (getattr(pred, 'answer', None) or None)
                    summary = (getattr(pred, 'summary', None) or None)
                    next_steps = (getattr(pred, 'next_steps', None) or None)
                    if answer:
                        readable = answer
                        if summary:
                            readable += "\n\nSummary:\n" + summary
                        if next_steps:
                            readable += "\n\nNext Steps:\n" + next_steps
                        return DataRAGResult(mode="multi", answer=readable, context={"fusion": fused, **ctx})  # type: ignore[arg-type]
                except Exception:
                    pass
        # Fallback non-LLM summary
        readable = _summarize("multi", ctx)
        return DataRAGResult(mode="multi", answer=readable, context={"fusion": fused, **ctx})  # type: ignore[arg-type]

    # ---------------
    # Multi-head retrieval and learned fusion
    # ---------------
    def _multi_head(self, *, text: str, top_k: int, collection: Optional[str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        ns = self.namespace
        coll = collection or "default"
        # Simple graph query inference (neighbors of X)
        gq = None
        tl = (text or "").lower().strip()
        if "neighbor" in tl and " of " in tl:
            try:
                node_id = tl.split(" of ", 1)[1].split()[0]
                gq = {"neighbors": {"node_id": node_id, "k": max(1, int(top_k))}}
            except Exception:
                gq = None

        def _vector():
            q = QueryRequest(mode="vector", namespace=ns, text=text, collection=coll, top_k=top_k)
            return self.router.route_query(q)

        def _doc():
            q = QueryRequest(mode="document", namespace=ns, text=text, collection=coll, top_k=top_k)
            return self.router.route_query(q)

        def _table():
            # Heuristic: attempt where={text:<full>} only if looks like key:value pairs are absent
            where = None
            return self.router.route_query(QueryRequest(mode="table", namespace=ns, table="default", where=where, top_k=max(50, top_k)))

        def _graph():
            if gq is None:
                return {"mode": "graph", "neighbors": {"edges": []}}
            self._ensure_graph_ready()
            return self.router.route_query(QueryRequest(mode="graph", namespace=ns, graph=gq))

        with ThreadPoolExecutor(max_workers=4) as ex:
            fv = ex.submit(_vector)
            fd = ex.submit(_doc)
            ft = ex.submit(_table)
            fg = ex.submit(_graph)
            vec = fv.result() or {}
            doc = fd.result() or {}
            tab = ft.result() or {}
            gra = fg.result() or {}

        graph_prefetch = self._graph_prefetch(text, doc)
        f_prefetch = min(len(graph_prefetch.get('edges', [])) / 20.0, 1.0)
        mcts_top = self._mcts_top(namespace=ns, limit=max(8, top_k * 2))
        mcts_index: Dict[str, Dict[str, Any]] = {}
        for node in mcts_top:
            aliases = {
                str(node.get('id', '') or ''),
                str(node.get('relative_path', '') or ''),
                str(node.get('path', '') or ''),
            }
            for alias in aliases:
                normalized = alias.replace('\\', '/').lstrip('./')
                if normalized:
                    mcts_index[normalized] = node

        # Features
        vec_hits = vec.get("hits") or []
        vec_max = float(vec_hits[0].get("score", 0.0) if vec_hits and isinstance(vec_hits[0], dict) and "score" in vec_hits[0] else 0.0)
        if not vec_max and vec_hits and isinstance(vec_hits[0], dict) and "v" in vec_hits[0]:
            # Cosine not in payload; approximate from presence
            vec_max = 1.0 if vec_hits else 0.0
        doc_hits = doc.get("hits") or []
        tab_rows = tab.get("rows") or []
        gra_edges = (gra.get("neighbors") or {}).get("edges", [])
        # Normalize
        def _norm(x: float, cap: float) -> float:
            return min(max(0.0, x), cap) / (cap or 1.0)
        f_vec = max(0.0, min(1.0, vec_max))
        f_doc = _norm(float(len(doc_hits)), 10)
        f_tab = _norm(float(len(tab_rows)), 50)
        f_gra = _norm(float(len(gra_edges)), 20)

        seeds = set(graph_prefetch.get('seeds', []) or [])
        normalized_seeds = {str(seed).replace('\\', '/').lstrip('./') for seed in seeds}
        overlap = normalized_seeds & set(mcts_index.keys())
        seed_count = float(len(normalized_seeds))
        overlap_ratio = (len(overlap) / seed_count) if seed_count else 0.0
        overlap_priorities = [float(mcts_index[s].get('priority', 0.0)) for s in overlap]
        max_priority = max((float(node.get('priority', 0.0)) for node in mcts_top), default=0.0)
        avg_overlap_priority = sum(overlap_priorities) / float(len(overlap_priorities)) if overlap_priorities else 0.0
        priority_norm = 0.0
        if max_priority > 0.0:
            priority_norm = max(0.0, min(1.0, avg_overlap_priority / max_priority))
        graph_seed_norm = _norm(float(len(normalized_seeds)), 30)

        # Learned fusion via RL verifiers (weights from workspace if available)
        weights = {}
        try:
            cfg = load_effective_rl_config_dict(Path(self.workspace))  # type: ignore[name-defined]
            if isinstance(cfg, dict):
                weights = dict(cfg.get('weights') or {})
        except Exception:
            weights = {}

        verifiers = get_verifiers()
        rc = RewardConfig(weights=weights, penalty_kinds=[], scales={})
        metrics = {
            'pass_rate': f_vec,  # treat semantic similarity as success proxy
            'quality_vec': f_vec,
            'quality_doc': f_doc,
            'quality_table': f_tab,
            'quality_graph': f_gra,
            'quality_graph_prefetch': f_prefetch,
            'quality_mcts_alignment': overlap_ratio,
            'quality_policy': 1.0,
            'blast_radius': 0.0,
            'graph_mcts_overlap': overlap_ratio,
            'graph_mcts_priority': priority_norm,
            'graph_seed_ratio': graph_seed_norm,
            'graph_neighbors_ratio': f_gra,
        }
        total, _, details = aggregate_reward(AgentResult(metrics=metrics, info={}), verifiers, rc)
        fusion = {
            'score': float(total),
            'features': {
                'vec': f_vec,
                'doc': f_doc,
                'table': f_tab,
                'graph': f_gra,
                'graph_prefetch': f_prefetch,
                'graph_mcts_alignment': overlap_ratio,
                'graph_mcts_priority': priority_norm,
                'graph_seed_ratio': graph_seed_norm,
            },
            'verifier_details': details,
        }
        ctx = {
            'vector': vec,
            'document': doc,
            'table': tab,
            'graph': gra,
            'graph_prefetch': graph_prefetch,
            'mcts_top': mcts_top,
            'graph_metrics': {
                'seeds': sorted(normalized_seeds),
                'overlap': sorted(overlap),
                'overlap_priority': avg_overlap_priority,
                'overlap_ratio': overlap_ratio,
            },
        }
        return fusion, ctx

    def _mcts_top(self, namespace: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            labels = self.router.st.get(f"{namespace}:graph:_node_labels") or []
        except Exception:
            return []
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for label in labels:
            try:
                idx_key = self.router._k(namespace, 'collection', f'graph::node::{label}', '_ids')
                ids = self.router.st.get(idx_key) or []
            except Exception:
                continue
            for node_id in ids:
                try:
                    rec = self.router.st.get(self.router._k(namespace, 'graph', 'node', label, node_id)) or {}
                except Exception:
                    continue
                priority = rec.get('mcts_priority')
                if priority is None:
                    continue
                scored.append((float(priority), {
                    'id': node_id,
                    'label': label,
                    'priority': float(priority),
                    'relative_path': rec.get('relative_path', node_id),
                    'path': rec.get('path', rec.get('relative_path', node_id)),
                    'language': rec.get('language'),
                    'pagerank': rec.get('pagerank'),
                    'mcts_updated_at': rec.get('mcts_updated_at'),
                }))
        if not scored:
            return []
        scored.sort(key=lambda item: item[0], reverse=True)
        top_nodes = [entry for _, entry in scored[: max(1, limit)]]
        return top_nodes

    def _graph_prefetch(self, text: str, doc_ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_graph_ready()
        seeds: set[str] = set()
        for match in FILE_TOKEN_RE.findall(text or ''):
            node = self._resolve_node_id(match)
            if node:
                seeds.add(node)

        for hit in doc_ctx.get('hits') or []:
            if isinstance(hit, dict):
                for key in ('path', 'relative_path'):
                    val = hit.get(key) or hit.get('payload', {}).get(key)
                    if isinstance(val, str):
                        node = self._resolve_node_id(val)
                        if node:
                            seeds.add(node)

        seeds = set(sorted(seeds))
        neighbors_map: Dict[str, Any] = {}
        edge_set: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for seed in seeds:
            try:
                nb = self.router.neighbors(self.namespace, seed, limit=15)
            except Exception:
                nb = {}
            neighbors_map[seed] = nb
            for edge in nb.get('edges', []) or []:
                key = (edge.get('src'), edge.get('dst'), edge.get('label'))
                if key not in edge_set:
                    edge_set[key] = edge

        path_results: List[Dict[str, Any]] = []
        seed_list = list(seeds)
        for i in range(len(seed_list) - 1):
            src = seed_list[i]
            dst = seed_list[i + 1]
            try:
                path = self.router.shortest_path(self.namespace, src, dst)
                if path:
                    path_results.append({'src': src, 'dst': dst, **path})
                    for edge in path.get('edges', []) or []:
                        key = (edge.get('src'), edge.get('dst'), edge.get('label'))
                        if key not in edge_set:
                            edge_set[key] = edge
            except Exception:
                continue

        return {
            'seeds': seed_list,
            'neighbors': neighbors_map,
            'edges': list(edge_set.values()),
            'paths': path_results,
        }

    def _resolve_node_id(self, token: str) -> Optional[str]:
        candidate = token.strip().strip('"').strip("'").replace('\\', '/').lstrip('./')
        candidates = [candidate, candidate.lstrip('/'), candidate.split('/', 1)[-1]]
        seen: set[str] = set()
        for cand in candidates:
            cand = cand.strip()
            if not cand or cand in seen:
                continue
            seen.add(cand)
            possible = [cand]
            if '.' not in Path(cand).name:
                for ext in ['.py', '.rs', '.go', '.ts', '.tsx', '.js', '.jsx', '.java', '.sh', '.html', '.css']:
                    possible.append(f"{cand}{ext}")
            for option in possible:
                key = f"{self.namespace}:graph:node:code_file:{option}"
                try:
                    rec = self.router.st.get(key)
                    if rec:
                        return option
                except Exception:
                    continue
        return None

    def _ensure_graph_ready(self) -> None:
        if self._graph_ready:
            return
        try:
            labels_key = f"{self.namespace}:graph:_node_labels"
            existing = self.router.st.get(labels_key) or []
            if existing:
                self._graph_ready = True
                return
        except Exception:
            # fall through to rebuild
            pass
        try:
            graph = build_code_graph(Path(self.workspace))
            _sync_code_graph(self.router, self.namespace, graph, Path(self.workspace))
            self._graph_ready = True
        except Exception:
            # Do not raise inside retrieval path; simply mark as attempted
            self._graph_ready = False


def _sync_code_graph(router: RedDBRouter, namespace: str, graph: Dict[str, Any], workspace: Path) -> None:
    """Persist a code graph into RedDB for downstream Graph-RAG queries."""
    files = graph.get('files') or []
    edges = graph.get('edges') or []
    ws = workspace.resolve()

    def _rel_id(path_str: str) -> str:
        try:
            return str(Path(path_str).resolve().relative_to(ws)).replace(os.sep, '/')
        except Exception:
            return str(Path(path_str)).replace(os.sep, '/')

    # Quick lookup for file metadata by relative path
    facts_by_rel: Dict[str, Dict[str, Any]] = {}
    for file_rec in files:
        path = file_rec.get('path')
        if path:
            facts_by_rel[_rel_id(path)] = file_rec

    # Track nodes/edges we touched so we can drop stale entries afterwards
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()

    def _edge_weight(kind: str, src: Dict[str, Any], dst: Dict[str, Any]) -> float:
        weight = 1.0
        if src.get('language') and dst.get('language') and src.get('language') != dst.get('language'):
            weight += 0.5
        if kind == 'call':
            weight += 0.2
        elif kind.endswith('import'):
            weight += 0.1
        else:
            weight += 0.3
        return round(max(weight, 0.05), 4)

    edge_metrics_raw = collect_edge_metrics(ws, router=router, namespace=namespace)
    edge_metrics = {
        (src.replace('\\', '/'), dst.replace('\\', '/')): metric
        for (src, dst), metric in edge_metrics_raw.items()
    }

    metadata_cache: Dict[str, Any] = {}
    node_records: Dict[str, Dict[str, Any]] = {}

    for file_rec in files:
        path = file_rec.get('path')
        if not path:
            continue
        node_id = _rel_id(path)
        seen_nodes.add(node_id)
        node_data = {
            'id': node_id,
            'path': path,
            'relative_path': node_id,
            'language': file_rec.get('language', 'unknown'),
            'lines': file_rec.get('lines', 0),
            'imports': file_rec.get('imports') or [],
            'classes': file_rec.get('classes') or [],
            'functions': file_rec.get('functions') or [],
            'references': file_rec.get('references') or [],
            'label': 'code_file',
            'node_updated_at': time.time(),
        }
        try:
            file_path = ws / node_id
            snippet = file_path.read_text(errors='ignore')[:4096] if file_path.exists() else ''
        except Exception:
            snippet = ''
        if snippet:
            emb = compute_embedding(snippet)
            if emb:
                node_data['embedding'] = emb
                node_data['embedding_size'] = len(emb)
        metadata = build_node_metadata(ws, node_id, metadata_cache)
        node_data.update({k: v for k, v in metadata.items() if not k.startswith('_')})
        node_records[node_id] = node_data

    adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    in_degree: Dict[str, int] = defaultdict(int)

    for edge_rec in edges:
        src = edge_rec.get('source')
        dst = edge_rec.get('target')
        kind = edge_rec.get('kind') or 'relation'
        if not src or not dst:
            continue
        src_id = _rel_id(src)
        dst_id = _rel_id(dst)
        edge_label = f"code_{kind}"
        edge_id = f"{src_id}->{dst_id}:{kind}"
        src_fact = facts_by_rel.get(src_id, {})
        dst_fact = facts_by_rel.get(dst_id, {})
        weight = _edge_weight(kind, src_fact, dst_fact)
        metric = edge_metrics.get((src_id, dst_id)) or edge_metrics.get((dst_id, src_id))
        history_key = router._k(namespace, 'graph', 'edge_history', edge_label, edge_id)
        history = router.st.get(history_key) or {}
        if metric:
            history = merge_edge_metrics(history, metric)
            router.st.put(history_key, history)
            weight += metric.git_copresence * 0.05 + metric.runtime_hits * 0.1
        if history:
            weight += history.get('decayed_weight', 0.0) * 0.05
        seen_edges.add(edge_id)
        adjacency[src_id].append((dst_id, weight))
        in_degree[dst_id] += 1
        edge_props = {
            'kind': kind,
            'weight': weight,
            'updated_at': time.time(),
        }
        if history:
            edge_props['metrics'] = history
        router.upsert_edge(
            namespace,
            src_id,
            dst_id,
            edge_label,
            edge_props,
            edge_id=edge_id,
        )

    for node_id in node_records:
        adjacency.setdefault(node_id, [])
        in_degree.setdefault(node_id, 0)

    centrality = pagerank(adjacency)
    communities = connected_components(adjacency)
    mixed_nodes = mixed_language_neighbors(
        {nid: {'language': rec.get('language')} for nid, rec in node_records.items()}, adjacency
    )

    generated_at = time.time()
    snapshot = {
        'generated_at': generated_at,
        'pagerank': centrality,
        'communities': communities,
        'mixed_language_nodes': mixed_nodes,
    }
    router.st.put(f"{namespace}:graph:analytics", snapshot)

    out_degree = {node: len(neigh) for node, neigh in adjacency.items()}

    for node_id, node_data in node_records.items():
        node_data['pagerank'] = centrality.get(node_id, 0.0)
        node_data['community'] = communities.get(node_id)
        node_data['out_degree'] = out_degree.get(node_id, 0)
        node_data['in_degree'] = in_degree.get(node_id, 0)
        router.upsert_node(namespace, node_data.get('label', 'code_file'), node_data)

    snapshot_index_key = router._k(namespace, 'graph', 'snapshots', 'index')
    snapshot_index = router.st.get(snapshot_index_key) or []
    snapshot_index.append(generated_at)
    router.st.put(snapshot_index_key, snapshot_index[-10:])
    snapshot_key = router._k(namespace, 'graph', 'snapshot', str(int(generated_at)))
    snapshot_payload = {
        'timestamp': generated_at,
        'nodes': {
            nid: {
                'language': data.get('language'),
                'owner': data.get('owner'),
                'pagerank': data.get('pagerank'),
            } for nid, data in node_records.items()
        },
        'edges': list(seen_edges),
    }
    router.st.put(snapshot_key, snapshot_payload)

    _prune_stale_graph_entries(router, namespace, seen_nodes, seen_edges)

    metadata_key = f"{namespace}:graph:code_graph_sync"
    try:
        router.st.put(
            metadata_key,
            {
                'workspace': str(ws),
                'files': len(files),
                'edges': len(edges),
                'timestamp': time.time(),
            },
        )
    except Exception:
        pass


def _prune_stale_graph_entries(
    router: RedDBRouter,
    namespace: str,
    seen_nodes: set[str],
    seen_edges: set[str],
) -> None:
    """Remove graph entries that are no longer present in the latest code graph."""

    storage = router.st
    # Prune nodes
    node_labels = storage.get(f"{namespace}:graph:_node_labels") or []
    for label in node_labels:
        idx_key = f"{namespace}:collection:graph::node::{label}:_ids"
        ids = storage.get(idx_key) or []
        keep: list[str] = []
        for node_id in ids:
            if label == 'code_file' and node_id not in seen_nodes:
                storage.delete(f"{namespace}:graph:node:{label}:{node_id}")
                continue
            keep.append(node_id)
        if keep != ids:
            storage.put(idx_key, keep)

    # Prune edges
    edge_labels = storage.get(f"{namespace}:graph:_edge_labels") or []
    for label in edge_labels:
        idx_key = f"{namespace}:collection:graph::edge::{label}:_ids"
        ids = storage.get(idx_key) or []
        keep: list[str] = []
        for edge_id in ids:
            if label.startswith('code_') and edge_id not in seen_edges:
                storage.delete(f"{namespace}:graph:edge:{label}:{edge_id}")
                continue
            keep.append(edge_id)
        if keep != ids:
            storage.put(idx_key, keep)


def _truthy_env(name: str) -> Optional[str]:
    import os
    v = os.getenv(name)
    if v is None:
        return None
    return v if v.lower() in {"1", "true", "yes", "on"} else None


def _summarize(mode: str, out: Dict[str, Any]) -> str:
    try:
        if mode == "vector":
            hits = out.get("hits") or []
            if not hits:
                return "No semantic matches"
            samples = [h.get("payload", {}).get("text", "") or str(h) for h in hits[:3]]
            return f"Top semantic matches: \n- " + "\n- ".join(_clip(s, 160) for s in samples)
        if mode == "document":
            hits = out.get("hits") or []
            samples = [str(h.get("text") or h) for h in hits[:3]]
            return f"Top documents:\n- " + "\n- ".join(_clip(s, 160) for s in samples) if samples else "No document hits"
        if mode == "table":
            rows = out.get("rows") or []
            return f"Found {len(rows)} rows" if rows else "No rows match"
        if mode == "graph":
            nb = out.get("neighbors") or {}
            edges = nb.get("edges") or []
            return f"Found {len(edges)} related edges" if edges else "No graph matches"
    except Exception:
        pass
    return "No results"


FILE_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.(?:py|rs|go|tsx|ts|js|jsx|java|sh|bash|html|css)")


def _clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else (s[: n - 3] + "...")


def _overview_vector(vctx: Dict[str, Any]) -> str:
    try:
        hits = vctx.get('hits') or []
        samples = []
        for h in hits[:3]:
            if isinstance(h, dict):
                payload = h.get('payload') or {}
                txt = payload.get('text') or ''
                if txt:
                    samples.append(_clip(txt, 200))
        return "\n".join(f"- {s}" for s in samples) if samples else "(no semantic matches)"
    except Exception:
        return "(no semantic matches)"


def _overview_document(dctx: Dict[str, Any]) -> str:
    try:
        hits = dctx.get('hits') or []
        samples = []
        for h in hits[:3]:
            if isinstance(h, dict):
                samples.append(_clip(str(h.get('text', '')), 200))
            else:
                samples.append(_clip(str(h), 200))
        return "\n".join(f"- {s}" for s in samples) if samples else "(no document hits)"
    except Exception:
        return "(no document hits)"


def _overview_table(tctx: Dict[str, Any]) -> str:
    try:
        rows = tctx.get('rows') or []
        if not rows:
            return "(no matching rows)"
        samples = []
        for r in rows[:3]:
            samples.append(_clip(json.dumps(r), 200))  # type: ignore[name-defined]
        return f"rows={len(rows)}\n" + "\n".join(f"- {s}" for s in samples)
    except Exception:
        return "(no matching rows)"


def _overview_graph(gctx: Dict[str, Any]) -> str:
    try:
        nb = gctx.get('neighbors') or {}
        edges = nb.get('edges') or []
        if not edges:
            return "(no related edges)"
        samples = []
        for e in edges[:5]:
            if isinstance(e, dict):
                samples.append(f"{e.get('src')} --{e.get('label','')}→ {e.get('dst')}")
        return f"edges={len(edges)}\n" + "\n".join(f"- {s}" for s in samples)
    except Exception:
        return "(no related edges)"
