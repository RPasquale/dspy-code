from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from ..db.redb_router import RedDBRouter, QueryRequest
from ..dbkit import RedDBStorage
from ..rl.rlkit import AgentResult, RewardConfig, aggregate_reward, get_verifiers
from ..rl.rl_helpers import load_effective_rl_config_dict
from ..llm import configure_lm, temporary_lm
from ..signatures.data_summary import SummarizeResults


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
            'quality_policy': 1.0,
            'blast_radius': 0.0,
        }
        total, vec_scores, details = aggregate_reward(AgentResult(metrics=metrics, info={}), verifiers, rc)
        fusion = {
            'score': float(total),
            'features': {'vec': f_vec, 'doc': f_doc, 'table': f_tab, 'graph': f_gra},
            'verifier_details': details,
        }
        ctx = {'vector': vec, 'document': doc, 'table': tab, 'graph': gra}
        return fusion, ctx


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
