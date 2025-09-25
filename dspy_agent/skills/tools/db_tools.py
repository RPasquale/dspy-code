from __future__ import annotations

from typing import Any, Dict, Optional

from ...db.redb_router import RedDBRouter, IngestRequest, QueryRequest
from ...dbkit import RedDBStorage
from ..data_rag import DataRAG


def _router(namespace: str) -> RedDBRouter:
    st = RedDBStorage(url=None, namespace=namespace)
    return RedDBRouter(storage=st)


def db_ingest(payload: Dict[str, Any], *, namespace: str = "default") -> Dict[str, Any]:
    """Agent tool: Ingest a record into the intelligent backend.

    payload keys (auto inferred where possible):
      - kind: document|vector|table|graph|auto
      - id, collection, table, text, vector, vectorize, metadata, node, edge
    """
    r = _router(namespace)
    req = IngestRequest(**payload)
    return r.route_ingest(req)


def db_query(payload: Dict[str, Any], *, namespace: str = "default") -> Dict[str, Any]:
    """Agent tool: Query the intelligent backend.

    payload keys:
      - mode: vector|document|table|graph|auto
      - text, collection/index, top_k, table, where, graph
    """
    r = _router(namespace)
    req = QueryRequest(**payload)
    return r.route_query(req)


def db_multi_head(query: str, *, namespace: str = "default", collection: Optional[str] = None, top_k: int = 5, use_lm: bool = False) -> Dict[str, Any]:
    """Agent tool: Run multi-head retrieval with learned fusion, optionally summarized by LLM.
    Returns a dict with 'mode', 'answer', and 'context'.
    """
    rag = DataRAG(namespace=namespace)
    res = rag(query, top_k=top_k, collection=collection, use_lm=use_lm)
    return {"mode": res.mode, "answer": res.answer, "context": res.context}

