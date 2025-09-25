from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

from ..dbkit import RedDBStorage


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


@dataclass
class IngestRequest:
    kind: str  # vector|graph|document|table|auto
    id: Optional[str] = None
    namespace: str = "default"
    collection: Optional[str] = None  # for documents/vectors
    table: Optional[str] = None  # for tables
    text: Optional[str] = None  # free text / documents
    vector: Optional[List[float]] = None  # explicit embedding
    vectorize: bool = False  # request on-the-fly embedding
    metadata: Dict[str, Any] = None  # arbitrary fields
    # Graph
    node: Optional[Dict[str, Any]] = None
    edge: Optional[Dict[str, Any]] = None  # {src, dst, label, props}
    schema: Optional[Dict[str, Any]] = None  # table schema hint


@dataclass
class QueryRequest:
    mode: str  # vector|graph|document|table|auto
    namespace: str = "default"
    text: Optional[str] = None
    collection: Optional[str] = None
    index: Optional[str] = None  # alias for collection when vector
    top_k: int = 5
    table: Optional[str] = None
    where: Optional[Dict[str, Any]] = None
    graph: Optional[Dict[str, Any]] = None  # e.g., {neighbors: {node_id, k}}


class RedDBRouter:
    """High-level data router on top of RedDBStorage.

    - Uses KV for typed buckets, preserving namespaces.
    - Provides vector upsert/search, document insert/search, table upsert/query, graph upsert/query.
    - Falls back to in-memory mode when REDDB_URL is not configured.
    """

    def __init__(self, storage: Optional[RedDBStorage] = None, *, workspace: Optional[Path] = None) -> None:
        self.st = storage or RedDBStorage(url=None, namespace="dspy")
        self.workspace = Path(workspace) if workspace else Path.cwd()
        # Best-effort native adapter for redb-open (KV fallback if unavailable)
        self._native = _RedBOpenAdapter(self.st)

    # ------------------------
    # Namespaced bucket helpers
    # ------------------------
    def _k(self, *parts: str) -> str:
        return ":".join([p for p in parts if p])

    # ------------------------
    # Document store (collections)
    # ------------------------
    def put_document(self, ns: str, collection: str, id: str, doc: Dict[str, Any]) -> None:
        if self._native.available and self._native.put_document(ns, collection, id, doc):
            return
        self.st.put(self._k(ns, "collection", collection, id), doc)

    def search_documents(self, ns: str, collection: str, text: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        if self._native.available:
            out = self._native.search_documents(ns, collection, text, limit=limit)
            if out is not None:
                return out
        # Linear scan over collection keys in memory fallback; in HTTP mode rely on KV scan via a prefix stream when available.
        # Here we use a simple convention: pointer to list of IDs
        idx_key = self._k(ns, "collection", collection, "_ids")
        ids = self.st.get(idx_key) or []
        out: List[Dict[str, Any]] = []
        t = (text or "").lower()
        for did in ids:
            doc = self.st.get(self._k(ns, "collection", collection, did)) or {}
            blob = json.dumps(doc).lower()
            if t in blob:
                out.append(doc)
            if len(out) >= limit:
                break
        return out

    def _index_document_id(self, ns: str, collection: str, id: str) -> None:
        idx_key = self._k(ns, "collection", collection, "_ids")
        ids = self.st.get(idx_key) or []
        if id not in ids:
            ids.append(id)
            self.st.put(idx_key, ids)

    # ------------------------
    # Vector store (collections)
    # ------------------------
    def upsert_vector(self, ns: str, index: str, id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        if self._native.available and self._native.upsert_vector(ns, index, id, vector, payload):
            self._index_document_id(ns, f"vector::{index}", id)
            return
        key = self._k(ns, "vector", index, id)
        self.st.put(key, {"v": vector, "payload": payload})
        self._index_document_id(ns, f"vector::{index}", id)

    def vector_search(self, ns: str, index: str, query: List[float], *, top_k: int = 5) -> List[Dict[str, Any]]:
        if self._native.available:
            out = self._native.vector_search(ns, index, query, top_k=top_k)
            if out is not None:
                return out
        idx_key = self._k(ns, "collection", f"vector::{index}", "_ids")
        ids = self.st.get(idx_key) or []
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for id in ids:
            rec = self.st.get(self._k(ns, "vector", index, id)) or {}
            vec = rec.get("v") or []
            score = _cosine(query, vec)
            scored.append((score, {"id": id, **rec}))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [x for _, x in scored[: max(1, int(top_k))]]

    # ------------------------
    # Tables (rows)
    # ------------------------
    def upsert_row(self, ns: str, table: str, row: Dict[str, Any]) -> None:
        if self._native.available and self._native.upsert_row(ns, table, row):
            self._index_document_id(ns, f"table::{table}", str(row.get("id")))
            return
        # Keep an auto-increment id if not provided
        rid = str(row.get("id") or row.get("_id") or self._next_row_id(ns, table))
        row = {**row, "id": rid}
        self.st.put(self._k(ns, "table", table, rid), row)
        self._index_document_id(ns, f"table::{table}", rid)

    def _next_row_id(self, ns: str, table: str) -> int:
        k = self._k(ns, "table", table, "_seq")
        n = int(self.st.get(k) or 0) + 1
        self.st.put(k, n)
        return n

    def table_query(self, ns: str, table: str, where: Optional[Dict[str, Any]] = None, *, limit: int = 50) -> List[Dict[str, Any]]:
        if self._native.available:
            out = self._native.table_query(ns, table, where=where, limit=limit)
            if out is not None:
                return out
        idx_key = self._k(ns, "collection", f"table::{table}", "_ids")
        ids = self.st.get(idx_key) or []
        out: List[Dict[str, Any]] = []
        for rid in ids:
            row = self.st.get(self._k(ns, "table", table, rid)) or {}
            if where and not _matches_where(row, where):
                continue
            out.append(row)
            if len(out) >= limit:
                break
        return out

    # ------------------------
    # Graph (nodes/edges)
    # ------------------------
    def upsert_node(self, ns: str, label: str, node: Dict[str, Any]) -> None:
        if self._native.available and self._native.upsert_node(ns, label, node):
            self._register_node_label(ns, label)
            return
        nid = str(node.get("id") or node.get("_id") or node.get("name") or self._next_row_id(ns, f"graph::{label}"))
        rec = {**node, "id": nid, "label": label}
        self.st.put(self._k(ns, "graph", "node", label, nid), rec)
        self._index_document_id(ns, f"graph::node::{label}", nid)

    def upsert_edge(self, ns: str, src: str, dst: str, label: str, props: Optional[Dict[str, Any]] = None) -> None:
        if self._native.available and self._native.upsert_edge(ns, src, dst, label, props=props):
            self._register_edge_label(ns, label)
            return
        eid = str(self._next_row_id(ns, f"graph::edge::{label}"))
        rec = {"id": eid, "src": src, "dst": dst, "label": label, "props": props or {}}
        self.st.put(self._k(ns, "graph", "edge", label, eid), rec)
        self._index_document_id(ns, f"graph::edge::{label}", eid)

    def neighbors(self, ns: str, node_id: str, *, limit: int = 10) -> Dict[str, Any]:
        if self._native.available:
            out = self._native.neighbors(ns, node_id, limit=limit)
            if out is not None:
                return out
        # Collect edges across all labels
        prefix = self._k(ns, "collection", "graph::edge::")
        # We can't list by prefix via HTTP; rely on maintaining indices per label
        result_nodes: List[Dict[str, Any]] = []
        result_edges: List[Dict[str, Any]] = []
        # Heuristic: inspect a few common labels
        for edge_label in self._edge_labels(ns):
            ids = self.st.get(self._k(ns, "collection", f"graph::edge::{edge_label}", "_ids")) or []
            for eid in ids:
                e = self.st.get(self._k(ns, "graph", "edge", edge_label, eid)) or {}
                if e.get("src") == node_id or e.get("dst") == node_id:
                    result_edges.append(e)
                    if len(result_edges) >= limit:
                        break
        # Load unique nodes
        seen: set[str] = set()
        for e in result_edges:
            for nid in (e.get("src"), e.get("dst")):
                if not nid or nid in seen:
                    continue
                seen.add(nid)
                node = self._find_node_any_label(ns, nid)
                if node:
                    result_nodes.append(node)
        return {"node": node_id, "neighbors": result_nodes, "edges": result_edges}

    def _edge_labels(self, ns: str) -> List[str]:
        # Track known edge labels in a registry key
        k = self._k(ns, "graph", "_edge_labels")
        labels = self.st.get(k) or []
        return list(dict.fromkeys(labels))

    def _register_edge_label(self, ns: str, label: str) -> None:
        k = self._k(ns, "graph", "_edge_labels")
        labels = self.st.get(k) or []
        if label not in labels:
            labels.append(label)
            self.st.put(k, labels)

    def _find_node_any_label(self, ns: str, node_id: str) -> Optional[Dict[str, Any]]:
        # Try labels from registry
        k = self._k(ns, "graph", "_node_labels")
        labs = self.st.get(k) or []
        for label in labs:
            n = self.st.get(self._k(ns, "graph", "node", label, node_id))
            if n:
                return n
        return None

    def _register_node_label(self, ns: str, label: str) -> None:
        k = self._k(ns, "graph", "_node_labels")
        labs = self.st.get(k) or []
        if label not in labs:
            labs.append(label)
            self.st.put(k, labs)

    # ------------------------
    # Ingest & Query routing
    # ------------------------
    def route_ingest(self, req: IngestRequest) -> Dict[str, Any]:
        kind = (req.kind or "auto").lower()
        ns = req.namespace or "default"
        meta = dict(req.metadata or {})
        if kind == "auto":
            kind = self._infer_kind(req)

        if kind == "document":
            coll = req.collection or "default"
            did = str(req.id or self._next_row_id(ns, f"collection::{coll}"))
            doc = {"id": did, "text": req.text or "", "meta": meta}
            self.put_document(ns, coll, did, doc)
            self._index_document_id(ns, coll, did)
            return {"ok": True, "kind": "document", "id": did}

        if kind == "vector":
            coll = req.collection or "default"
            did = str(req.id or self._next_row_id(ns, f"vector::{coll}"))
            vec = list(req.vector or [])
            if not vec and (req.vectorize and req.text):
                try:
                    from ..embedding.embedder import Embedder
                    vec = Embedder().embed_text(req.text or "")
                except Exception:
                    vec = []
            self.upsert_vector(ns, coll, did, vec, {"text": req.text or "", **meta})
            return {"ok": True, "kind": "vector", "id": did}

        if kind == "table":
            table = req.table or "default"
            row = dict(meta)
            if req.text:
                row["text"] = req.text
            if req.id:
                row["id"] = req.id
            self.upsert_row(ns, table, row)
            return {"ok": True, "kind": "table", "table": table}

        if kind == "graph":
            if req.node:
                label = str(req.node.get("label") or "node")
                self._register_node_label(ns, label)
                self.upsert_node(ns, label, req.node)
            if req.edge:
                lbl = str(req.edge.get("label") or "edge")
                self._register_edge_label(ns, lbl)
                self.upsert_edge(ns, str(req.edge.get("src")), str(req.edge.get("dst")), lbl, req.edge.get("props") or {})
            return {"ok": True, "kind": "graph"}

        return {"ok": False, "error": f"unsupported kind: {kind}"}

    def route_query(self, req: QueryRequest) -> Dict[str, Any]:
        mode = (req.mode or "auto").lower()
        ns = req.namespace or "default"
        if mode == "auto":
            mode = self._infer_query_mode(req)

        if mode == "vector":
            idx = req.index or req.collection or "default"
            qvec: List[float] = []
            if req.text:
                try:
                    from ..embedding.embedder import Embedder
                    qvec = Embedder().embed_text(req.text)
                except Exception:
                    qvec = []
            hits = self.vector_search(ns, idx, qvec, top_k=int(req.top_k or 5))
            return {"mode": "vector", "index": idx, "hits": hits}

        if mode == "document":
            coll = req.collection or "default"
            hits = self.search_documents(ns, coll, req.text or "", limit=int(req.top_k or 5))
            return {"mode": "document", "collection": coll, "hits": hits}

        if mode == "table":
            table = req.table or "default"
            rows = self.table_query(ns, table, where=req.where or {}, limit=int(req.top_k or 50))
            return {"mode": "table", "table": table, "rows": rows}

        if mode == "graph":
            g = dict(req.graph or {})
            if "neighbors" in g:
                nb = g["neighbors"] or {}
                node_id = str(nb.get("node_id") or nb.get("id") or "")
                k = int(nb.get("k") or 10)
                out = self.neighbors(ns, node_id, limit=k)
                return {"mode": "graph", "neighbors": out}
            return {"mode": "graph", "error": "unsupported graph query"}

        return {"error": f"unsupported mode: {mode}"}

    # ------------------------
    # Heuristics
    # ------------------------
    def _infer_kind(self, req: IngestRequest) -> str:
        if req.edge or req.node:
            return "graph"
        if req.table or req.schema:
            return "table"
        if req.vector or req.vectorize:
            return "vector"
        return "document"

    def _infer_query_mode(self, req: QueryRequest) -> str:
        t = (req.text or "").strip().lower()
        if t.startswith("select ") or " where " in t or req.table:
            return "table"
        if "neighbor" in t or req.graph:
            return "graph"
        # Default: vector first for semantic, then document fallback (handled by caller)
        return "vector"


def _matches_where(row: Dict[str, Any], where: Dict[str, Any]) -> bool:
    for k, v in where.items():
        if str(row.get(k)) != str(v):
            return False
    return True


# ---------------------
# Native redb-open adapter (best-effort; falls back to KV)
# ---------------------
class _RedBOpenAdapter:
    def __init__(self, storage: RedDBStorage) -> None:
        self.storage = storage
        self.base = storage.url or ""
        self.available = False
        if self.base:
            try:
                # Probe /health if available via storage health_check
                hc = storage.health_check() or {}
                # Consider HTTP mode ok but do not claim native endpoints unless explicitly opted-in
                # This avoids accidental reliance on unknown API shapes.
                self.available = bool(hc.get("mode") == "http" and os.getenv("REDB_OPEN_NATIVE", "false").lower() in {"1","true","yes","on"})
            except Exception:
                self.available = False

    # The native methods return True/False on write success, or a list on read; None means "not supported".
    def put_document(self, ns: str, collection: str, id: str, doc: Dict[str, Any]) -> bool:
        try:
            payload = {"id": id, "doc": doc}
            self.storage._http_post(f"/api/collections/{ns}/{collection}/docs", payload)
            return True
        except Exception:
            return False

    def search_documents(self, ns: str, collection: str, text: str, *, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        try:
            payload = {"q": text, "limit": int(limit)}
            data = self.storage._http_post(f"/api/collections/{ns}/{collection}/search", payload)
            if isinstance(data, dict):
                hits = data.get("hits") or []
                out: List[Dict[str, Any]] = []
                for h in hits:
                    if isinstance(h, dict):
                        if "doc" in h:
                            out.append(h.get("doc") or {})
                        else:
                            out.append(h)
                return out
            return None
        except Exception:
            return None

    def upsert_vector(self, ns: str, index: str, id: str, vector: List[float], payload: Dict[str, Any]) -> bool:
        try:
            body = {"id": id, "vector": vector, "payload": payload}
            self.storage._http_post(f"/api/vectors/{ns}/{index}/upsert", body)
            return True
        except Exception:
            return False

    def vector_search(self, ns: str, index: str, query: List[float], *, top_k: int = 5) -> Optional[List[Dict[str, Any]]]:
        try:
            body = {"vector": query, "top_k": int(top_k)}
            data = self.storage._http_post(f"/api/vectors/{ns}/{index}/search", body)
            if isinstance(data, dict):
                hits = data.get("hits") or []
                out: List[Dict[str, Any]] = []
                for h in hits:
                    if isinstance(h, dict):
                        out.append(h)
                return out
            return None
        except Exception:
            return None

    def upsert_row(self, ns: str, table: str, row: Dict[str, Any]) -> bool:
        try:
            self.storage._http_post(f"/api/tables/{ns}/{table}/rows", row)
            return True
        except Exception:
            return False

    def table_query(self, ns: str, table: str, where: Optional[Dict[str, Any]] = None, *, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
        try:
            body = {"where": where or {}, "limit": int(limit)}
            data = self.storage._http_post(f"/api/tables/{ns}/{table}/query", body)
            if isinstance(data, dict) and isinstance(data.get("rows"), list):
                return list(data.get("rows") or [])
            return None
        except Exception:
            return None

    def upsert_node(self, ns: str, label: str, node: Dict[str, Any]) -> bool:
        try:
            body = {**node, "label": label}
            self.storage._http_post(f"/api/graph/{ns}/nodes", body)
            return True
        except Exception:
            return False

    def upsert_edge(self, ns: str, src: str, dst: str, label: str, props: Optional[Dict[str, Any]] = None) -> bool:
        try:
            body = {"src": src, "dst": dst, "label": label, "props": props or {}}
            self.storage._http_post(f"/api/graph/{ns}/edges", body)
            return True
        except Exception:
            return False

    def neighbors(self, ns: str, node_id: str, *, limit: int = 10) -> Optional[Dict[str, Any]]:
        try:
            data = self.storage._http_get(f"/api/graph/{ns}/neighbors/{node_id}?k={int(limit)}")
            if isinstance(data, dict):
                return data
            return None
        except Exception:
            return None
