from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as e:  # pragma: no cover - optional at runtime
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore
    HTTPException = Exception  # type: ignore

from ..db.redb_router import RedDBRouter, IngestRequest as _Ingest, QueryRequest as _Query
from ..dbkit import RedDBStorage


class IngestRequest(BaseModel):
    kind: str = Field(default="auto")
    id: Optional[str] = None
    namespace: str = Field(default="default")
    collection: Optional[str] = None
    table: Optional[str] = None
    text: Optional[str] = None
    vector: Optional[List[float]] = None
    vectorize: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    node: Optional[Dict[str, Any]] = None
    edge: Optional[Dict[str, Any]] = None
    schema: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    mode: str = Field(default="auto")
    namespace: str = Field(default="default")
    text: Optional[str] = None
    collection: Optional[str] = None
    index: Optional[str] = None
    top_k: int = 5
    table: Optional[str] = None
    where: Optional[Dict[str, Any]] = None
    graph: Optional[Dict[str, Any]] = None


def _make_router() -> RedDBRouter:
    url = os.getenv("REDDB_URL")
    ns = os.getenv("REDDB_NAMESPACE", "agent")
    st = RedDBStorage(url=url, namespace=ns)
    return RedDBRouter(storage=st)


def build_app() -> Any:
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed. Install with: pip install fastapi uvicorn")
    app = FastAPI(title="Intelligent Data Backend", version="0.1.0")
    router = _make_router()

    @app.get("/api/db/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "ts": time.time(), "storage": router.st.health_check()}

    @app.get("/api/db/stats")
    def stats() -> Dict[str, Any]:
        return {"ok": True, "ts": time.time(), "namespace": router.st.ns}

    @app.post("/api/db/ingest")
    def ingest(req: IngestRequest) -> Dict[str, Any]:
        try:
            out = router.route_ingest(_Ingest(**req.dict()))
            if not out.get("ok"):
                raise HTTPException(status_code=400, detail=out)
            return out
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/db/query")
    def query(req: QueryRequest) -> Dict[str, Any]:
        try:
            out = router.route_query(_Query(**req.dict()))
            # Fallback to document search if vector produced no hits
            if out.get("mode") == "vector" and not (out.get("hits") or []):
                q2 = QueryRequest(mode="document", namespace=req.namespace, text=req.text, collection=req.collection)
                out = router.route_query(_Query(**q2.dict()))
            return out
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(e))

    return app


def start_fastapi_backend(host: str = "0.0.0.0", port: int = 8767) -> None:
    if FastAPI is None:
        raise RuntimeError("fastapi/uvicorn are not installed")
    import uvicorn  # type: ignore

    uvicorn.run(build_app(), host=host, port=int(port), log_level="info")


if __name__ == "__main__":
    start_fastapi_backend()

