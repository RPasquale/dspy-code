from __future__ import annotations

"""
Minimal local RedDB-compatible HTTP server (mock) for testing the agent.

Implements the KV + Streams HTTP API used by RedDBStorage:
  - KV:
      PUT  /api/kv/{namespace}/{key}        body: JSON value
      GET  /api/kv/{namespace}/{key}
      DELETE /api/kv/{namespace}/{key}
  - Streams:
      POST /api/streams/{ns}/{stream}/append  body: JSON value
      GET  /api/streams/{ns}/{stream}/read?start=0&count=100

This server keeps data in-process memory for simplicity. Use only for local testing.
"""

import time
from typing import Any, Dict, List, Tuple

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
except Exception:
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    JSONResponse = dict  # type: ignore


class _Store:
    def __init__(self) -> None:
        self.kv: Dict[Tuple[str, str], Any] = {}
        self.streams: Dict[Tuple[str, str], List[Any]] = {}

    # KV
    def kv_put(self, ns: str, key: str, value: Any) -> None:
        self.kv[(ns, key)] = value

    def kv_get(self, ns: str, key: str) -> Any:
        if (ns, key) not in self.kv:
            raise KeyError(key)
        return self.kv[(ns, key)]

    def kv_delete(self, ns: str, key: str) -> None:
        self.kv.pop((ns, key), None)

    # Streams
    def stream_append(self, ns: str, stream: str, value: Any) -> int:
        s = self.streams.setdefault((ns, stream), [])
        s.append(value)
        return len(s) - 1  # offset

    def stream_read(self, ns: str, stream: str, start: int = 0, count: int = 100) -> List[Dict[str, Any]]:
        s = self.streams.get((ns, stream), [])
        start = max(0, int(start))
        end = min(len(s), start + max(0, int(count)))
        out: List[Dict[str, Any]] = []
        for i in range(start, end):
            out.append({"offset": i, "value": s[i]})
        return out


def build_app() -> Any:
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed. Install with: pip install fastapi uvicorn")
    app = FastAPI(title="RedDB Mock", version="0.1.0")
    store = _Store()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "ts": time.time(), "mode": "mock"}

    @app.get("/api/health")
    def api_health() -> Dict[str, Any]:
        return {"ok": True, "ts": time.time(), "mode": "mock"}

    # KV
    @app.put("/api/kv/{ns}/{key}")
    def kv_put(ns: str, key: str, value: Any) -> Dict[str, Any]:
        store.kv_put(ns, key, value)
        return {"ok": True}

    @app.get("/api/kv/{ns}/{key}")
    def kv_get(ns: str, key: str) -> Any:
        try:
            return store.kv_get(ns, key)
        except KeyError:
            raise HTTPException(status_code=404, detail="not found")

    @app.delete("/api/kv/{ns}/{key}")
    def kv_delete(ns: str, key: str) -> Dict[str, Any]:
        store.kv_delete(ns, key)
        return {"ok": True}

    # Streams
    @app.post("/api/streams/{ns}/{stream}/append")
    def stream_append(ns: str, stream: str, value: Any) -> Dict[str, Any]:
        off = store.stream_append(ns, stream, value)
        return {"ok": True, "offset": off}

    @app.get("/api/streams/{ns}/{stream}/read")
    def stream_read(ns: str, stream: str, start: int = 0, count: int = 100) -> List[Dict[str, Any]]:
        return store.stream_read(ns, stream, start=start, count=count)

    return app


def start_reddb_mock(host: str = "0.0.0.0", port: int = 8080) -> None:
    if FastAPI is None:
        raise RuntimeError("fastapi/uvicorn are not installed")
    import uvicorn  # type: ignore
    uvicorn.run(build_app(), host=host, port=int(port), log_level="info")


if __name__ == "__main__":
    start_reddb_mock()

