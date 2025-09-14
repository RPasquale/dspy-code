from __future__ import annotations

import os
import json as _json
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol, Tuple
from pathlib import Path
from urllib import request as _req


class Storage(Protocol):
    def put(self, key: str, value: Any) -> None: ...
    def get(self, key: str) -> Optional[Any]: ...
    def delete(self, key: str) -> None: ...
    def append(self, stream: str, value: Any) -> None: ...
    def read(self, stream: str, start: int = 0, count: int = 100) -> Iterable[Tuple[int, Any]]: ...


class RedDBStorage:
    """Thin adapter around RedDB for key-value + append-only streams.

    If REDDB_URL is unset, uses an in-memory fallback for local/dev.
    """

    def __init__(self, url: Optional[str] = None, namespace: str = "dspy") -> None:
        self.url = url or os.getenv("REDDB_URL", "")
        self.ns = os.getenv("REDDB_NAMESPACE", namespace)
        self.token = os.getenv("REDDB_TOKEN", None)
        self.client = None
        self._kv: dict[str, Any] = {}
        self._streams: dict[str, list[Any]] = {}

    def _k(self, key: str) -> str:
        return f"{self.ns}:{key}"

    # KV
    def put(self, key: str, value: Any) -> None:
        if self._http_enabled:
            try:
                self._http_put(f"/api/kv/{self.ns}/{key}", value); return
            except Exception: pass
        self._kv[self._k(key)] = value

    def get(self, key: str) -> Optional[Any]:
        if self._http_enabled:
            try: return self._http_get(f"/api/kv/{self.ns}/{key}")
            except Exception: return None
        return self._kv.get(self._k(key))

    def delete(self, key: str) -> None:
        if self._http_enabled:
            try: self._http_delete(f"/api/kv/{self.ns}/{key}"); return
            except Exception: return
        self._kv.pop(self._k(key), None)

    # Streams
    def append(self, stream: str, value: Any) -> None:
        s = self._stream_key(stream)
        if self._http_enabled:
            try: self._http_post(f"/api/streams/{self.ns}/{stream}/append", value); return
            except Exception: return
        self._streams.setdefault(s, []).append(value)

    def read(self, stream: str, start: int = 0, count: int = 100) -> Iterable[Tuple[int, Any]]:
        s = self._stream_key(stream)
        if self._http_enabled:
            try:
                rows = self._http_get(f"/api/streams/{self.ns}/{stream}/read?start={start}&count={count}") or []
                for r in rows: yield int(r.get("offset", 0)), r.get("value")
                return
            except Exception: return []
        data = self._streams.get(s, []); end = min(len(data), start + count)
        for i in range(start, end): yield i, data[i]

    def _stream_key(self, stream: str) -> str:
        return self._k(f"stream:{stream}")

    # HTTP helpers
    @property
    def _http_enabled(self) -> bool:
        return bool(self.url)

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token: h["Authorization"] = f"Bearer {self.token}"
        return h

    def _http_url(self, path: str) -> str:
        base = self.url.rstrip("/"); p = path if path.startswith("/") else f"/{path}"
        return f"{base}{p}"

    def _http_get(self, path: str) -> Any:
        req = _req.Request(self._http_url(path), headers=self._headers(), method="GET")
        with _req.urlopen(req, timeout=5) as resp:
            data = resp.read(); return _json.loads(data.decode("utf-8")) if data else None

    def _http_post(self, path: str, body: Any) -> Any:
        data = _json.dumps(body).encode("utf-8")
        req = _req.Request(self._http_url(path), headers=self._headers(), data=data, method="POST")
        with _req.urlopen(req, timeout=5) as resp:
            data = resp.read(); return _json.loads(data.decode("utf-8")) if data else None

    def _http_put(self, path: str, body: Any) -> Any:
        data = _json.dumps(body).encode("utf-8")
        req = _req.Request(self._http_url(path), headers=self._headers(), data=data, method="PUT")
        with _req.urlopen(req, timeout=5) as resp:
            data = resp.read(); return _json.loads(data.decode("utf-8")) if data else None

    def _http_delete(self, path: str) -> None:
        req = _req.Request(self._http_url(path), headers=self._headers(), method="DELETE")
        with _req.urlopen(req, timeout=5): return None


def get_storage() -> Optional[Storage]:
    from .config import get_settings
    s = get_settings()
    if s.db_backend.lower() == "reddb":
        return RedDBStorage(url=s.reddb_url, namespace=s.reddb_namespace or "dspy")
    return None

