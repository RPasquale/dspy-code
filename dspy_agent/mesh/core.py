from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error


@dataclass
class MeshCoreConfig:
    base_url: str
    timeout_sec: float = 2.5


class MeshCoreClient:
    """Tiny HTTP client for RedDB Mesh Core gateway (best-effort).

    Exposes a minimal surface for status, topics, and event tails.
    """

    def __init__(self, base_url: Optional[str] = None, timeout_sec: float = 2.5) -> None:
        url = (base_url or os.getenv('MESH_CORE_URL') or '').strip() or 'http://mesh-core:8080'
        self.cfg = MeshCoreConfig(base_url=url.rstrip('/'), timeout_sec=timeout_sec)

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.cfg.base_url}{path}"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
            raw = resp.read().decode('utf-8')
            try:
                return json.loads(raw)
            except Exception:
                return {'raw': raw}

    def status(self) -> Dict[str, Any]:
        t0 = time.time()
        try:
            # Convention: /api/status should exist; fallback to root
            try:
                data = self._get('/api/status')
            except Exception:
                data = self._get('/')
            rtt = (time.time() - t0) * 1000.0
            return {'ok': True, 'rtt_ms': round(rtt, 1), 'endpoint': self.cfg.base_url, 'data': data}
        except Exception as e:
            return {'ok': False, 'error': str(e), 'endpoint': self.cfg.base_url}

    def topics(self) -> Dict[str, Any]:
        try:
            # Convention: /api/topics
            return self._get('/api/topics')
        except Exception as e:
            return {'error': str(e)}

    def tail(self, topic: str, limit: int = 50) -> Dict[str, Any]:
        try:
            # Convention: /api/tail?topic=&limit=
            q = f"/api/tail?topic={urllib.parse.quote(topic)}&limit={int(limit)}"
            return self._get(q)
        except Exception as e:
            return {'error': str(e)}

__all__ = ['MeshCoreClient']

