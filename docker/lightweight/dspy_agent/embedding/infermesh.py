from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence
from urllib import request as _req


@dataclass
class InferMeshConfig:
    base_url: str
    model: str
    api_key: Optional[str] = None
    timeout_sec: float = 30.0
    retries: int = 2
    backoff_sec: float = 0.5


class InferMeshEmbedder:
    def __init__(self, base_url: str, model: str, *, api_key: Optional[str] = None, timeout_sec: float = 30.0, retries: int = 2, backoff_sec: float = 0.5) -> None:
        self.cfg = InferMeshConfig(base_url.rstrip('/'), model, api_key=api_key, timeout_sec=timeout_sec, retries=retries, backoff_sec=backoff_sec)

    def _headers(self) -> dict:
        h = {'Content-Type': 'application/json'}
        if self.cfg.api_key:
            h['Authorization'] = f'Bearer {self.cfg.api_key}'
        return h

    def _post_embed(self, inputs: Sequence[str]) -> List[List[float]]:
        body = json.dumps({'model': self.cfg.model, 'inputs': list(inputs)}).encode('utf-8')
        req = _req.Request(self.cfg.base_url + '/embed', headers=self._headers(), data=body, method='POST')
        with _req.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
            data = resp.read()
            obj = json.loads(data.decode('utf-8')) if data else {}
            vecs = obj.get('vectors') or obj.get('embeddings') or []
            return [[float(x) for x in (v or [])] for v in vecs]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        last_err: Optional[Exception] = None
        for attempt in range(max(1, self.cfg.retries + 1)):
            try:
                return self._post_embed(texts)
            except Exception as e:
                last_err = e
                if attempt >= self.cfg.retries:
                    break
                time.sleep(self.cfg.backoff_sec * (2 ** attempt))
        return [[] for _ in texts]


__all__ = ['InferMeshEmbedder', 'InferMeshConfig']

