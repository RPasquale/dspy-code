from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from urllib import request as _req


logger = logging.getLogger("dspy.infermesh")


def _log_event(level: int, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    try:
        message = json.dumps(payload, sort_keys=True, default=str)
    except TypeError:
        message = f"{event} | " + " ".join(f"{k}={v!r}" for k, v in sorted(fields.items()))
    logger.log(level, message)


@dataclass
class InferMeshConfig:
    base_url: str
    model: str
    api_key: Optional[str] = None
    timeout_sec: float = 30.0
    retries: int = 2
    backoff_sec: float = 0.5
    routing_strategy: Optional[str] = None
    tenant: Optional[str] = None
    priority: Optional[str] = None
    batch_size: Optional[int] = None
    cache_ttl: Optional[int] = None
    cache_key_template: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    hints: Optional[Dict[str, Any]] = None


class InferMeshEmbedder:
    """Simple HTTP client adapter for InferMesh embedding service.

    Exposes an .embed(texts: list[str]) -> list[list[float]] interface compatible
    with dspy_agent.embedding.embeddings_index build routines.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        timeout_sec: float = 30.0,
        retries: int = 2,
        backoff_sec: float = 0.5,
        routing_strategy: Optional[str] = None,
        tenant: Optional[str] = None,
        priority: Optional[str] = None,
        batch_size: Optional[int] = None,
        cache_ttl: Optional[int] = None,
        cache_key_template: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        hints: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = InferMeshConfig(
            base_url.rstrip('/'),
            model,
            api_key=api_key,
            timeout_sec=timeout_sec,
            retries=retries,
            backoff_sec=backoff_sec,
            routing_strategy=routing_strategy,
            tenant=tenant,
            priority=priority,
            batch_size=batch_size,
            cache_ttl=cache_ttl,
            cache_key_template=cache_key_template,
            options=options,
            metadata=metadata,
            hints=hints,
        )

    def _headers(self) -> dict:
        h = {'Content-Type': 'application/json'}
        if self.cfg.api_key:
            h['Authorization'] = f'Bearer {self.cfg.api_key}'
        return h

    def _build_payload(self, inputs: Sequence[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {'model': self.cfg.model, 'inputs': list(inputs)}

        options: Dict[str, Any] = {}
        if self.cfg.options:
            options.update(self.cfg.options)
        if self.cfg.routing_strategy and 'routing_strategy' not in options:
            options['routing_strategy'] = self.cfg.routing_strategy
        if self.cfg.priority and 'priority' not in options:
            options['priority'] = self.cfg.priority
        if self.cfg.batch_size and 'batch_size' not in options:
            options['batch_size'] = int(self.cfg.batch_size)
        if options:
            payload['options'] = options

        metadata: Dict[str, Any] = {}
        if self.cfg.metadata:
            metadata.update(self.cfg.metadata)
        if self.cfg.tenant and 'tenant' not in metadata:
            metadata['tenant'] = self.cfg.tenant
        if metadata:
            payload['metadata'] = metadata

        cache_cfg: Dict[str, Any] = {}
        if self.cfg.cache_ttl is not None:
            cache_cfg['ttl_seconds'] = int(self.cfg.cache_ttl)
        if self.cfg.cache_key_template:
            cache_cfg['key_template'] = self.cfg.cache_key_template
        if cache_cfg:
            payload['cache'] = cache_cfg

        if self.cfg.hints:
            payload['hints'] = self.cfg.hints

        return payload

    def _post_embed(self, inputs: Sequence[str]) -> List[List[float]]:
        payload = self._build_payload(inputs)
        body = json.dumps(payload).encode('utf-8')
        req = _req.Request(self.cfg.base_url + '/embed', headers=self._headers(), data=body, method='POST')
        with _req.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
            data = resp.read()
            obj = json.loads(data.decode('utf-8')) if data else {}
            vecs = obj.get('vectors') or obj.get('embeddings') or []
            return [[float(x) for x in (v or [])] for v in vecs]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        # best-effort retries with exponential backoff
        last_err: Optional[Exception] = None
        for attempt in range(max(1, self.cfg.retries + 1)):
            try:
                _log_event(
                    logging.DEBUG,
                    "infermesh_request",
                    attempt=attempt + 1,
                    size=len(texts),
                    base_url=self.cfg.base_url,
                )
                return self._post_embed(texts)
            except Exception as e:
                last_err = e
                if attempt >= self.cfg.retries:
                    break
                _log_event(
                    logging.WARNING,
                    "infermesh_retry",
                    attempt=attempt + 1,
                    delay_seconds=self.cfg.backoff_sec * (2 ** attempt),
                    error=repr(e),
                )
                time.sleep(self.cfg.backoff_sec * (2 ** attempt))
        # Fallback: empty vectors to preserve alignment
        _log_event(
            logging.ERROR,
            "infermesh_fallback",
            retries=self.cfg.retries,
            attempted=max(1, self.cfg.retries + 1),
            error=repr(last_err) if last_err else None,
            texts=len(texts),
        )
        return [[] for _ in texts]


__all__ = ['InferMeshEmbedder', 'InferMeshConfig']
