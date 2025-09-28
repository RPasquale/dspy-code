"""Lightweight verifiers for signature streaming events."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    data = event.get('payload') if isinstance(event, Mapping) else None
    if isinstance(data, Mapping):
        return data
    return {}


def _metrics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = payload.get('metrics') if isinstance(payload, Mapping) else None
    return metrics if isinstance(metrics, Mapping) else payload


def verify_patch_applied(event: Mapping[str, Any]) -> float:
    payload = _payload(event)
    metrics = _metrics(payload)
    if bool(payload.get('applied')) or metrics.get('status') in {'applied', 'success'}:
        return 1.0
    if metrics.get('patch_failed'):
        return -1.0
    return 0.0


def verify_tests_passed(event: Mapping[str, Any]) -> float:
    payload = _payload(event)
    metrics = _metrics(payload)
    total = metrics.get('tests_total') or metrics.get('tests')
    passed = metrics.get('tests_passed') or metrics.get('passed')
    try:
        total = int(total)
        passed = int(passed)
    except Exception:
        return 0.0
    if total <= 0:
        return 0.0
    return float(passed) / float(total)


def verify_syntax_ok(event: Mapping[str, Any]) -> float:
    payload = _payload(event)
    metrics = _metrics(payload)
    lint_errors = metrics.get('lint_errors') or metrics.get('syntax_errors')
    try:
        errors = int(lint_errors)
    except Exception:
        errors = 0
    return -float(errors)


def verify_embedding_generated(event: Mapping[str, Any]) -> float:
    payload = _payload(event)
    metrics = _metrics(payload)
    vector = None
    if 'vector' in metrics:
        vector = metrics.get('vector')
    elif 'vectors' in metrics:
        vector = metrics.get('vectors')
    if isinstance(vector, (list, tuple)) and vector:
        first = vector[0] if isinstance(vector[0], (list, tuple)) else vector
        length = len(first)
        if length > 0 and not math.isnan(float(first[0])):
            return 1.0
    return 0.0


__all__ = [
    'verify_patch_applied',
    'verify_tests_passed',
    'verify_syntax_ok',
    'verify_embedding_generated',
]
