from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple
from pathlib import Path
import time
import urllib.request
import urllib.error

_CACHE_PATH = Path('.dspy_pricing_cache.json')
_CACHE_TTL = 3600.0  # 1 hour


def _env_overrides() -> Dict[str, float]:
    try:
        raw = os.getenv('PRICING_OVERRIDES')
        if not raw:
            return {}
        data = json.loads(raw)
        return {str(k): float(v) for k, v in dict(data).items()}
    except Exception:
        return {}


def get_s3_storage_price_gb(cloud: str = 'aws', *, live: bool = False) -> float:
    ov = _env_overrides()
    key = f"{cloud}.s3.gb"
    if key in ov:
        return ov[key]
    if live:
        try:
            # Attempt simple AWS pricing fetch via public docs (placeholder). In real impl, use boto3 Pricing API.
            # Fallback to defaults if unavailable.
            # NOTE: Without network or boto3, this branch will usually fallback.
            pass
        except Exception:
            pass
    if cloud.lower() == 'aws':
        return 0.023
    return 0.02


def get_gpu_hour_rate(provider: str = 'prime-intellect', *, model: Optional[str] = None, live: bool = False) -> float:
    ov = _env_overrides()
    key = f"{provider}.gpu.hour"
    if key in ov:
        return ov[key]
    if live:
        try:
            base = os.getenv('PRIME_INTELLECT_API_BASE')
            token = os.getenv('PRIME_INTELLECT_API_TOKEN')
            if base and token and provider.lower() == 'prime-intellect':
                url = base.rstrip('/') + '/pricing'
                req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    # Expecting {"gpu_hour": 0.89} or more detailed pricing
                    rate = data.get('gpu_hour') or data.get('hourly')
                    if rate:
                        return float(rate)
        except Exception:
            pass
    provider = provider.lower()
    if provider == 'prime-intellect':
        return 0.89
    if provider == 'aws':
        return 2.50
    return 1.00


def _load_cache() -> Tuple[dict, float]:
    try:
        if _CACHE_PATH.exists():
            data = json.loads(_CACHE_PATH.read_text())
            return dict(data.get('rates') or {}), float(data.get('ts') or 0.0)
    except Exception:
        pass
    return {}, 0.0


def _save_cache(rates: dict) -> None:
    try:
        _CACHE_PATH.write_text(json.dumps({'ts': time.time(), 'rates': rates}, indent=2))
    except Exception:
        pass


def get_pricing(cloud: str = 'aws', provider: str = 'prime-intellect', *, live: bool = False) -> Dict[str, float]:
    """Return a consolidated pricing map; tries cache/env, falls back to defaults.

    Live pricing hooks are TODO for providers; this function ensures a single place for caching.
    """
    rates, ts = _load_cache()
    fresh = (time.time() - ts) <= _CACHE_TTL
    if not fresh or live:
        # Attempt env overrides first
        s3 = get_s3_storage_price_gb(cloud, live=live)
        gpu = get_gpu_hour_rate(provider, live=live)
        rates.update({
            f"{cloud}.s3.gb": float(s3),
            f"{provider}.gpu.hour": float(gpu),
        })
        _save_cache(rates)
    return rates
