
from __future__ import annotations

import os
from typing import Optional, Tuple, List, Dict

import dspy

from .config import get_settings

import json as _json
import urllib.request as _req
import urllib.error as _err


def _clamp_litellm_timeout(max_seconds: int = 120) -> None:
    """Ensure LiteLLM timeouts are not excessively high.

    Some LiteLLM adapters default to 600s when environment is unset or mis-set.
    Keep it snappy to avoid long hangs in interactive sessions.
    """
    try:
        cur = float(os.getenv("LITELLM_TIMEOUT", "0"))
    except Exception:
        cur = 0.0
    # Clamp if unset, invalid, or too large
    if cur <= 0 or cur > float(max_seconds):
        os.environ["LITELLM_TIMEOUT"] = str(int(max_seconds))
    # Reasonable retry defaults
    os.environ.setdefault("LITELLM_MAX_RETRIES", "2")
    os.environ.setdefault("LITELLM_NUM_RETRIES", "2")


def _ollama_tags(api_base: str, timeout: float = 0.5) -> List[str]:
    """Fetch available model tags from Ollama. Returns [] on error."""
    try:
        url = api_base.rstrip("/") + "/api/tags"
        req = _req.Request(url, method="GET")
        with _req.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            data = _json.loads(raw)
            models = data.get("models") or []
            tags: List[str] = []
            for m in models:
                # m: {model: "qwen3:1.7b", name: "qwen3", ...}
                tag = m.get("model") or m.get("name")
                if isinstance(tag, str):
                    tags.append(tag)
            return tags
    except Exception:
        return []


def check_ollama_ready(api_base: str, model: Optional[str]) -> Tuple[bool, bool]:
    """Quick readiness probe for Ollama.

    Returns (server_ok, model_ok).
    - server_ok: server reachable and responded to /api/tags
    - model_ok: requested model appears in tags list (best-effort)
    """
    tags = _ollama_tags(api_base)
    if not tags:
        return False, False
    if not model:
        return True, True  # server is up; model unspecified
    # Normalize tag comparison
    m = model.strip()
    # Consider both full tag and base name matches
    base = m.split(":", 1)[0]
    found = any((t == m) or (t.split(":", 1)[0] == base) for t in tags)
    return True, bool(found)


def _truthy(env: str, default: bool = False) -> bool:
    val = os.getenv(env)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


_SAMPLING_HINTS: Dict[str, float] = {}


def get_sampling_hints() -> Dict[str, float]:
    """Return the most recently applied sampling hints (temperature, entropy, etc.)."""

    return dict(_SAMPLING_HINTS)


def configure_lm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    target_entropy: Optional[float] = None,
    clip_higher: Optional[float] = None,
) -> Optional[dspy.LM]:
    settings = get_settings()
    if settings.local_mode:
        return None

    # Determine provider: openai (default) or ollama
    if provider is None:
        provider = "ollama" if _truthy("USE_OLLAMA") else "openai"

    # Effective parameters
    effective_model = model_name or settings.model_name
    effective_api_key = api_key or settings.openai_api_key
    effective_base_url = base_url or settings.openai_base_url

    lm_kwargs: dict = {}

    if provider == "ollama":
        # Use native Ollama API (not the OpenAI compatibility layer)
        # LiteLLM's ollama adapter expects api_base WITHOUT '/v1' and hits '/api/generate'.
        if effective_base_url is None:
            effective_base_url = "http://localhost:11434"
        else:
            # If someone set an OpenAI-style base (ending with /v1), strip it for ollama provider
            if effective_base_url.rstrip('/').endswith('/v1'):
                effective_base_url = effective_base_url.rstrip('/')[:-3]  # remove trailing '/v1'
        # Ollama doesn't require a real key; pass a placeholder if missing
        if effective_api_key is None:
            effective_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        # Common default model name for Ollama
        if effective_model in (None, "gpt-4o-mini"):
            effective_model = os.getenv("OLLAMA_MODEL", "llama3")

        # Fast readiness probe to avoid 10-minute hangs when server/model missing
        server_ok, model_ok = check_ollama_ready(effective_base_url, effective_model)
        if not server_ok:
            # No server → skip LM entirely
            return None
        if not model_ok:
            # Server up but model missing → also skip; user can pull model or change name
            return None

        provider_model = f"ollama/{effective_model}"
        lm_kwargs.update({
            "api_key": effective_api_key,
            "api_base": effective_base_url,
        })
    else:
        # Default to OpenAI
        provider = "openai"
        provider_model = f"openai/{effective_model}"
        # If explicit args aren't provided, rely on env vars OPENAI_API_KEY/OPENAI_API_BASE
        if effective_api_key is not None:
            lm_kwargs["api_key"] = effective_api_key
        if effective_base_url is not None:
            lm_kwargs["api_base"] = effective_base_url

    # Instantiate the LM using DSPy v3 API (via LiteLLM)
    _clamp_litellm_timeout(30)
    lm_kwargs.update({
        "timeout": 30,            # DSPy/LiteLLM timeout
        "request_timeout": 30,    # Some LiteLLM adapters use this key
        "max_retries": 2,
        "num_retries": 2,
    })

    if temperature is not None:
        try:
            temp_val = float(temperature)
            lm_kwargs["temperature"] = temp_val
            _SAMPLING_HINTS["temperature"] = temp_val
            os.environ["DSPY_SAMPLING_TEMPERATURE"] = f"{temp_val:.4f}"
        except Exception:
            pass
    if target_entropy is not None:
        try:
            entropy_val = float(target_entropy)
            _SAMPLING_HINTS["target_entropy"] = entropy_val
            os.environ["DSPY_TARGET_ENTROPY"] = f"{entropy_val:.4f}"
        except Exception:
            pass
    if clip_higher is not None:
        try:
            clip_val = float(clip_higher)
            _SAMPLING_HINTS["clip_higher"] = clip_val
            os.environ["DSPY_CLIP_HIGHER"] = f"{clip_val:.4f}"
        except Exception:
            pass
    
    lm = dspy.LM(
        model=provider_model,
        **lm_kwargs,
    )
    dspy.configure(lm=lm)
    return lm
