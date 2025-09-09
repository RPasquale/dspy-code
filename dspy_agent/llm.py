from __future__ import annotations

import os
from typing import Optional

import dspy

from .adapter import StrictJSONAdapter

from .config import get_settings


def _truthy(env: str, default: bool = False) -> bool:
    val = os.getenv(env)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def _strip_v1(url: Optional[str]) -> Optional[str]:
    if not url:
        return url
    if url.endswith("/v1"):
        return url[:-3]
    return url


def configure_lm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
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

    if provider == "ollama":
        # Use native DSPy provider string for Ollama chat models
        if effective_base_url is None:
            effective_base_url = "http://localhost:11434"
        else:
            effective_base_url = _strip_v1(effective_base_url)
        # Ollama doesn't require a real key
        if effective_api_key is None:
            effective_api_key = os.getenv("OLLAMA_API_KEY", "")
        if effective_model in (None, "gpt-4o-mini"):
            effective_model = os.getenv("OLLAMA_MODEL", "deepseek-coder:1.3b")
        model_id = f"ollama_chat/{effective_model}"
    else:
        # OpenAI-compatible (remote or local gateways)
        model_id = effective_model if effective_model.startswith("openai/") else f"openai/{effective_model}"

    lm = dspy.LM(
        model=model_id,
        api_key=effective_api_key or "",
        api_base=effective_base_url,
    )
    # Use a tolerant JSON adapter for providers that often lack structured-outputs support.
    # This avoids noisy fallbacks like: "Failed to use structured output format, falling back to JSON mode."
    force_json = _truthy("DSPY_FORCE_JSON_OBJECT", default=(provider == "ollama"))
    if force_json:
        dspy.configure(lm=lm, adapter=StrictJSONAdapter())
    else:
        dspy.configure(lm=lm)
    return lm
