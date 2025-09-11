
from __future__ import annotations

import os
from typing import Optional

import dspy

from .config import get_settings


def _truthy(env: str, default: bool = False) -> bool:
    val = os.getenv(env)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


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
        # Default to Ollama's OpenAI-compatible endpoint
        if effective_base_url is None:
            effective_base_url = "http://localhost:11434/v1"
        # Ollama doesn't require a real key; dspy.OpenAI expects something present
        if effective_api_key is None:
            effective_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        # Common default model name for Ollama
        if effective_model in (None, "gpt-4o-mini"):
            effective_model = os.getenv("OLLAMA_MODEL", "llama3")

    # Use OpenAI-compatible LM. For Ollama, we point base_url to its /v1 API.
    lm = dspy.OpenAI(
        model=effective_model,
        api_key=effective_api_key,
        base_url=effective_base_url,
    )
    dspy.configure(lm=lm)
    return lm
