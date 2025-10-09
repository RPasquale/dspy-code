
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional, Tuple, List, Dict, Any

import dspy

from .config import get_settings
from .agents.adapter import StrictJSONAdapter  # type: ignore
try:
    # best-effort import of adapter circuit breaker for health reporting
    from .agents.adapter import _CB as _ADAPTER_CB  # type: ignore
except Exception:  # pragma: no cover - optional
    _ADAPTER_CB = None  # type: ignore

import json as _json
import urllib.request as _req
import urllib.error as _err


logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    """Best-effort float parsing for environment overrides."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
        if value <= 0:
            raise ValueError
        return value
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    """Best-effort int parsing for environment overrides."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
        return value
    except Exception:
        return default


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


def _split_model_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    models: List[str] = []
    for chunk in str(raw).replace(";", ",").split(","):
        token = chunk.strip()
        if token:
            models.append(token)
    return models


def _ollama_tags(api_base: str, timeout: Optional[float] = None) -> List[str]:
    """Fetch available model tags from Ollama. Returns [] on error."""
    cache_key = api_base.rstrip("/")
    now = time.time()
    with _OLLAMA_TAG_CACHE_LOCK:
        cached = _OLLAMA_TAG_CACHE.get(cache_key)
        if cached and (now - cached[0]) < _OLLAMA_TAG_TTL:
            return list(cached[1])

    try:
        url = api_base.rstrip("/") + "/api/tags"
        req = _req.Request(url, method="GET")
        eff_timeout = timeout if timeout is not None else _env_float("OLLAMA_TAG_TIMEOUT", 2.0)
        with _req.urlopen(req, timeout=eff_timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            data = _json.loads(raw)
            models = data.get("models") or []
            tags: List[str] = []
            for m in models:
                # m: {model: "qwen3:1.7b", name: "qwen3", ...}
                tag = m.get("model") or m.get("name")
                if isinstance(tag, str):
                    tags.append(tag)
            with _OLLAMA_TAG_CACHE_LOCK:
                _OLLAMA_TAG_CACHE[cache_key] = (now, list(tags))
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


def detect_available_ollama_models(preferred: Optional[List[str]] = None) -> List[str]:
    """Return a best-effort ordered list of Ollama model tags to use."""

    # Start with explicit env overrides
    env_models = _split_model_list(os.getenv("OLLAMA_MODELS"))
    if env_models:
        return env_models
    single = os.getenv("OLLAMA_MODEL") or os.getenv("MODEL_NAME")
    if single:
        models = _split_model_list(single)
        if models:
            return models

    preferred = preferred or ["deepseek-coder:1.3b", "qwen3:1.7b"]

    base_hint = os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
    base_urls: List[str] = []
    if base_hint:
        base_urls.append(base_hint)
    base_urls.extend([
        "http://127.0.0.1:11435",
        "http://127.0.0.1:11434",
        "http://ollama:11434",
    ])

    seen: set[str] = set()
    tags: List[str] = []
    for base in base_urls:
        key = base.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        tags = _ollama_tags(key)
        if tags:
            break

    if tags:
        ordered: List[str] = []
        for pref in preferred:
            if pref in ordered:
                continue
            if any(tag == pref for tag in tags):
                ordered.append(pref)
                continue
            pref_base = pref.split(":", 1)[0]
            for tag in tags:
                if tag.split(":", 1)[0] == pref_base and tag not in ordered:
                    ordered.append(tag)
                    break
        for tag in tags:
            if tag not in ordered:
                ordered.append(tag)
        return ordered

    return list(preferred)


def get_default_ollama_model() -> str:
    """Return the preferred Ollama model tag to use by default."""

    models = detect_available_ollama_models()
    return models[0] if models else "deepseek-coder:1.3b"


def _truthy(env: str, default: bool = False) -> bool:
    val = os.getenv(env)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


_SAMPLING_HINTS: Dict[str, float] = {}
_LM_CACHE: Dict[Tuple[Any, ...], dspy.LM] = {}
_OLLAMA_LOCK = threading.Lock()
_OLLAMA_ACTIVE_KEY: Optional[Tuple[str, str]] = None
_OLLAMA_ACTIVE_LM: Optional[dspy.LM] = None
_OLLAMA_WARNED_SWITCH = False
_OLLAMA_TAG_CACHE: Dict[str, Tuple[float, List[str]]] = {}
_OLLAMA_TAG_CACHE_LOCK = threading.Lock()
_OLLAMA_TAG_TTL = 15.0


def get_sampling_hints() -> Dict[str, float]:
    """Return the most recently applied sampling hints (temperature, entropy, etc.)."""

    return dict(_SAMPLING_HINTS)


def get_circuit_breaker_status() -> Dict[str, object]:
    """Expose adapter circuit-breaker state for health endpoints.

    Returns a small dict with keys: open(bool), next_retry(float seconds).
    """
    try:
        if _ADAPTER_CB is None:
            return {"open": False, "next_retry_sec": 0.0}
        import time as _t
        now = _t.time()
        # Adapter exposes is_open and private _opened_until
        opened_until = getattr(_ADAPTER_CB, "_opened_until", 0.0)
        is_open = bool(getattr(_ADAPTER_CB, "is_open", False))
        wait = max(0.0, float(opened_until) - now)
        return {"open": is_open, "next_retry_sec": wait}
    except Exception:
        return {"open": False, "next_retry_sec": 0.0}


def _configure_dspy(lm: dspy.LM) -> None:
    """Set the active DSPy LM, handling adapter compatibility."""

    try:
        dspy.configure(lm=lm, adapter=StrictJSONAdapter())
    except TypeError:
        dspy.configure(lm=lm)


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
    global _OLLAMA_ACTIVE_LM, _OLLAMA_ACTIVE_KEY, _OLLAMA_WARNED_SWITCH
    # If running with the lightweight local stub, skip LM entirely
    # However, in Docker environments, we need to override this to allow LM configuration
    try:
        if getattr(dspy, 'IS_STUB', False):
            # In Docker environments, override IS_STUB to allow LM configuration
            if os.getenv('DOCKER_ENV', 'false').lower() in {'true', '1', 'yes', 'on'}:
                dspy.IS_STUB = False
            else:
                return None
    except Exception:
        pass
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
    cache_key: Optional[Tuple[Any, ...]] = None

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
            preferred = os.getenv("OLLAMA_MODEL")
            if not preferred:
                raw_models = os.getenv("OLLAMA_MODELS", "")
                for candidate in raw_models.replace(",", " ").split():
                    c = candidate.strip()
                    if c:
                        preferred = c
                        break
            effective_model = preferred or "llama3"

        # Fast readiness probe for insight; proceed even if warnings fire
        server_ok, model_ok = check_ollama_ready(effective_base_url, effective_model)
        if not server_ok:
            logger.warning(
                "Ollama server at %s not responding to /api/tags probe; continuing setup anyway",
                effective_base_url,
            )
        if server_ok and not model_ok and not _truthy("OLLAMA_ALLOW_MISSING_MODEL"):
            available = detect_available_ollama_models()
            fallback_model = next((m for m in available if m), None)
            if fallback_model:
                if fallback_model != effective_model:
                    logger.warning(
                        "Ollama model '%s' not found; falling back to '%s'",
                        effective_model,
                        fallback_model,
                    )
                effective_model = fallback_model
                model_ok = True
            else:
                logger.warning(
                    "Ollama model '%s' not found and no fallback model detected; continuing anyway",
                    effective_model,
                )

        provider_model = f"ollama/{effective_model}"
        lm_kwargs.update({
            "api_key": effective_api_key,
            "api_base": effective_base_url,
        })

        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE")
        if keep_alive:
            lm_kwargs["keep_alive"] = keep_alive

        with _OLLAMA_LOCK:
            ollama_key = (effective_model or "", effective_base_url or "")
            if _OLLAMA_ACTIVE_LM is not None:
                if _OLLAMA_ACTIVE_KEY != ollama_key:
                    if not _OLLAMA_WARNED_SWITCH:
                        logger.info(
                            "Reusing existing Ollama model %s (requested %s). Set OLLAMA_ALLOW_SWITCH=1 to permit hot swaps.",
                            _OLLAMA_ACTIVE_KEY[0] if _OLLAMA_ACTIVE_KEY else "",
                            effective_model,
                        )
                        _OLLAMA_WARNED_SWITCH = True
                    if not _truthy("OLLAMA_ALLOW_SWITCH"):
                        if _OLLAMA_ACTIVE_LM is not None:
                            _configure_dspy(_OLLAMA_ACTIVE_LM)
                        return _OLLAMA_ACTIVE_LM
                else:
                    if _OLLAMA_ACTIVE_LM is not None:
                        _configure_dspy(_OLLAMA_ACTIVE_LM)
                    return _OLLAMA_ACTIVE_LM
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
    timeout_default = 180 if provider == "ollama" else 60
    timeout_seconds = _env_float("DSPY_LM_TIMEOUT", float(timeout_default))
    if provider == "ollama":
        timeout_seconds = _env_float("DSPY_OLLAMA_TIMEOUT", timeout_seconds)
    else:
        timeout_seconds = _env_float("DSPY_OPENAI_TIMEOUT", timeout_seconds)

    retries_default = _env_int("DSPY_LM_MAX_RETRIES", 3)
    num_retries_default = _env_int("DSPY_LM_NUM_RETRIES", retries_default)

    _clamp_litellm_timeout(int(timeout_seconds))
    lm_kwargs.update({
        "timeout": timeout_seconds,
        "request_timeout": timeout_seconds,
        "max_retries": retries_default,
        "num_retries": num_retries_default,
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
    
    if cache_key is None:
        cache_key = (provider, provider_model, effective_base_url, effective_api_key, tuple(sorted(lm_kwargs.items())))

    cached = _LM_CACHE.get(cache_key)
    if cached is not None:
        _configure_dspy(cached)
        return cached

    lm = dspy.LM(
        model=provider_model,
        **lm_kwargs,
    )
    _configure_dspy(lm)
    _LM_CACHE[cache_key] = lm

    if provider == "ollama":
        with _OLLAMA_LOCK:
            _OLLAMA_ACTIVE_LM = lm
            _OLLAMA_ACTIVE_KEY = (effective_model or "", effective_base_url or "")

    return lm


class temporary_lm:
    """Context manager to temporarily set the active DSPy LM.

    Example:
        fast = configure_lm(provider="ollama", model_name="qwen2:0.5b")
        with temporary_lm(fast):
            # Calls inside use the fast LM
            ...
        # Restores previous LM
    """
    def __init__(self, lm: Optional[dspy.LM]) -> None:
        self._lm = lm
        self._prev = None

    def __enter__(self):
        try:
            self._prev = getattr(dspy.settings, 'lm', None)
        except Exception:
            self._prev = None
        try:
            if self._lm is not None:
                dspy.settings.configure(lm=self._lm)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._prev is not None:
                dspy.settings.configure(lm=self._prev)
        except Exception:
            pass
        return False


class LLMProvider:
    """Minimal LLM provider used by tests.

    Provides a `generate(prompt)` method and an overridable `_make_request` used
    by tests to simulate network failures.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        # May return None in local/no-LM mode — callers handle that.
        self._lm = configure_lm(provider=provider, model_name=model, base_url=base_url, api_key=api_key)

    def _make_request(self, prompt: str) -> Optional[str]:
        # If no LM configured, behave as a no-op returning None
        if self._lm is None:
            return None
        try:
            # Basic DSPy usage: create a trivial signature on the fly.
            # To avoid strict dependencies, fallback to a simple echo-style response.
            return None
        except Exception:
            return None

    def generate(self, prompt: str) -> Optional[str]:
        try:
            return self._make_request(prompt)
        except Exception as e:  # pragma: no cover - exercised in tests via patch
            return f"error: {e}"
