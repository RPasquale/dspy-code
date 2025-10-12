from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import re
import time
import random
from pathlib import Path

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature


_EVENT_LOG = Path('.dspy_reports') / 'reliability.jsonl'


def _log_event(kind: str, data: Dict[str, Any]) -> None:
    try:
        _EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "kind": kind, **data}
        with _EVENT_LOG.open('a') as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


class _CircuitBreaker:
    """Lightweight circuit breaker with exponential backoff and jitter.

    - Opens after `fail_threshold` consecutive failures.
    - Cooldown grows as base_seconds * 2**streak + uniform(0, jitter).
    - Half-open trial after cooldown; success closes, failure re-opens.
    """

    def __init__(self, fail_threshold: int = 3, base_seconds: float = 2.0, jitter: float = 1.0) -> None:
        self.fail_threshold = int(fail_threshold)
        self.base_seconds = float(base_seconds)
        self.jitter = float(jitter)
        self._consec_fail = 0
        self._opened_until = 0.0
        self._half_open = False

    def allow(self) -> bool:
        now = time.time()
        if now < self._opened_until:
            return False
        # Past cooldown: allow a half-open attempt
        return True

    def on_success(self) -> None:
        self._consec_fail = 0
        self._opened_until = 0.0
        self._half_open = False

    def on_failure(self, err: Optional[str] = None) -> None:
        self._consec_fail += 1
        if self._consec_fail >= self.fail_threshold:
            cooldown = self.base_seconds * (2 ** max(0, self._consec_fail - self.fail_threshold))
            cooldown += random.uniform(0.0, self.jitter)
            self._opened_until = time.time() + cooldown
            self._half_open = True
            _log_event("circuit_open", {"cooldown_sec": cooldown, "failures": self._consec_fail, "error": (err or "")[:160]})

    @property
    def is_open(self) -> bool:
        return time.time() < self._opened_until


_CB = _CircuitBreaker()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to extract and parse the largest JSON object from `text`.

    Tries in order: strict json, fenced blocks, greedy brace matching, regex.
    Returns dict on success, else None.
    """
    if not text:
        return None
    # 1) strict parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) code fences ```json ... ``` or ``` ... ```
    fences = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    for blk in fences:
        try:
            obj = json.loads(blk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    # 3) greedy brace search
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = text[start_idx:end_idx + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 4) regex minimal object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _ensure_args_json_string(val: Any) -> str:
    """Return a safe JSON string for args_json field.

    Accepts dict/str/None. Sanitizes malformed strings by extracting inner JSON.
    Always returns a JSON object string (e.g., "{}" on failure).
    """
    # If already a mapping, dump directly.
    if isinstance(val, dict):
        try:
            return json.dumps(val)
        except Exception:
            return "{}"
    # If value looks like string, attempt robust parsing and re-dump
    s = str(val or "").strip()
    if not s:
        return "{}"
    obj = _extract_json_object(s)
    if isinstance(obj, dict):
        try:
            return json.dumps(obj)
        except Exception:
            return "{}"
    return "{}"


class StrictJSONAdapter(JSONAdapter):
    """A JSON adapter that always uses simple json_object response format.

    DSPy's default JSONAdapter first attempts "structured outputs" (JSON Schema)
    when the provider advertises support, then falls back to json_object with a
    warning if that fails. Some OpenAI-compatible runtimes (e.g., Ollama) do not
    implement structured outputs fully, which leads to noisy warnings.

    This adapter skips the structured-output attempt entirely and always uses
    {"type": "json_object"}, avoiding the warning while preserving robust JSON
    parsing behavior.
    """

    def __init__(self):
        # Disable native function-calling to keep things simple across providers.
        super().__init__(callbacks=None, use_native_function_calling=False)

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        # Force simple JSON object mode; skip structured outputs completely.
        lm_kwargs = dict(lm_kwargs)
        lm_kwargs["response_format"] = {"type": "json_object"}
        model_id = getattr(lm, "model", "")
        if isinstance(model_id, str) and model_id.startswith("ollama/"):
            lm_kwargs.setdefault("format", "json")
        
        # Add better error handling for empty responses
        # Circuit breaker: short-circuit if open
        if not _CB.allow():
            _log_event("adapter_short_circuit", {"reason": "circuit_open"})
            return [self._fallback(signature, reason="circuit_open")]
        try:
            result = super(JSONAdapter, self).__call__(lm, lm_kwargs, signature, demos, inputs)
            # If result is empty or malformed, provide a fallback
            if not result or (isinstance(result, list) and len(result) == 0):
                _CB.on_failure("empty_result")
                _log_event("adapter_empty", {"signature": getattr(signature, '__name__', 'unknown')})
                return [self._fallback(signature, reason="empty_result")]
            # Sanitize args_json field on each item
            cleaned: List[Dict[str, Any]] = []
            for item in result:
                if not isinstance(item, dict):
                    continue
                fixed = dict(item)
                if "args_json" in fixed:
                    fixed["args_json"] = _ensure_args_json_string(fixed.get("args_json"))
                cleaned.append(fixed)
            _CB.on_success()
            return cleaned if cleaned else [self._fallback(signature, reason="malformed_items")]
        except Exception as e:
            _CB.on_failure(str(e))
            _log_event("adapter_error", {"error": str(e)[:200], "signature": getattr(signature, '__name__', 'unknown')})
            return [self._fallback(signature, reason=f"error:{str(e)[:80]}")]

    async def acall(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        # Force simple JSON object mode; skip structured outputs completely.
        lm_kwargs = dict(lm_kwargs)
        lm_kwargs["response_format"] = {"type": "json_object"}
        
        # Add better error handling for empty responses
        if not _CB.allow():
            _log_event("adapter_short_circuit", {"reason": "circuit_open"})
            return [self._fallback(signature, reason="circuit_open")]
        try:
            result = await super(JSONAdapter, self).acall(lm, lm_kwargs, signature, demos, inputs)
            # If result is empty or malformed, provide a fallback
            if not result or (isinstance(result, list) and len(result) == 0):
                _CB.on_failure("empty_result")
                _log_event("adapter_empty", {"signature": getattr(signature, '__name__', 'unknown')})
                return [self._fallback(signature, reason="empty_result")]
            cleaned: List[Dict[str, Any]] = []
            for item in result:
                if not isinstance(item, dict):
                    continue
                fixed = dict(item)
                if "args_json" in fixed:
                    fixed["args_json"] = _ensure_args_json_string(fixed.get("args_json"))
                cleaned.append(fixed)
            _CB.on_success()
            return cleaned if cleaned else [self._fallback(signature, reason="malformed_items")]
        except Exception as e:
            _CB.on_failure(str(e))
            _log_event("adapter_error", {"error": str(e)[:200], "signature": getattr(signature, '__name__', 'unknown')})
            return [self._fallback(signature, reason=f"error:{str(e)[:80]}")]

    # Helpers -------------------------------------------------------------
    def _fallback(self, signature: type[Signature], *, reason: str = "") -> Dict[str, Any]:
        """Create a minimal, schema-aware fallback object with intelligent defaults."""
        fallback: Dict[str, Any] = {}
        
        # Get signature name for context-aware fallbacks
        sig_name = getattr(signature, '__name__', 'unknown').lower()
        
        for field_name, field_info in signature.output_fields.items():
            if field_name == "tool":
                # Provide context-aware tool fallbacks
                if "edit" in sig_name or "patch" in sig_name:
                    fallback[field_name] = "plan"
                elif "search" in sig_name or "grep" in sig_name:
                    fallback[field_name] = "grep"
                elif "orchestrate" in sig_name or "controller" in sig_name:
                    fallback[field_name] = "respond"
                else:
                    fallback[field_name] = "respond"
            elif field_name == "args_json":
                fallback[field_name] = "{}"
            elif field_name == "rationale":
                if "circuit_open" in reason:
                    fallback[field_name] = "LLM service temporarily unavailable. Using fallback strategy."
                elif "empty_result" in reason:
                    fallback[field_name] = "LLM returned empty response. Using fallback strategy."
                else:
                    fallback[field_name] = f"Fallback response ({reason})" if reason else "Fallback response"
            elif field_name in ["context", "key_points", "missing_info", "next_steps"]:
                # Context builder specific fallbacks
                if field_name == "context":
                    fallback[field_name] = "Unable to process request due to LLM issues. Please try again."
                elif field_name == "key_points":
                    fallback[field_name] = "- LLM service temporarily unavailable\n- Please retry the request"
                elif field_name == "missing_info":
                    fallback[field_name] = "- Need LLM service to be available\n- Check system status"
                elif field_name == "next_steps":
                    fallback[field_name] = "- Wait for LLM service to recover\n- Retry the request\n- Check system logs"
            elif field_name in ["action", "plan", "summary"]:
                # General action fallbacks
                fallback[field_name] = "Unable to process request. Please try again."
            else:
                # Default fallback for unknown fields
                fallback[field_name] = ""
        
        return fallback
