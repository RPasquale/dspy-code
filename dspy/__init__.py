"""
Lightweight local stub of the `dspy` package used for testing and offline runs.

This stub provides minimal classes and functions referenced by dspy_agent:
 - Signature, Module, InputField, OutputField, Predict
 - LM and configure(lm=..., adapter=...)
 - Subpackages: adapters.json_adapter.JSONAdapter, clients.lm.LM, signatures.signature.Signature

If the real `dspy` is installed, prefer that in production. This stub is only to
keep local development workflows unblocked when DSPy isn't available.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

IS_STUB = True

class Signature:
    pass


class Module:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def InputField(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
    return None


def OutputField(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
    return None


class _DefaultPredict:
    def __init__(self, signature: type[Signature]) -> None:
        self.signature = signature

    def __call__(self, **kwargs: Any) -> Any:
        # Return an object with common fields expected by our skills
        defaults = {
            "context": "",
            "key_points": "",
            "missing_info": "",
            "next_steps": "",
            "summary": "",
            "bullets": "",
            "entry_points": "",
            "risk_areas": "",
            "plan": "",
            "commands": "",
            "assumptions": "",
            "risks": "",
        }
        return SimpleNamespace(**defaults)


def Predict(signature: type[Signature]) -> _DefaultPredict:  # type: ignore[override]
    return _DefaultPredict(signature)


# Some code prefers ChainOfThought; map to Predict in the stub
ChainOfThought = Predict  # type: ignore


class LM:  # Simple placeholder
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def configure(*, lm: Any = None, adapter: Any = None) -> None:
    # No-op configuration hook for the stub
    return


# Subpackages: adapters.json_adapter, clients.lm, signatures.signature
# These modules are provided so imports like `from dspy.adapters.json_adapter import JSONAdapter`
# and `from dspy.clients.lm import LM` succeed.

# adapters.json_adapter
class JSONAdapter:
    def __init__(self, callbacks: Any | None = None, use_native_function_calling: bool | None = None) -> None:
        self.callbacks = callbacks
        self.use_native_function_calling = use_native_function_calling

    def __call__(self, lm: LM, lm_kwargs: dict, signature: type[Signature], demos: list[dict], inputs: dict) -> list[dict]:
        # Return a trivial JSON-like response envelope
        return [{"args_json": "{}"}]

    async def acall(self, lm: LM, lm_kwargs: dict, signature: type[Signature], demos: list[dict], inputs: dict) -> list[dict]:
        return [{"args_json": "{}"}]


# Expose nested modules for import paths used by the repo
import sys as _sys
import types as _types

# dspy.adapters
_adapters = _types.ModuleType("dspy.adapters")
_json_adapter = _types.ModuleType("dspy.adapters.json_adapter")
_json_adapter.JSONAdapter = JSONAdapter
_adapters.json_adapter = _json_adapter  # type: ignore[attr-defined]
_sys.modules["dspy.adapters"] = _adapters
_sys.modules["dspy.adapters.json_adapter"] = _json_adapter

# dspy.clients
_clients = _types.ModuleType("dspy.clients")
_lm_mod = _types.ModuleType("dspy.clients.lm")
_lm_mod.LM = LM
_clients.lm = _lm_mod  # type: ignore[attr-defined]
_sys.modules["dspy.clients"] = _clients
_sys.modules["dspy.clients.lm"] = _lm_mod

# dspy.signatures
_sigs = _types.ModuleType("dspy.signatures")
_sig_mod = _types.ModuleType("dspy.signatures.signature")
_sig_mod.Signature = Signature
_sigs.signature = _sig_mod  # type: ignore[attr-defined]
_sys.modules["dspy.signatures"] = _sigs
_sys.modules["dspy.signatures.signature"] = _sig_mod
