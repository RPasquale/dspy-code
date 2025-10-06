"""DSPy import proxy.

If the real package is available (e.g., installed by uv), load it. Otherwise fall
back to a minimal local stub used for offline development.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _truthy(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


_REAL_DSPY = None
if not _truthy(os.getenv("DSPY_FORCE_STUB"), False):
    try:
        this_file = Path(__file__).resolve()
        workspace_root = this_file.parent.parent
        filtered_path: list[str] = []
        for entry in sys.path:
            try:
                if Path(entry).resolve() == workspace_root:
                    continue
            except Exception:
                pass
            filtered_path.append(entry)
        spec = importlib.machinery.PathFinder.find_spec("dspy", filtered_path)
        if spec and spec.loader and spec.origin and Path(spec.origin) != this_file:
            module = importlib.util.module_from_spec(spec)
            sys.modules[__name__] = module
            spec.loader.exec_module(module)
            _REAL_DSPY = module
    except Exception:
        _REAL_DSPY = None

if _REAL_DSPY is not None:
    globals().update(_REAL_DSPY.__dict__)
    if not hasattr(sys.modules[__name__], "IS_STUB"):
        setattr(sys.modules[__name__], "IS_STUB", False)
else:
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


    ChainOfThought = Predict  # type: ignore


    class LM:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs


    def configure(*, lm: Any = None, adapter: Any = None) -> None:
        return


    import types as _types

    _adapters = _types.ModuleType("dspy.adapters")
    _json_adapter = _types.ModuleType("dspy.adapters.json_adapter")

    class JSONAdapter:
        def __init__(self, callbacks: Any | None = None, use_native_function_calling: bool | None = None) -> None:
            self.callbacks = callbacks
            self.use_native_function_calling = use_native_function_calling

        def __call__(self, lm: LM, lm_kwargs: dict, signature: type[Signature], demos: list[dict], inputs: dict) -> list[dict]:
            return [{"args_json": "{}"}]

        async def acall(self, lm: LM, lm_kwargs: dict, signature: type[Signature], demos: list[dict], inputs: dict) -> list[dict]:
            return [{"args_json": "{}"}]

    _json_adapter.JSONAdapter = JSONAdapter
    _adapters.json_adapter = _json_adapter  # type: ignore[attr-defined]
    sys.modules["dspy.adapters"] = _adapters
    sys.modules["dspy.adapters.json_adapter"] = _json_adapter

    _clients = _types.ModuleType("dspy.clients")
    _lm_mod = _types.ModuleType("dspy.clients.lm")
    _lm_mod.LM = LM
    _clients.lm = _lm_mod  # type: ignore[attr-defined]
    sys.modules["dspy.clients"] = _clients
    sys.modules["dspy.clients.lm"] = _lm_mod

    _sigs = _types.ModuleType("dspy.signatures")
    _sig_mod = _types.ModuleType("dspy.signatures.signature")
    _sig_mod.Signature = Signature
    _sigs.signature = _sig_mod  # type: ignore[attr-defined]
    sys.modules["dspy.signatures"] = _sigs
    sys.modules["dspy.signatures.signature"] = _sig_mod

    __all__ = [
        "ChainOfThought",
        "InputField",
        "IS_STUB",
        "JSONAdapter",
        "LM",
        "Module",
        "OutputField",
        "Predict",
        "Signature",
        "configure",
    ]
