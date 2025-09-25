"""
Lightweight, in-repo shim for `diskcache` to avoid SQLite PRAGMA errors
in constrained environments (e.g., sandboxed macOS seatbelt) where the
real `diskcache.FanoutCache` cannot initialize.

This shim provides a minimal, dict-like `FanoutCache` API used by the
`dspy` library for on-disk caching. It stores entries purely in-memory
for the current process and ignores disk-related parameters.

Only methods and behaviors needed by our usage are implemented:
- `__contains__`, `__getitem__`, `__setitem__`, `get`
- `close()` (no-op)

If your environment supports the real `diskcache`, remove this module
to restore normal behavior.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator


class FanoutCache:
    def __init__(
        self,
        shards: int | None = None,
        timeout: int | float | None = None,
        directory: str | None = None,
        size_limit: int | None = None,
        **_: Any,
    ) -> None:
        # Store data in-memory only; ignore disk params.
        self._store: dict[Any, Any] = {}

    # Mapping protocol (minimal)
    def __contains__(self, key: Any) -> bool:  # type: ignore[override]
        return key in self._store

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        return self._store[key]

    def __setitem__(self, key: Any, value: Any) -> None:  # type: ignore[override]
        self._store[key] = value

    def get(self, key: Any, default: Any = None) -> Any:
        return self._store.get(key, default)

    def close(self) -> None:
        # No-op for in-memory shim
        pass

    # Optional helpers to behave more like a Mapping (not strictly required)
    def keys(self) -> Iterable[Any]:
        return self._store.keys()

    def items(self) -> Iterable[tuple[Any, Any]]:
        return self._store.items()

    def values(self) -> Iterable[Any]:
        return self._store.values()

