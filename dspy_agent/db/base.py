from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol


class Storage(Protocol):
    """Minimal storage protocol for local streaming persistence.

    Implementations should be lightweight and safe to import when unavailable
    (i.e., only import heavy/optional deps inside constructors).
    """

    def put(self, key: str, value: Any) -> None:
        """Set a value for a key (idempotent upsert)."""

    def get(self, key: str) -> Optional[Any]:
        """Get a value for a key, or None if missing."""

    def delete(self, key: str) -> None:
        """Delete a key if present (no-op if missing)."""

    def append(self, stream: str, value: Any) -> None:
        """Append a value to a named stream/log (fire-and-forget)."""

    def read(self, stream: str, start: int = 0, count: int = 100) -> Iterable[tuple[int, Any]]:
        """Read a range from a stream, yielding (offset, value)."""

