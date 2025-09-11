from __future__ import annotations

from typing import Optional

from ..config import get_settings
from .base import Storage


def get_storage() -> Optional[Storage]:
    """Return a configured storage instance or None.

    Defaults to RedDB when DB_BACKEND=reddb or REDDB_URL is set.
    """
    s = get_settings()
    if s.db_backend.lower() == "reddb":
        try:
            from .reddb import RedDBStorage  # lazy import
        except Exception:
            # If import fails, return a stub in-memory adapter (via class import still ok)
            from .reddb import RedDBStorage  # type: ignore
        return RedDBStorage(url=s.reddb_url, namespace=s.reddb_namespace or "dspy")
    return None

