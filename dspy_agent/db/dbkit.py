from __future__ import annotations

from typing import Dict, Any, List


class DatabaseKit:
    """Lightweight, in-memory database kit for tests.

    Provides connect/store/retrieve/search/disconnect APIs.
    """

    def __init__(self) -> None:
        self._connected = False
        self._data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def store(self, collection: str, item: Dict[str, Any]) -> None:
        cid = str(item.get("id") or item.get("_id") or "")
        if not cid:
            # Generate a simple ID if missing
            cid = str(len(self._data.get(collection, {})) + 1)
            item = {**item, "id": cid}
        self._data.setdefault(collection, {})[cid] = dict(item)

    def retrieve(self, collection: str, id: str) -> Dict[str, Any] | None:
        return (self._data.get(collection) or {}).get(str(id))

    def search(self, collection: str, query: str) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        items = list((self._data.get(collection) or {}).values())
        if not q:
            return items
        # Support simple key:value query; fallback to substring over whole item
        if ":" in q:
            key, _, val = q.partition(":")
            key = key.strip(); val = val.strip().lower()
            out: List[Dict[str, Any]] = []
            for it in items:
                v = it.get(key)
                if v is None:
                    continue
                if val in str(v).lower():
                    out.append(it)
            return out
        return [it for it in items if q.lower() in str(it).lower()]
