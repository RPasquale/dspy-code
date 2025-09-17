from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

from ..streaming.kafka_log import get_kafka_logger


class CodeWatcher:
    def __init__(self, root: Path, interval: float = 2.0) -> None:
        self.root = root
        self.interval = interval
        self._mtimes: Dict[str, float] = {}
        self.kafka = get_kafka_logger()

    def _iter_files(self):
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            # Skip typical excludes
            parts = p.parts
            if any(seg in {".git", ".venv", "node_modules", "dist", "build", ".dspy_index"} for seg in parts):
                continue
            yield p

    def scan_once(self):
        now = time.time()
        for p in self._iter_files():
            try:
                m = p.stat().st_mtime
            except Exception:
                continue
            key = str(p.resolve())
            old = self._mtimes.get(key)
            if old is None:
                self._mtimes[key] = m
                continue
            if m > old:
                self._mtimes[key] = m
                evt = {"path": key, "event": "modified", "mtime": m, "ts": now}
                if self.kafka is not None:
                    self.kafka.send('code.fs.events', evt)

    def run(self):
        while True:
            self.scan_once()
            time.sleep(self.interval)

