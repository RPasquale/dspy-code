from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

from ..streaming.kafka_log import get_kafka_logger


class CodeWatcher:
    def __init__(
        self,
        root: Path,
        interval: float = 2.0,
        *,
        container: str = "workspace",
        topic_prefix: str = "logs.raw.",
        snippet_bytes: int = 2048,
        snippet_lines: int = 20,
    ) -> None:
        self.root = root
        self.interval = interval
        self.container = container
        self.topic_prefix = topic_prefix
        self.snippet_bytes = max(256, snippet_bytes)
        self.snippet_lines = max(1, snippet_lines)
        self._mtimes: Dict[str, float] = {}
        self.kafka = get_kafka_logger()

        logs_dir = Path(os.getenv("DSPY_LOGS") or (self.root / "logs"))
        self._log_file: Optional[Path] = None
        try:
            target = logs_dir / container / "workspace_changes.log"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=True)
            self._log_file = target
        except Exception:
            self._log_file = None

    def _iter_files(self):
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in {'.pyc', '.pyo', '.db', '.sqlite', '.shm', '.wal'}:
                continue
            parts = p.parts
            if any(seg in {".git", ".venv", "node_modules", "dist", "build", ".dspy_index", "__pycache__"} for seg in parts):
                continue
            yield p

    def _tail_snippet(self, path: Path) -> str:
        try:
            size = path.stat().st_size
            read_bytes = min(size, self.snippet_bytes)
            with path.open("rb") as handle:
                if read_bytes < size:
                    handle.seek(-read_bytes, os.SEEK_END)
                data = handle.read(read_bytes)
            text = data.decode("utf-8", errors="ignore")
            lines = text.splitlines()[-self.snippet_lines :]
            return "\n".join(lines)
        except Exception:
            return ""

    def _record_event(self, event: dict) -> None:
        if self.kafka is not None:
            try:
                self.kafka.send("code.fs.events", event)
            except Exception:
                pass
            try:
                raw_topic = f"{self.topic_prefix}{self.container}" if self.topic_prefix else self.container
                self.kafka.send(raw_topic, event)
            except Exception:
                pass
        if self._log_file is not None:
            try:
                with self._log_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def scan_once(self):
        now = time.time()
        for p in self._iter_files():
            try:
                stat = p.stat()
            except Exception:
                continue
            m = stat.st_mtime
            key = str(p.resolve())
            old = self._mtimes.get(key)
            if old is None:
                self._mtimes[key] = m
                continue
            if m <= old:
                continue
            self._mtimes[key] = m
            try:
                rel = str(p.relative_to(self.root))
            except Exception:
                rel = key
            event = {
                "path": rel,
                "abs_path": key,
                "event": "modified",
                "mtime": m,
                "size": getattr(stat, "st_size", 0),
                "ts": now,
            }
            snippet = self._tail_snippet(p)
            if snippet:
                event["snippet"] = snippet
            self._record_event(event)

    def run(self):
        while True:
            self.scan_once()
            time.sleep(self.interval)

