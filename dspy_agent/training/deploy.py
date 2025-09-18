from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from ..db.factory import get_storage
from ..streaming.kafka_log import get_kafka_logger
from .deploy_model import (
    DEPLOY_LOG_STREAM,
    DEPLOY_EVENT_STREAM,
    DEPLOY_LOG_TOPIC,
    DEPLOY_EVENT_TOPIC,
    KV_LAST_STATUS,
    KV_LAST_IMAGE,
    KV_LAST_COMPOSE_HASH,
    KV_LAST_TS,
    DeployEvent,
)


def _hash_file(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


class DeploymentLogger:
    def __init__(self, workspace: Optional[Path], name: str = "lightweight") -> None:
        self.workspace = workspace or Path.cwd()
        self.name = name
        self.log_dir = (self.workspace / "logs" / "deployments")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{name}_{ts}.log"
        self._fh = self.log_path.open("a", buffering=1)
        self.storage = get_storage()
        # Optional Kafka for deployment logs
        try:
            self.kafka = get_kafka_logger()
        except Exception:
            self.kafka = None
        self._status = "pending"

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def event(self, phase: str, level: str, message: str) -> None:
        rec = DeployEvent(ts=time.time(), phase=phase, level=level, message=message)
        line = f"[{rec.ts:.3f}] {phase} {level}: {message}\n"
        try:
            self._fh.write(line)
        except Exception:
            pass
        # Best-effort stream append
        try:
            if self.storage is not None:
                self.storage.append(DEPLOY_LOG_STREAM, asdict(rec))  # type: ignore[attr-defined]
        except Exception:
            pass
        # Best-effort Kafka publish
        try:
            if getattr(self, 'kafka', None) is not None:
                self.kafka.send(DEPLOY_LOG_TOPIC, asdict(rec))  # type: ignore[union-attr]
        except Exception:
            pass

    def status(self, status: str) -> None:
        self._status = status
        try:
            if self.storage is not None:
                self.storage.put(KV_LAST_STATUS, status)  # type: ignore[attr-defined]
                self.storage.put(KV_LAST_TS, time.time())  # type: ignore[attr-defined]
        except Exception:
            pass

    def set_image(self, image: str) -> None:
        try:
            if self.storage is not None:
                self.storage.put(KV_LAST_IMAGE, image)  # type: ignore[attr-defined]
        except Exception:
            pass

    def set_compose_hash(self, p: Path) -> None:
        h = _hash_file(p)
        try:
            if self.storage is not None and h:
                self.storage.put(KV_LAST_COMPOSE_HASH, h)  # type: ignore[attr-defined]
        except Exception:
            pass

    def run_stream(self, cmd: list[str], phase: str) -> int:
        self.event(phase, "info", f"$ {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            self.event(phase, "error", f"failed to start: {e}")
            return 1
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            self.event(phase, "info", line)
        code = proc.wait()
        if code != 0:
            self.event(phase, "error", f"exit code {code}")
        else:
            self.event(phase, "info", f"completed (exit {code})")
        return code
