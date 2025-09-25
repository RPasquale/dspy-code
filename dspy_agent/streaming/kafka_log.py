from __future__ import annotations

import json
import os
from typing import Any, Optional

# Do not import confluent_kafka at module import time to avoid native crashes
Producer = None  # type: ignore


class KafkaLogger:
    def __init__(self, bootstrap: Optional[str] = None, client_id: Optional[str] = None, topic_prefix: str = "") -> None:
        self.bootstrap = bootstrap or os.getenv("KAFKA_BOOTSTRAP_SERVERS") or os.getenv("KAFKA_BOOTSTRAP") or ""
        self.client_id = client_id or os.getenv("KAFKA_CLIENT_ID") or "dspy-agent"
        self.topic_prefix = topic_prefix or os.getenv("KAFKA_TOPIC_PREFIX", "")
        self._producer = None
        # Lazy import: only attempt if bootstrap configured and not explicitly disabled
        if self.bootstrap and os.getenv("DSPY_DISABLE_KAFKA", "0").lower() not in {"1", "true", "yes"}:
            try:
                # Import within guarded block; failure should not crash process
                from confluent_kafka import Producer as _Producer  # type: ignore
                conf = {"bootstrap.servers": self.bootstrap, "client.id": self.client_id}
                self._producer = _Producer(conf)
            except Exception:
                self._producer = None

    @property
    def enabled(self) -> bool:
        return self._producer is not None

    def _name(self, topic: str) -> str:
        return f"{self.topic_prefix}{topic}" if self.topic_prefix else topic

    def send(self, topic: str, value: Any) -> None:
        if not self.enabled:
            return
        try:
            data = json.dumps(value, default=str).encode("utf-8")
        except Exception:
            # Best-effort fallback to str
            data = str(value).encode("utf-8")
        try:
            self._producer.produce(self._name(topic), data)
            self._producer.poll(0)
        except Exception:
            # Swallow errors to avoid impacting main flow
            pass

    # Backwards-compatible alias used in some scripts
    def publish(self, topic: str, value: Any) -> None:  # pragma: no cover - simple alias
        self.send(topic, value)


def get_kafka_logger() -> Optional[KafkaLogger]:
    kl = KafkaLogger()
    return kl if kl.enabled else None
