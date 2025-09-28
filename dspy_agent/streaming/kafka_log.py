from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Optional

# Do not import confluent_kafka at module import time to avoid native crashes
Producer = None  # type: ignore

_DISABLE_VALUES = {"1", "true", "yes"}


class KafkaLogger:
    def __init__(self, bootstrap: Optional[str] = None, client_id: Optional[str] = None, topic_prefix: str = "") -> None:
        env_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS") or os.getenv("KAFKA_BOOTSTRAP") or ""
        # Auto-detect if we're running in Docker network
        if not env_bootstrap and not bootstrap:
            # Check if we're in a Docker container
            if os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER"):
                env_bootstrap = "kafka:9092"  # Use internal Docker network
            else:
                env_bootstrap = "localhost:9092"  # Use host network
        self.bootstrap = env_bootstrap if bootstrap is None else bootstrap
        self.client_id = client_id or os.getenv("KAFKA_CLIENT_ID") or "dspy-agent"
        self.topic_prefix = topic_prefix or os.getenv("KAFKA_TOPIC_PREFIX", "")
        self._producer = None
        self._retry_backoff = max(1.0, self._env_float("DSPY_KAFKA_RETRY_BACKOFF", 5.0))
        self._connect_timeout = max(0.05, self._env_float("DSPY_KAFKA_CONNECT_TIMEOUT", 0.25))
        self._next_retry_at = 0.0
        disable_flag = os.getenv("DSPY_DISABLE_KAFKA", "0").lower()
        self._should_attempt = bool(self.bootstrap) and disable_flag not in _DISABLE_VALUES
        if self._should_attempt:
            self._maybe_init_producer(initial=True)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if not raw:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    @property
    def enabled(self) -> bool:
        return self._should_attempt

    def _maybe_init_producer(self, *, initial: bool = False) -> None:
        if not self._should_attempt or self._producer is not None:
            return
        now = time.monotonic()
        if not initial and now < self._next_retry_at:
            return
        if not self._bootstrap_available():
            self._schedule_retry(now)
            return
        try:
            from confluent_kafka import Producer as _Producer  # type: ignore
        except Exception:
            # Library unavailable; do not keep retrying
            self._should_attempt = False
            return
        conf = {
            "bootstrap.servers": self.bootstrap, 
            "client.id": self.client_id,
            "socket.timeout.ms": 10000,
            "api.version.request.timeout.ms": 10000,
            "broker.address.family": "v4"
        }
        log_level = os.getenv("DSPY_KAFKA_LOG_LEVEL")
        if log_level is not None:
            try:
                conf["log_level"] = int(log_level)
            except (TypeError, ValueError):
                pass
        try:
            self._producer = _Producer(conf)
            self._next_retry_at = 0.0
        except Exception:
            self._producer = None
            self._schedule_retry(now)

    def _schedule_retry(self, now: Optional[float] = None) -> None:
        base = time.monotonic() if now is None else now
        self._next_retry_at = base + self._retry_backoff

    def _bootstrap_available(self) -> bool:
        if not self.bootstrap:
            return False
        for host, port in self._iter_endpoints():
            try:
                with socket.create_connection((host, port), timeout=self._connect_timeout):
                    return True
            except OSError:
                continue
        return False

    def _iter_endpoints(self) -> list[tuple[str, int]]:
        endpoints: list[tuple[str, int]] = []
        for token in (self.bootstrap or "").split(","):
            token = token.strip()
            if not token:
                continue
            if "://" in token:
                token = token.split("://", 1)[1]
            host = token
            port = 9092
            if host.startswith("[") and "]" in host:
                idx = host.index("]")
                raw_host = host[1:idx]
                remainder = host[idx + 1 :]
                if remainder.startswith(":"):
                    port = self._safe_port(remainder[1:])
                host = raw_host
            elif host.count(":") == 1 and not host.endswith("]"):
                base, raw_port = host.split(":", 1)
                host = base
                port = self._safe_port(raw_port)
            endpoints.append((host or "localhost", port))
        return endpoints

    @staticmethod
    def _safe_port(raw: str) -> int:
        try:
            value = int(raw)
            if 0 < value < 65536:
                return value
        except Exception:
            pass
        return 9092

    def _close_producer(self) -> None:
        prod = self._producer
        self._producer = None
        if prod is None:
            return
        try:
            prod.flush(0)
        except Exception:
            pass

    def _name(self, topic: str) -> str:
        return f"{self.topic_prefix}{topic}" if self.topic_prefix else topic

    def send(self, topic: str, value: Any) -> None:
        if not self.enabled:
            return
        if self._producer is None:
            self._maybe_init_producer()
            if self._producer is None:
                return
        try:
            data = json.dumps(value, default=str).encode("utf-8")
        except Exception:
            data = str(value).encode("utf-8")
        try:
            self._producer.produce(self._name(topic), data)  # type: ignore[union-attr]
            self._producer.poll(0)  # type: ignore[union-attr]
        except Exception:
            self._close_producer()
            self._schedule_retry()

    # Backwards-compatible alias used in some scripts
    def publish(self, topic: str, value: Any) -> None:  # pragma: no cover - simple alias
        self.send(topic, value)


def get_kafka_logger() -> Optional[KafkaLogger]:
    kl = KafkaLogger()
    return kl if kl.enabled else None
