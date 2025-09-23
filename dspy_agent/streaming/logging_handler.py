from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional


def _make_producer(bootstrap: str):
    try:
        from confluent_kafka import Producer  # type: ignore
        p = Producer({'bootstrap.servers': bootstrap})
        class _P:
            def send(self, topic: str, value: dict):
                p.produce(topic, json.dumps(value, default=str).encode('utf-8'))
            def flush(self):
                p.flush()
        return _P()
    except Exception:
        pass
    try:
        from kafka import KafkaProducer  # type: ignore
        return KafkaProducer(bootstrap_servers=bootstrap, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    except Exception:  # pragma: no cover
        return None


class DSPyKafkaHandler(logging.Handler):
    """Logging handler to forward records to Kafka as JSON.

    Usage:
        handler = DSPyKafkaHandler(container='app', bootstrap='localhost:9092')
        logging.getLogger().addHandler(handler)
    """
    def __init__(
        self,
        container: str = 'app',
        *,
        bootstrap: Optional[str] = None,
        topic_prefix: str = 'logs.raw.',
        level: int | None = None,
    ) -> None:
        super().__init__(level or logging.INFO)
        self.container = container
        self.bootstrap = bootstrap or os.getenv('KAFKA_BOOTSTRAP_SERVERS') or os.getenv('KAFKA_BOOTSTRAP') or ''
        self.topic = f"{topic_prefix}{container}" if topic_prefix else container
        self._prod = _make_producer(self.bootstrap) if self.bootstrap else None

    def emit(self, record: logging.LogRecord) -> None:
        if not self._prod:
            return
        try:
            payload: dict[str, Any] = {
                'level': record.levelname,
                'logger': record.name,
                'message': self.format(record) if self.formatter else record.getMessage(),
                'pathname': record.pathname,
                'lineno': record.lineno,
                'module': record.module,
                'funcName': record.funcName,
                'ts': time.time(),
            }
            if record.exc_info:
                try:
                    import traceback
                    payload['exc'] = ''.join(traceback.format_exception(*record.exc_info))
                except Exception:
                    pass
            self._prod.send(self.topic, payload)
        except Exception:
            # Do not break application logs on errors
            try:
                self.handleError(record)
            except Exception:
                pass

