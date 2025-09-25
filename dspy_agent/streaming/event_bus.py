from __future__ import annotations

import json
import os
from typing import Any, Optional, Dict, Deque, Tuple, List
from pathlib import Path
from collections import deque

from .kafka_log import KafkaLogger, get_kafka_logger


class EventBus:
    def __init__(self, kafka: Optional[KafkaLogger] = None) -> None:
        self.kafka = kafka or get_kafka_logger()
        # RedDB fallback
        try:
            from ..db.enhanced_storage import get_enhanced_data_manager  # type: ignore
            self.dm = get_enhanced_data_manager()
        except Exception:
            self.dm = None
        self.log_dir = Path(os.getenv('EVENTBUS_LOG_DIR', str(Path.cwd() / 'logs')))

    def publish(self, topic: str, value: Any) -> None:
        record = value
        try:
            _append_memory(topic, record)
        except Exception:
            pass
        # Kafka path
        if self.kafka is not None and self.kafka.enabled:
            try:
                self.kafka.send(topic, record)
            except Exception:
                pass
        # RedDB fallback streams
        if self.dm is not None:
            try:
                self.dm.storage.append(topic, record)  # type: ignore[attr-defined]
            except Exception:
                pass
        # File log fallback
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            p = self.log_dir / f"{topic.replace('.', '_')}.jsonl"
            with p.open('a') as fh:
                fh.write(json.dumps(record, default=str) + '\n')
        except Exception:
            pass


def get_event_bus() -> EventBus:
    return EventBus()


# ------------------------
# In-memory event ring
# ------------------------

_RING_SIZE = int(os.getenv('EVENTBUS_RING_SIZE', '2000'))
_ring: Dict[str, Deque[Tuple[int, Any]]] = {}
_seq: Dict[str, int] = {}


def _append_memory(topic: str, rec: Any) -> int:
    dq = _ring.get(topic)
    if dq is None:
        dq = deque(maxlen=max(100, _RING_SIZE))
        _ring[topic] = dq
        _seq[topic] = 0
    _seq[topic] = int(_seq.get(topic, 0)) + 1
    s = _seq[topic]
    dq.append((s, rec))
    return s


def memory_topics() -> List[str]:
    return sorted(list(_ring.keys()))


def memory_last_seq(topic: str) -> int:
    return int(_seq.get(topic, 0))


def memory_tail(topic: str, limit: int = 100) -> List[Dict[str, Any]]:
    dq = _ring.get(topic)
    if not dq:
        return []
    n = max(1, min(limit, len(dq)))
    return [rec for (_, rec) in list(dq)[-n:]]


def memory_delta(topic: str, since_seq: int, max_items: int = 1000) -> Tuple[List[Dict[str, Any]], int]:
    dq = _ring.get(topic)
    if not dq:
        return [], since_seq
    out: List[Dict[str, Any]] = []
    last = since_seq
    for (s, rec) in dq:
        if s > since_seq:
            out.append(rec)
            last = s
            if len(out) >= max(1, max_items):
                break
    return out, last

