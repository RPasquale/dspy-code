from __future__ import annotations

import json
from typing import Optional, Any

try:
    from confluent_kafka import Consumer, Producer  # type: ignore
except Exception:  # pragma: no cover
    Consumer = None  # type: ignore
    Producer = None  # type: ignore

from ..db.factory import get_storage


class RouterWorker:
    def __init__(self, bootstrap: Optional[str] = None, in_pattern: str = 'agent.results.*', metrics_topic: str = 'agent.metrics') -> None:
        # Allow local-only mode when Kafka not available or bootstrap not provided
        self.local_only = (bootstrap is None) or (Consumer is None) or (Producer is None)
        self.bootstrap = bootstrap or ''
        self.in_pattern = in_pattern
        self.metrics_topic = metrics_topic
        self.storage = get_storage()
        if not self.local_only:
            self.consumer = Consumer({
                'bootstrap.servers': self.bootstrap,
                'group.id': 'dspy-router',
                'auto.offset.reset': 'latest',
            })
            self.producer = Producer({'bootstrap.servers': self.bootstrap})
        else:
            self.consumer = None  # type: ignore
            self.producer = None  # type: ignore

    def run(self) -> None:
        if self.local_only:
            return
        # confluent-kafka doesn't support subscribe by pattern directly; list topics and subscribe
        md = self.consumer.list_topics(timeout=5.0)
        topics = [t for t in md.topics.keys() if t.startswith('agent.results.')]
        if not topics:
            topics = ['agent.results.app']
        self.consumer.subscribe(topics)
        while True:
            msg = self.consumer.poll(0.5)
            if msg is None or msg.error():
                continue
            try:
                obj = json.loads(msg.value().decode('utf-8'))
            except Exception:
                continue
            self._route_and_persist(obj)

    # Local-only routing for tests
    def route_message(self, message: dict) -> dict:
        self._route_and_persist(message)
        return {'worker': 'local', 'routed': True}

    def _route_and_persist(self, obj: dict) -> None:
        container = (obj.get('container') or 'app').strip()
        summary = obj.get('summary', '')
        plan = obj.get('plan', '')
        ts = obj.get('ts', 0)
        # Persist last results to storage
        try:
            if self.storage is not None:
                self.storage.put(f'last:{container}:summary', summary)  # type: ignore
                self.storage.put(f'last:{container}:plan', plan)        # type: ignore
                self.storage.put(f'last:{container}:ts', ts)            # type: ignore
        except Exception:
            pass
        # Emit a compact metric record via Kafka when available
        if self.producer is not None:
            metric = {
                'container': container,
                'summary_len': len(summary or ''),
                'plan_len': len(plan or ''),
                'ts': ts,
            }
            try:
                self.producer.produce(self.metrics_topic, json.dumps(metric).encode('utf-8'))
                self.producer.poll(0)
            except Exception:
                pass
