from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional, Any

try:
    from confluent_kafka import Consumer, Producer  # type: ignore
except Exception:  # pragma: no cover
    Consumer = None  # type: ignore
    Producer = None  # type: ignore

from .streaming_runtime import process_ctx
from .skills.context_builder import ContextBuilder
from .skills.task_agent import TaskAgent
from .llm import configure_lm


@dataclass
class KafkaParams:
    bootstrap: str
    group: str
    in_topic: str
    out_topic: str
    container: str


class WorkerLoop:
    def __init__(self, params: KafkaParams):
        self.p = params
        self.running = False
        self.builder = ContextBuilder()
        self.agent = TaskAgent()
        self.lm = configure_lm(provider="ollama", model_name=None, base_url=None, api_key=None)

    def run(self) -> None:
        if Consumer is None or Producer is None:
            raise RuntimeError("confluent-kafka not installed")
        conf_c = {
            'bootstrap.servers': self.p.bootstrap,
            'group.id': self.p.group,
            'auto.offset.reset': 'latest',
        }
        conf_p = {'bootstrap.servers': self.p.bootstrap}
        consumer = Consumer(conf_c)
        producer = Producer(conf_p)
        consumer.subscribe([self.p.in_topic])
        self.running = True
        try:
            while self.running:
                msg = consumer.poll(0.5)
                if msg is None:
                    continue
                if msg.error():
                    continue
                payload = msg.value()
                try:
                    obj = json.loads(payload.decode('utf-8'))
                except Exception:
                    continue
                lines = obj.get('ctx') or obj.get('lines') or []
                text = "\n".join(lines) if isinstance(lines, list) else str(lines)
                result = process_ctx(self.p.container, text, self.lm, self.builder, self.agent)
                data = json.dumps(result).encode('utf-8')
                producer.produce(self.p.out_topic, data)
                producer.poll(0)
        finally:
            try:
                consumer.close()
            except Exception:
                pass

