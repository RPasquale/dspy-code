import json
import sys
from types import SimpleNamespace

from dspy_agent.streaming.kafka_log import KafkaLogger, get_kafka_logger


def _clear_kafka_env(monkeypatch):
    monkeypatch.delenv('KAFKA_BOOTSTRAP', raising=False)
    monkeypatch.delenv('KAFKA_BOOTSTRAP_SERVERS', raising=False)
    monkeypatch.delenv('DSPY_DISABLE_KAFKA', raising=False)
    monkeypatch.delenv('DSPY_KAFKA_LOG_LEVEL', raising=False)
    monkeypatch.delenv('DSPY_KAFKA_RETRY_BACKOFF', raising=False)
    monkeypatch.delenv('DSPY_KAFKA_CONNECT_TIMEOUT', raising=False)
    monkeypatch.setenv('KAFKA_BOOTSTRAP', '')
    monkeypatch.setenv('KAFKA_BOOTSTRAP_SERVERS', '')


def test_kafka_logger_disabled_without_bootstrap(monkeypatch):
    _clear_kafka_env(monkeypatch)
    kl = KafkaLogger(bootstrap=None)
    assert kl.bootstrap == ''
    assert not kl.enabled
    assert get_kafka_logger() is None


def test_kafka_logger_respects_disable_flag(monkeypatch):
    _clear_kafka_env(monkeypatch)
    monkeypatch.setenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    monkeypatch.setenv('DSPY_DISABLE_KAFKA', 'true')
    kl = KafkaLogger()
    assert not kl.enabled


def test_kafka_logger_skips_when_unreachable(monkeypatch):
    _clear_kafka_env(monkeypatch)
    monkeypatch.setenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    monkeypatch.setenv('DSPY_DISABLE_KAFKA', '0')
    monkeypatch.setenv('DSPY_KAFKA_RETRY_BACKOFF', '0.1')
    monkeypatch.setattr('dspy_agent.streaming.kafka_log.KafkaLogger._bootstrap_available', lambda self: False)

    calls = []

    class DummyProducer:
        def __init__(self, conf):
            calls.append(conf)

    monkeypatch.setitem(sys.modules, 'confluent_kafka', SimpleNamespace(Producer=DummyProducer))

    kl = KafkaLogger()
    assert kl.enabled
    assert calls == []

    kl.send('topic', {'foo': 'bar'})
    assert calls == []


def test_kafka_logger_produces_when_available(monkeypatch):
    _clear_kafka_env(monkeypatch)
    monkeypatch.setenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    monkeypatch.setenv('DSPY_DISABLE_KAFKA', '0')
    monkeypatch.setattr('dspy_agent.streaming.kafka_log.KafkaLogger._bootstrap_available', lambda self: True)

    produced = []

    class DummyProducer:
        def __init__(self, conf):
            self.conf = conf

        def produce(self, topic, value):
            produced.append((topic, value))

        def poll(self, timeout):
            return None

        def flush(self, timeout):
            return None

    monkeypatch.setitem(sys.modules, 'confluent_kafka', SimpleNamespace(Producer=DummyProducer))

    kl = KafkaLogger()
    assert kl.enabled

    kl.send('logs.ctx.test', {'value': 1})
    assert produced, 'expected a message to be produced'

    topic, payload = produced[0]
    assert topic == 'logs.ctx.test'
    assert json.loads(payload.decode('utf-8')) == {'value': 1}
