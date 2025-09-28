import importlib.util
import socket
import hashlib
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'docker' / 'lightweight' / 'scripts' / 'embed_worker.py'

if 'kafka' not in sys.modules:
    kafka_stub = types.ModuleType('kafka')
    kafka_stub.KafkaConsumer = object  # type: ignore[attr-defined]
    kafka_stub.KafkaProducer = object  # type: ignore[attr-defined]
    sys.modules['kafka'] = kafka_stub

_spec = importlib.util.spec_from_file_location('embed_worker_module', MODULE_PATH)
_embed_worker = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_embed_worker)  # type: ignore[attr-defined]



def test_resolve_bootstrap_alias(monkeypatch):
    monkeypatch.setenv('DSPY_KAFKA_LOCAL_ALIAS', 'kafka')
    assert _embed_worker._resolve_bootstrap('localhost:9092') == 'kafka:9092'
    assert _embed_worker._resolve_bootstrap('PLAINTEXT://127.0.0.1:29092,foo:1234').startswith('PLAINTEXT://kafka:29092')


def test_spark_resolve_bootstrap_alias(monkeypatch):
    pytest.importorskip('pyspark')
    spark_path = Path(__file__).resolve().parents[1] / 'docker' / 'lightweight' / 'scripts' / 'streaming' / 'spark_vectorize.py'
    spark_spec = importlib.util.spec_from_file_location('spark_vectorize_module', spark_path)
    spark_module = importlib.util.module_from_spec(spark_spec)  # type: ignore[arg-type]
    assert spark_spec is not None and spark_spec.loader is not None
    spark_spec.loader.exec_module(spark_module)  # type: ignore[attr-defined]
    monkeypatch.setenv('DSPY_KAFKA_LOCAL_ALIAS', 'kafka')
    assert spark_module._resolve_bootstrap('localhost:9092') == 'kafka:9092'


def test_streamkit_resolve_bootstrap(monkeypatch):
    from dspy_agent.streaming import streamkit

    monkeypatch.delenv('KAFKA_BOOTSTRAP', raising=False)
    monkeypatch.delenv('KAFKA_BOOTSTRAP_SERVERS', raising=False)
    monkeypatch.setenv('DSPY_KAFKA_LOCAL_ALIAS', 'kafka')
    monkeypatch.setenv('DSPY_KAFKA_HOST_ALIAS', 'localhost')
    value = streamkit._resolve_bootstrap(None)
    assert 'kafka:9092' in value
    assert 'localhost:9092' in value


def test_metrics_server_fallback(monkeypatch):
    # Occupy a port to force fallback
    s = socket.socket()
    try:
        s.bind(('127.0.0.1', 0))
    except PermissionError:
        pytest.skip('socket bind not permitted in sandbox')
    busy_port = s.getsockname()[1]
    try:
        monkeypatch.setenv('EMBED_METRICS_FALLBACK_PORTS', '')
        monkeypatch.setenv('EMBED_METRICS_ALLOW_RANDOM', '1')
        srv = _embed_worker.start_metrics_server(busy_port)
        assert srv is not None
        assert srv.server_address[1] != busy_port
    finally:
        if 'srv' in locals() and srv is not None:
            srv.shutdown()
            srv.server_close()
        s.close()


@pytest.mark.parametrize('inputs,expected', [
    ([{'text': 'hello', 'vector': [1.0, 2.0]}], ('hello', 2)),
    ([{'text': 'test', 'vector': []}], ('test', 0)),
])
def test_cache_put_and_flush_logic(monkeypatch, inputs, expected):
    # Verify cache path does not raise and produces consistent doc ids
    cache = _embed_worker.TTLCache(ttl_sec=1, max_items=2)
    text, dim = expected
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    cache.put(key, inputs[0]['vector'])
    assert cache.get(key) == inputs[0]['vector']
    if inputs[0]['vector']:
        assert len(inputs[0]['vector']) == dim
