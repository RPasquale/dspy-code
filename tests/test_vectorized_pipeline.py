import time

from dspy_agent.streaming.streamkit import LocalBus
from dspy_agent.streaming.vectorized_pipeline import (
    KafkaStreamInspector,
    RLVectorizer,
    SparkCheckpointMonitor,
    VectorizedStreamOrchestrator,
)


def test_rl_vectorizer_basic(tmp_path):
    vectorizer = RLVectorizer(tmp_path)
    payload = {"ctx": ["ERROR: database connection failed", "Traceback (most recent call last)"]}
    record = vectorizer.vectorize("logs.ctx.backend", payload)
    assert record is not None
    assert len(record.features) == vectorizer.feature_size
    assert record.metadata["source_topic"] == "logs.ctx.backend"
    assert 0.0 <= min(record.features) <= 1.0
    assert 0.0 <= max(record.features) <= 1.0
    assert "digest" in record.metadata


def test_vectorized_orchestrator_streams(tmp_path):
    bus = LocalBus()
    vectorizer = RLVectorizer(tmp_path)
    orchestrator = VectorizedStreamOrchestrator(
        bus,
        ["logs.ctx.backend"],
        vectorizer,
        out_topic="agent.rl.vectorized",
        idle_sleep=0.01,
    )
    out_queue = bus.subscribe("agent.rl.vectorized")
    orchestrator.start()
    try:
        bus.publish("logs.ctx.backend", {"ctx": ["Error: timeout while calling service"]})
        emitted = out_queue.get(timeout=1.5)
    finally:
        orchestrator.stop()
        orchestrator.join(timeout=1.0)
    assert isinstance(emitted, dict)
    assert emitted["topic"] == "logs.ctx.backend"
    assert len(emitted["features"]) == vectorizer.feature_size
    metrics = bus.vector_metrics()
    assert metrics.get("processed", 0) >= 1


def test_kafka_stream_inspector_no_bootstrap():
    inspector = KafkaStreamInspector("", ["agent.rl.vectorized"])
    snapshot = inspector.probe()
    assert snapshot.available is False
    assert snapshot.reason == 'bootstrap not configured'


def test_spark_checkpoint_monitor(tmp_path):
    monitor = SparkCheckpointMonitor(tmp_path / "checkpoint", freshness_threshold=60.0)
    status_missing = monitor.snapshot()
    assert status_missing.active is False
    assert status_missing.reason == 'checkpoint missing'

    checkpoint = tmp_path / "checkpoint"
    data_dir = checkpoint / "sources"
    data_dir.mkdir(parents=True)
    file_path = data_dir / "0"
    file_path.write_text("state")
    time.sleep(0.01)  # ensure mtime is observed

    monitor = SparkCheckpointMonitor(checkpoint, freshness_threshold=60.0)
    status_active = monitor.snapshot()
    assert status_active.active is True
    assert status_active.latest_file is not None


def test_local_bus_health_defaults():
    bus = LocalBus()
    kafka_status = bus.kafka_health()
    assert kafka_status["available"] is False
    spark_status = bus.spark_health()
    assert spark_status["active"] is False
