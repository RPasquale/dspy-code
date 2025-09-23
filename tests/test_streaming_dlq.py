from pathlib import Path
import json


class _KafkaRaiser:
    def send(self, topic, message):  # pragma: no cover - simple test shim
        raise RuntimeError("kafka down")


def test_localbus_dlq_written(tmp_path, monkeypatch):
    # Work in temp dir
    monkeypatch.chdir(tmp_path)
    from dspy_agent.streaming.streamkit import LocalBus

    bus = LocalBus(storage=None, kafka=_KafkaRaiser())
    bus.publish("agent.metrics", {"ok": True})

    dlq = tmp_path / ".dspy_reports" / "dlq.jsonl"
    assert dlq.exists(), "DLQ file should exist"
    lines = dlq.read_text().strip().splitlines()
    assert lines and json.loads(lines[-1])["topic"] == "agent.metrics"

