import json
from pathlib import Path

from dspy_agent.streaming.streaming_config import StreamConfig, save_config, load_config


def test_stream_config_roundtrip(tmp_path: Path):
    cfg = StreamConfig.default()
    p = tmp_path / ".dspy_stream.json"
    save_config(cfg, p)
    assert p.exists()
    cfg2 = load_config(p)
    assert cfg2.kafka.bootstrap_servers == cfg.kafka.bootstrap_servers
    assert len(cfg2.kafka.topics) >= 3
    assert cfg2.kafka.vector_topic == cfg.kafka.vector_topic
    assert isinstance(cfg2.kafka.vector_topics, list)
