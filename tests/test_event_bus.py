import json
import os
from pathlib import Path

def test_publish_event_writes_jsonl(tmp_path, monkeypatch):
    # Route event bus to a temp dir
    log_dir = tmp_path / 'logs'
    monkeypatch.setenv('EVENTBUS_LOG_DIR', str(log_dir))
    # Import late to pick up env
    from dspy_agent.streaming import publish_event
    # Publish a UI event
    publish_event('ui.action', { 'action': 'click_test', 'extra': 123 })
    p = log_dir / 'ui_action.jsonl'
    assert p.exists(), 'event jsonl file should be created'
    lines = p.read_text().strip().splitlines()
    assert len(lines) >= 1
    rec = json.loads(lines[-1])
    # Envelope and event fields present
    assert 'ts' in rec or 'timestamp' in rec
    assert rec.get('topic') in ('ui.action', 'events.other')
    assert isinstance(rec.get('event'), dict) and rec['event'].get('action') == 'click_test'

