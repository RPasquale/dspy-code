import json
import types
import sys


def test_context_builder_fallback_on_exception(monkeypatch):
    from dspy_agent.skills.context_builder import ContextBuilder

    builder = ContextBuilder()
    # Force predict to raise to trigger fallback
    def boom(*args, **kwargs):
        raise RuntimeError("LLM error")
    builder.predict = boom  # type: ignore

    res = builder.forward(task="Fix tests", logs_preview="Traceback: KeyError: 'x'")
    assert hasattr(res, 'context') and isinstance(res.context, str)
    assert hasattr(res, 'key_points') and '- ' in res.key_points
    assert 'Traceback' in res.key_points or 'No explicit errors' in res.key_points


def test_adapter_args_json_sanitization():
    from dspy_agent.agents.adapter import _ensure_args_json_string

    # Plain object string
    s1 = '{"a": 1, "b": 2}'
    out1 = _ensure_args_json_string(s1)
    assert json.loads(out1) == {"a": 1, "b": 2}

    # Code-fenced JSON
    s2 = """```json
    {"a": true, "b": [1,2,3]}
    ```"""
    out2 = _ensure_args_json_string(s2)
    assert json.loads(out2) == {"a": True, "b": [1, 2, 3]}

    # Garbage → fallback
    s3 = "not json at all"
    out3 = _ensure_args_json_string(s3)
    assert json.loads(out3) == {}

