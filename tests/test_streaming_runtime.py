import time
from pathlib import Path

import types, sys


def _stub_dspy():
    pkg = types.ModuleType('dspy'); pkg.__path__ = []
    class Signature: pass
    class Module:
        def __init__(self, *a, **k): pass
    def InputField(*a, **k): return None
    def OutputField(*a, **k): return None
    class Predict:
        def __init__(self, *a, **k): pass
        def __call__(self, **kwargs):
            return types.SimpleNamespace(context="ctx", key_points="k", plan="p", bullets="b", summary="s")
    pkg.Signature = Signature; pkg.Module = Module; pkg.InputField = InputField; pkg.OutputField = OutputField; pkg.Predict = Predict
    sys.modules['dspy'] = pkg
    adapters = types.ModuleType('dspy.adapters'); adapters.__path__ = []
    json_adapter = types.ModuleType('dspy.adapters.json_adapter')
    class JSONAdapter: pass
    json_adapter.JSONAdapter = JSONAdapter
    sys.modules['dspy.adapters'] = adapters
    sys.modules['dspy.adapters.json_adapter'] = json_adapter
    clients = types.ModuleType('dspy.clients'); clients.__path__ = []
    lm = types.ModuleType('dspy.clients.lm')
    class LM: pass
    lm.LM = LM
    sys.modules['dspy.clients'] = clients
    sys.modules['dspy.clients.lm'] = lm
    signatures = types.ModuleType('dspy.signatures'); signatures.__path__ = []
    signature_mod = types.ModuleType('dspy.signatures.signature')
    class Signature: pass
    signature_mod.Signature = Signature
    sys.modules['dspy.signatures'] = signatures
    sys.modules['dspy.signatures.signature'] = signature_mod


_stub_dspy()
from dspy_agent.streaming_runtime import LocalBus, Aggregator, autodiscover_logs, process_ctx
from dspy_agent.skills.context_builder import ContextBuilder
from dspy_agent.skills.task_agent import TaskAgent


def test_localbus_pubsub():
    bus = LocalBus()
    q = bus.subscribe("t")
    bus.publish("t", {"x": 1})
    got = q.get(timeout=0.5)
    assert got == {"x": 1}


def test_aggregator_flush(tmp_path: Path):
    # Use LocalBus directly
    from dspy_agent.streaming_runtime import LocalBus
    bus = LocalBus()
    in_q = bus.subscribe("raw")
    agg = Aggregator(bus, "raw", "ctx", window_sec=10.0)
    # do not start thread; simulate inputs
    bus.publish("raw", {"line": "INFO ok"})
    bus.publish("raw", {"line": "ERROR boom"})
    # flush now
    agg.flush_now()
    out = bus.subscribe("ctx")
    # Publish emits to all subscribers; to ensure delivery, push again and flush
    agg._buf.append("ERROR again")
    agg.flush_now()
    got = out.get(timeout=0.5)
    assert "ctx" in got


def test_autodiscover(tmp_path: Path):
    logs = tmp_path / "backend" / "logs"
    logs.mkdir(parents=True)
    p = logs / "app.log"
    p.write_text("hello\nERROR boom\n")
    discs = autodiscover_logs(tmp_path)
    assert any(d.container == "backend" for d in discs)


def test_process_ctx_heuristic():
    # Force heuristic by passing lm=None
    builder = ContextBuilder()
    agent = TaskAgent()
    res = process_ctx("backend", "ERROR boom", None, builder, agent)
    assert res["container"] == "backend"
    assert "summary" in res
