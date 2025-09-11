import types, sys, time
from pathlib import Path


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
            return types.SimpleNamespace(context="ctx", key_points="kp", plan="pl", bullets="b", summary="s")
    pkg.Signature = Signature; pkg.Module = Module; pkg.InputField = InputField; pkg.OutputField = OutputField; pkg.Predict = Predict
    sys.modules['dspy'] = pkg
    # minimal submodules
    sys.modules['dspy.adapters'] = types.ModuleType('dspy.adapters')
    clients = types.ModuleType('dspy.clients'); clients.__path__ = []
    lm = types.ModuleType('dspy.clients.lm')
    class LM: pass
    lm.LM = LM
    sys.modules['dspy.clients'] = clients
    sys.modules['dspy.clients.lm'] = lm
    signs = types.ModuleType('dspy.signatures'); signs.__path__ = []
    sig_sig = types.ModuleType('dspy.signatures.signature')
    class S: pass
    sig_sig.Signature = S
    sys.modules['dspy.signatures'] = signs
    sys.modules['dspy.signatures.signature'] = sig_sig
    adp = types.ModuleType('dspy.adapters.json_adapter')
    class JA: pass
    adp.JSONAdapter = JA
    sys.modules['dspy.adapters.json_adapter'] = adp


def test_local_pipeline_writes_dataset(tmp_path: Path):
    _stub_dspy()
    from dspy_agent.streaming_runtime import LocalBus, FileTailer, Aggregator, Trainer
    # Prepare a fake log file and write lines
    log = tmp_path / "logs" / "app.log"
    log.parent.mkdir(parents=True)
    log.write_text("INFO start\n")
    bus = LocalBus()
    t1 = FileTailer(log, bus, "logs.raw.app", poll_interval=0.1)
    t2 = Aggregator(bus, "logs.raw.app", "logs.ctx.app", window_sec=0.2)
    trainer = Trainer(tmp_path, bus, containers=["app"], min_batch=1, interval_sec=0.2)
    t1.start(); t2.start(); trainer.start()
    # Append an error line to trigger aggregation and trainer append
    with log.open("a") as f:
        f.write("ERROR boom\n")
    time.sleep(0.8)
    # Check dataset exists and has at least one line
    ds = tmp_path / ".dspy_data" / "context_train.jsonl"
    assert ds.exists() and ds.stat().st_size > 0
    # Stop threads
    t1.stop(); t2.stop(); trainer.stop()

