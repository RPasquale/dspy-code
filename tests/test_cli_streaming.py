import types
import sys
from pathlib import Path
from typer.testing import CliRunner


def _stub_dspy():
    pkg = types.ModuleType('dspy')
    pkg.__path__ = []
    class Signature: pass
    class Module:
        def __init__(self, *a, **k):
            pass
    def InputField(*a, **k): return None
    def OutputField(*a, **k): return None
    class Predict:
        def __init__(self, *a, **k): pass
        def __call__(self, **kwargs):
            return types.SimpleNamespace(context="ctx", key_points="k", plan="p", bullets="b", summary="s")
    pkg.Signature = Signature; pkg.Module = Module; pkg.InputField = InputField; pkg.OutputField = OutputField; pkg.Predict = Predict
    adapters = types.ModuleType('dspy.adapters'); adapters.__path__ = []
    sys.modules['dspy'] = pkg
    sys.modules['dspy.adapters'] = adapters
    sys.modules['dspy.clients'] = types.ModuleType('dspy.clients')
    sys.modules['dspy.clients.lm'] = types.ModuleType('dspy.clients.lm')
    sys.modules['dspy.signatures'] = types.ModuleType('dspy.signatures')
    sys.modules['dspy.signatures.signature'] = types.ModuleType('dspy.signatures.signature')


def test_stream_init_topics(tmp_path: Path):
    _stub_dspy()
    from dspy_agent.cli import app
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res = runner.invoke(app, ["stream_init"])  # type: ignore
        assert res.exit_code == 0
        assert Path(".dspy_stream.json").exists()
        res2 = runner.invoke(app, ["stream_topics"])  # type: ignore
        assert res2.exit_code == 0
        assert "kafka-topics" in res2.output


def test_spark_script_and_k8s_render(tmp_path: Path):
    _stub_dspy()
    from dspy_agent.cli import app
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path(".dspy_stream.json").write_text('{"kafka": {"bootstrap_servers": "localhost:9092", "group_id": "dspy-code", "acks": "all", "topics": []}, "spark": {}, "k8s": {}, "containers": [{"container":"backend","services":["users"]}]}')
        res = runner.invoke(app, ["spark_script"])  # type: ignore
        assert res.exit_code == 0
        assert Path("scripts/streaming/spark_logs.py").exists()
        res2 = runner.invoke(app, ["k8s_render"])  # type: ignore
        assert res2.exit_code == 0
        outdir = Path("deploy/k8s")
        assert any(p.name.startswith("dspy-worker-") for p in outdir.iterdir())

