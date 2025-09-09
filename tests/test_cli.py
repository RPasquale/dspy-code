from pathlib import Path
import tempfile
import types
import sys
import unittest

try:
    from typer.testing import CliRunner
    _HAS_TYPER = True
except Exception:
    _HAS_TYPER = False


class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _HAS_TYPER:
            raise unittest.SkipTest("typer not installed")
        # Provide a dummy 'dspy' package with minimal submodules used by CLI imports
        pkg = types.ModuleType('dspy')
        pkg.__path__ = []  # mark as package
        # Embeddings stub used by CLI embeddings commands
        class _Embeddings:
            def __init__(self, *a, **k):
                pass
        pkg.Embeddings = _Embeddings
        # core dspy symbols used in skills
        class Signature: pass
        class Module:
            def __init__(self, *a, **k):
                pass
        def InputField(*a, **k):
            return None
        def OutputField(*a, **k):
            return None
        class Predict:
            def __init__(self, *a, **k):
                pass
            def __call__(self, **kwargs):
                # Return object with possible fields used in CLI printing
                return types.SimpleNamespace(
                    context="",
                    key_points="",
                    summary="",
                    bullets="",
                    plan="",
                    commands="",
                )
        pkg.Signature = Signature
        pkg.Module = Module
        pkg.InputField = InputField
        pkg.OutputField = OutputField
        pkg.Predict = Predict

        adapters = types.ModuleType('dspy.adapters'); adapters.__path__ = []
        json_adapter = types.ModuleType('dspy.adapters.json_adapter')
        class JSONAdapter:  # minimal placeholder
            def __init__(self, *a, **k):
                pass
        json_adapter.JSONAdapter = JSONAdapter

        clients = types.ModuleType('dspy.clients'); clients.__path__ = []
        lm = types.ModuleType('dspy.clients.lm')
        class LM:  # placeholder
            pass
        lm.LM = LM

        signatures = types.ModuleType('dspy.signatures'); signatures.__path__ = []
        signature = types.ModuleType('dspy.signatures.signature')
        class Signature:  # placeholder
            pass
        signature.Signature = Signature

        sys.modules['dspy'] = pkg
        sys.modules['dspy.adapters'] = adapters
        sys.modules['dspy.adapters.json_adapter'] = json_adapter
        sys.modules['dspy.clients'] = clients
        sys.modules['dspy.clients.lm'] = lm
        sys.modules['dspy.signatures'] = signatures
        sys.modules['dspy.signatures.signature'] = signature

        from dspy_agent.cli import app  # type: ignore
        cls.app = app
        cls.runner = CliRunner()

    def test_tree_command(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.py").write_text("print('x')\n")
            res = self.runner.invoke(self.app, ["tree", "--root", str(root), "--depth", "1"])  # type: ignore
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertIn("a.py", res.output)

    def test_context_no_lm(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logs = root / "logs"
            logs.mkdir()
            (logs / "app.log").write_text("info start\nERROR: something failed\nend\n")
            res = self.runner.invoke(self.app, [
                "context", "--logs", str(logs), "--no-lm"
            ])  # type: ignore
            self.assertEqual(res.exit_code, 0, res.output)
            # Should include extract header and error content
            self.assertIn("Extracted Events", res.output)
            self.assertIn("ERROR", res.output)


if __name__ == "__main__":
    unittest.main()
