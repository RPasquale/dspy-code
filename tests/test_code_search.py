from pathlib import Path
import tempfile
import unittest
from unittest import mock

from dspy_agent.code_tools.code_search import (
    search_text,
    search_file,
    extract_context,
    python_extract_symbol,
    run_ast_grep,
)


class TestCodeSearch(unittest.TestCase):
    def test_search_text_and_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.txt").write_text("hello world\nsecond line\n")
            (root / "b.py").write_text("def greet():\n    return 'hello'\n")

            hits = search_text(root, r"hello", regex=True)
            self.assertGreaterEqual(len(hits), 2)
            self.assertTrue(any(h.path.name == "a.txt" for h in hits))

            fhits = search_file(root / "b.py", "greet", regex=False)
            self.assertTrue(any("greet" in h.line for h in fhits))

    def test_extract_context(self):
        text = "\n".join([f"line{i}" for i in range(1, 11)])
        s, e, seg = extract_context(text, line_no=5, before=2, after=2)
        self.assertEqual((s, e), (3, 7))
        self.assertIn("line3", seg)
        self.assertIn("line7", seg)

    def test_python_extract_symbol(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.py"
            p.write_text(
                """
class C:
    def method(self):
        return 1

def target():
    return 2
""".strip()
            )
            info = python_extract_symbol(p, "target")
            self.assertIsNotNone(info)
            start, end, seg = info  # type: ignore
            self.assertGreaterEqual(end, start)
            self.assertIn("def target", seg)

    def test_run_ast_grep_missing(self):
        # Simulate ast-grep not being available
        with mock.patch("dspy_agent.code_tools.code_search.shutil.which", return_value=None):
            code, out, err = run_ast_grep(Path.cwd(), pattern="def ", lang="python")
            self.assertEqual(code, 127)
            self.assertIn("ast-grep not found", err)

    def test_run_ast_grep_invocation(self):
        # Simulate ast-grep available and returning success
        fake_proc = mock.Mock(returncode=0, stdout="OK", stderr="")
        with mock.patch("dspy_agent.code_tools.code_search.shutil.which", return_value="ast-grep"), \
             mock.patch("dspy_agent.code_tools.code_search.subprocess.run", return_value=fake_proc) as mrun:
            code, out, err = run_ast_grep(Path("/tmp/x"), pattern="def ", lang="python", json=True)
            self.assertEqual(code, 0)
            self.assertEqual(out, "OK")
            self.assertEqual(err, "")
            # Ensure command was constructed with pattern and language
            args = mrun.call_args[0][0]
            self.assertIn("-p", args)
            self.assertIn("python", args)


if __name__ == "__main__":
    unittest.main()

