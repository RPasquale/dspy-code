from pathlib import Path
import tempfile
import unittest

from dspy_agent.code_tools.code_snapshot import build_code_snapshot


class TestCodeSnapshot(unittest.TestCase):
    def test_build_code_snapshot_for_file_with_outline(self):
        py_src = """
class A:
    pass

def foo():
    return 1
""".strip()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.py"
            p.write_text(py_src)
            snap = build_code_snapshot(p)
            self.assertIn(f"===== FILE {p} =====", snap)
            self.assertIn("-- Outline --", snap)
            self.assertIn("class A", snap)
            self.assertIn("def foo", snap)

    def test_build_code_snapshot_for_directory_samples_and_truncates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # create a couple of files
            (root / "a.py").write_text("def f():\n    return 'x'\n")
            big = "X" * 10000
            (root / "data.txt").write_text(big)
            # ignored directories should be skipped
            (root / ".venv").mkdir()
            (root / ".venv" / "junk.txt").write_text("ignore me")
            snap = build_code_snapshot(root, max_files=10, max_file_bytes=2000)
            self.assertIn(f"===== DIRECTORY {root} (", snap)
            # Contains sampled file headers
            self.assertIn(f"--- {root / 'a.py'} ---", snap)
            self.assertIn(f"--- {root / 'data.txt'} ---", snap)
            # Outline marker should be present for a.py
            self.assertIn("Outline:", snap)


if __name__ == "__main__":
    unittest.main()
