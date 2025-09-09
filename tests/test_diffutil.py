from pathlib import Path
import tempfile
import unittest

from dspy_agent.diffutil import unified_diff_from_texts, unified_diff_file_vs_text


class TestDiffUtil(unittest.TestCase):
    def test_unified_diff_from_texts_basic(self):
        a = "hello\nworld\n"
        b = "hello\nWORLD\n"
        diff = unified_diff_from_texts(a, b, a_path="a.txt", b_path="b.txt", n=1)
        self.assertTrue(diff.startswith("--- a.txt\n+++- b.txt") or diff.startswith("--- a.txt\n+++ b.txt"))
        self.assertIn("-world\n+WORLD\n", diff)

    def test_unified_diff_file_vs_text_new_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "file.txt"
            # file does not exist yet; should diff against empty
            diff = unified_diff_file_vs_text(p, "line1\nline2\n", n=0)
            # Entire new content should be additions
            self.assertIn("+line1\n", diff)
            self.assertIn("+line2\n", diff)

    def test_unified_diff_file_vs_text_existing(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "file.txt"
            p.write_text("old\nvalue\n")
            diff = unified_diff_file_vs_text(p, "old\nVALUE\n", n=1)
            self.assertIn("-value\n+VALUE\n", diff)


if __name__ == "__main__":
    unittest.main()
