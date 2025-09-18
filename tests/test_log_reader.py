from pathlib import Path
import tempfile
import unittest

from dspy_agent.streaming.log_reader import iter_log_paths, read_capped, extract_key_events, load_logs


class TestLogReader(unittest.TestCase):
    def test_iter_and_read_and_extract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.log").write_text("line1\nWARNING: beware\nline3\n")
            (root / "b.out").write_text("ok\n")
            # Non-log extension should be ignored
            (root / "c.md").write_text("not a log\n")

            paths = list(iter_log_paths([root]))
            names = {p.name for p in paths}
            self.assertIn("a.log", names)
            self.assertIn("b.out", names)
            self.assertNotIn("c.md", names)

            # Capped read truncation marker
            big = "X" * 200
            p = root / "big.err"
            p.write_text(big)
            text = read_capped(p, max_bytes=50)
            self.assertIn("... [truncated] ...", text)

            # Key event extraction
            key = extract_key_events((root / "a.log").read_text())
            self.assertIn("WARNING", key)

    def test_load_logs_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "app.log").write_text("Traceback (most recent call last):\nboom\n")
            bundle, count = load_logs([root])
            self.assertEqual(count, 1)
            self.assertIn("=====", bundle)
            self.assertIn("Traceback (most recent call last):", bundle)


if __name__ == "__main__":
    unittest.main()

