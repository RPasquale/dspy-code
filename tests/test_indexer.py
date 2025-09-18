from pathlib import Path
import tempfile
import unittest

from dspy_agent.embedding.indexer import (
    tokenize,
    iter_chunks,
    build_index,
    vectorize_query,
    cosine,
    semantic_search,
    save_index,
    load_index,
)


class TestIndexer(unittest.TestCase):
    def test_tokenize_simple(self):
        text = "def FooBar(x): return x + 1"
        toks = tokenize(text)
        self.assertIn("foobar", toks)
        self.assertIn("return", toks)
        # no punctuation-only tokens
        self.assertTrue(all(t.isidentifier() for t in toks))

    def test_iter_chunks_ast_and_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.py"
            p.write_text(
                """
def a():
    return 1

class C:
    def m(self):
        return 2
""".strip()
            )
            chunks = list(iter_chunks(p))
            # AST should split into defs/classes (>= 2 chunks)
            self.assertGreaterEqual(len(chunks), 2)
            self.assertTrue(chunks[0].path.endswith("m.py"))

            # Non-Python file falls back to line chunking
            q = Path(td) / "notes.txt"
            q.write_text("one\n" * 15)
            q_chunks = list(iter_chunks(q, lines_per_chunk=5))
            self.assertEqual(len(q_chunks), 3)
            self.assertEqual(q_chunks[0].start_line, 1)
            self.assertEqual(q_chunks[-1].end_line, 15)

    def test_build_index_and_search_and_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "alpha.py").write_text("def alpha():\n    return 'hello world'\n")
            (root / "beta.py").write_text("def beta():\n    return 'goodbye'\n")

            meta, items = build_index(root)
            self.assertGreater(meta.n_docs, 0)
            self.assertTrue(items, "expected some index items")

            results = semantic_search("hello", meta, items, top_k=3)
            self.assertTrue(results and any("alpha.py" in r[1].path for r in results))

            # vectorize and cosine sanity
            vq = vectorize_query("alpha hello", meta)
            self.assertTrue(vq)
            sim_self = cosine(vq, vq)
            self.assertTrue(0.99 <= sim_self <= 1.01)

            # save/load roundtrip
            out_dir = save_index(root, meta, items, out_dir=root / ".dspy_index_test")
            meta2, items2 = load_index(root, in_dir=out_dir)
            self.assertEqual(meta2.n_docs, meta.n_docs)
            self.assertEqual(len(items2), len(items))


if __name__ == "__main__":
    unittest.main()
