from pathlib import Path
import tempfile
import unittest

from dspy_agent.embedding.embeddings_index import (
    build_emb_index,
    save_emb_index,
    load_emb_index,
    embed_query,
    emb_search,
)


class DummyEmbedderEmbed:
    def embed(self, texts):
        # Simple 3-dim vector based on lengths
        return [[float(len(t)), float(len(t) % 7), 1.0] for t in texts]


class DummyEmbedderEncode:
    def encode(self, texts, prompt_name=None):
        # Different dimension to ensure not confused with other dummy
        return [[float(sum(map(ord, t)) % 100), 2.0] for t in texts]


def dummy_callable(texts):
    return [[1.0] for _ in texts]


class TestEmbeddingsIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Two small Python files to create a few chunks
        (self.root / "a.py").write_text("""
def a():
    return 1

class C:
    def m(self):
        return 2
""".strip())
        (self.root / "b.txt").write_text("line1\nline2\nline3\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_with_embed_interface(self):
        items = build_emb_index(self.root, DummyEmbedderEmbed())
        self.assertTrue(items)
        # vectors should be 3-d
        self.assertEqual(len(items[0].vector), 3)

    def test_build_with_encode_interface(self):
        items = build_emb_index(self.root, DummyEmbedderEncode())
        self.assertTrue(items)
        # vectors should be 2-d
        self.assertEqual(len(items[0].vector), 2)

    def test_build_with_callable(self):
        items = build_emb_index(self.root, dummy_callable)
        self.assertTrue(items)
        self.assertEqual(len(items[0].vector), 1)

    def test_save_load_and_search(self):
        items = build_emb_index(self.root, DummyEmbedderEmbed())
        out_dir = save_emb_index(self.root, items, out_dir=self.root / ".dspy_index_test")
        loaded = load_emb_index(self.root, in_dir=out_dir)
        self.assertEqual(len(loaded), len(items))
        # query vector similar to first item
        qv = embed_query(DummyEmbedderEmbed(), "query text")
        hits = emb_search(qv, loaded, top_k=3)
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()

