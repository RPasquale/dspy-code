from __future__ import annotations

import hashlib
from typing import List


class Embedder:
    """Minimal text embedder for tests.

    Produces deterministic pseudo-embeddings from input text without
    external dependencies.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = max(8, int(dim))

    def _hash_vec(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        # Map bytes to floats in [0,1)
        vals = [b / 255.0 for b in h]
        # Pad/trim to desired dimension
        out = (vals * ((self.dim + len(vals) - 1) // len(vals)))[: self.dim]
        return out

    def embed_text(self, text: str) -> List[float]:
        return self._hash_vec(text or "")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

