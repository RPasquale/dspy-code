from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .indexer import Chunk, iter_chunks
from .db.factory import get_storage
import hashlib


@dataclass
class EmbIndexItem:
    path: str
    start_line: int
    end_line: int
    vector: List[float]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _embed_texts(embedder: Any, texts: List[str], is_query: bool = False) -> List[List[float]]:
    """Embed a list of texts.

    Supports:
    - DSPy Embeddings: object with .embed(list[str]) or callable
    - SentenceTransformers: object with .encode(list[str], prompt_name=...)
    """
    if hasattr(embedder, "embed"):
        return embedder.embed(texts)  # type: ignore
    if hasattr(embedder, "encode"):
        # sentence-transformers interface; pass prompt_name for queries when available
        try:
            if is_query:
                return embedder.encode(texts, prompt_name="query")  # type: ignore
            return embedder.encode(texts)  # type: ignore
        except TypeError:
            # Older versions may not accept prompt_name
            return embedder.encode(texts)  # type: ignore
    if callable(embedder):
        return embedder(texts)  # type: ignore
    raise RuntimeError("Unsupported Embeddings interface.")


def build_emb_index(
    root: Path,
    embedder: Any,
    include_globs: Sequence[str] | None = None,
    exclude_globs: Sequence[str] | None = None,
    lines_per_chunk: int = 200,
    smart: bool = True,
) -> List[EmbIndexItem]:
    files: List[Path] = []
    include_globs = include_globs or ["**/*"]
    exclude_globs = exclude_globs or ["**/.git/**", "**/.venv/**", "**/node_modules/**", "**/dist/**", "**/build/**"]
    for pat in include_globs:
        for p in root.glob(pat):
            if p.is_file() and not any(p.match(ex) for ex in exclude_globs):
                files.append(p)

    chunks: List[Chunk] = []
    for f in files:
        for ch in iter_chunks(f, lines_per_chunk=lines_per_chunk, smart=smart):
            chunks.append(ch)

    vectors = _embed_texts(embedder, [c.text for c in chunks], is_query=False)
    items: List[EmbIndexItem] = []
    for ch, vec in zip(chunks, vectors):
        items.append(EmbIndexItem(path=ch.path, start_line=ch.start_line, end_line=ch.end_line, vector=list(vec)))
    return items


def save_emb_index(root: Path, items: List[EmbIndexItem], out_dir: Optional[Path] = None, persist: bool = False, persist_limit: int = 1000) -> Path:
    out_dir = out_dir or (root / ".dspy_index")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "emb_index.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps({
                "path": it.path,
                "start_line": it.start_line,
                "end_line": it.end_line,
                "vector": it.vector,
            }) + "\n")
    if persist:
        try:
            st = get_storage()
        except Exception:
            st = None
        if st is not None:
            try:
                meta = {"root": str(root.resolve()), "count": len(items)}
                st.put('emb:index:meta', meta)  # type: ignore
                # Append up to persist_limit items to streams for backfill
                for i, it in enumerate(items[: max(0, persist_limit)]):
                    rec = {"path": it.path, "start_line": it.start_line, "end_line": it.end_line, "vector": it.vector}
                    st.append('emb.index', rec)  # type: ignore
                    # Also persist code chunk text for this item
                    try:
                        p = Path(it.path)
                        text = p.read_text(errors="ignore")
                        lines = text.splitlines()
                        seg = "\n".join(lines[it.start_line - 1 : it.end_line])
                        h = hashlib.sha256((it.path + str(it.start_line) + str(it.end_line)).encode('utf-8')).hexdigest()
                        st.append('code.chunks', {"hash": h, "path": it.path, "start_line": it.start_line, "end_line": it.end_line, "text": seg})  # type: ignore
                        # KV cache for quick lookup
                        st.put(f'code:chunk:{h}', {"path": it.path, "start_line": it.start_line, "end_line": it.end_line, "text": seg})  # type: ignore
                    except Exception:
                        pass
            except Exception:
                pass
    return out_dir


def load_emb_index(root: Path, in_dir: Optional[Path] = None) -> List[EmbIndexItem]:
    in_dir = in_dir or (root / ".dspy_index")
    path = in_dir / "emb_index.jsonl"
    if not path.exists():
        raise FileNotFoundError("Embedding index not found. Run 'emb-index' first.")
    items: List[EmbIndexItem] = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            items.append(EmbIndexItem(path=d["path"], start_line=d["start_line"], end_line=d["end_line"], vector=d["vector"]))
    return items


def embed_query(embedder: Any, query: str) -> List[float]:
    vecs = _embed_texts(embedder, [query], is_query=True)
    return list(vecs[0])


def emb_search(query_vec: List[float], items: List[EmbIndexItem], top_k: int = 5) -> List[Tuple[float, EmbIndexItem]]:
    scored: List[Tuple[float, EmbIndexItem]] = []
    for it in items:
        score = _cosine(query_vec, it.vector)
        if score > 0:
            scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
