from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import ast


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,63}")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    text: str


def _python_chunks(path: Path, max_bytes: int = 4000) -> Iterator[Chunk]:
    try:
        src = path.read_text(errors="ignore")
    except Exception:
        return
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    lines = src.splitlines()
    # Top-level defs and classes
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lineno = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if not lineno or not end:
                continue
            text = "\n".join(lines[lineno - 1 : end])
            if len(text.encode(errors="ignore")) > max_bytes:
                text = text.encode(errors="ignore")[:max_bytes].decode(errors="ignore")
            yield Chunk(path=str(path), start_line=lineno, end_line=end, text=text)


def iter_chunks(path: Path, max_bytes: int = 4000, lines_per_chunk: int = 200, smart: bool = True) -> Iterator[Chunk]:
    if smart and path.suffix == ".py":
        # First try AST-based chunking
        yielded = False
        for ch in _python_chunks(path, max_bytes=max_bytes):
            yielded = True
            yield ch
        if yielded:
            return
    # Fallback to line-chunking
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return
    lines = text.splitlines()
    if not lines:
        return
    start = 0
    n = len(lines)
    while start < n:
        end = min(n, start + lines_per_chunk)
        chunk_text = "\n".join(lines[start:end])
        # ensure chunk size cap
        if len(chunk_text.encode(errors="ignore")) > max_bytes:
            # crude truncation
            chunk_text = chunk_text.encode(errors="ignore")[:max_bytes].decode(errors="ignore")
        yield Chunk(path=str(path), start_line=start + 1, end_line=end, text=chunk_text)
        start = end


@dataclass
class IndexItem:
    path: str
    start_line: int
    end_line: int
    tokens: Dict[str, float]  # tf-idf weights


@dataclass
class IndexMeta:
    df: Dict[str, int]
    n_docs: int


def build_index(root: Path, include_globs: Sequence[str] | None = None, exclude_globs: Sequence[str] | None = None, lines_per_chunk: int = 200, smart: bool = True) -> Tuple[IndexMeta, List[IndexItem]]:
    include_globs = include_globs or ["**/*"]
    exclude_globs = exclude_globs or ["**/.git/**", "**/.venv/**", "**/node_modules/**", "**/dist/**", "**/build/**"]

    files: List[Path] = []
    for pat in include_globs:
        for p in root.glob(pat):
            if p.is_file() and not any(p.match(ex) for ex in exclude_globs):
                files.append(p)

    docs: List[Tuple[Chunk, Counter[str]]] = []
    df: Dict[str, int] = defaultdict(int)

    for f in files:
        for ch in iter_chunks(f, lines_per_chunk=lines_per_chunk, smart=smart):
            toks = tokenize(ch.text)
            cnt = Counter(toks)
            docs.append((ch, cnt))
            for t in cnt.keys():
                df[t] += 1

    n_docs = max(1, len(docs))
    items: List[IndexItem] = []
    for ch, cnt in docs:
        weights: Dict[str, float] = {}
        max_tf = max(cnt.values()) if cnt else 1
        for t, tf in cnt.items():
            idf = math.log((n_docs + 1) / (1 + df.get(t, 0))) + 1.0  # smoothed
            w = (0.5 + 0.5 * (tf / max_tf)) * idf
            weights[t] = w
        items.append(IndexItem(path=ch.path, start_line=ch.start_line, end_line=ch.end_line, tokens=weights))

    return IndexMeta(df=dict(df), n_docs=n_docs), items


def save_index(root: Path, meta: IndexMeta, items: List[IndexItem], out_dir: Optional[Path] = None) -> Path:
    out_dir = out_dir or (root / ".dspy_index")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({"df": meta.df, "n_docs": meta.n_docs}))
    with (out_dir / "index.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps({
                "path": it.path,
                "start_line": it.start_line,
                "end_line": it.end_line,
                "tokens": it.tokens,
            }) + "\n")
    return out_dir


def load_index(root: Path, in_dir: Optional[Path] = None) -> Tuple[IndexMeta, List[IndexItem]]:
    in_dir = in_dir or (root / ".dspy_index")
    meta_path = in_dir / "meta.json"
    idx_path = in_dir / "index.jsonl"
    if not meta_path.exists() or not idx_path.exists():
        raise FileNotFoundError("Index not found. Run 'index' first.")
    m = json.loads(meta_path.read_text())
    items: List[IndexItem] = []
    with idx_path.open() as f:
        for line in f:
            d = json.loads(line)
            items.append(IndexItem(path=d["path"], start_line=d["start_line"], end_line=d["end_line"], tokens=d["tokens"]))
    return IndexMeta(df=m["df"], n_docs=m["n_docs"]), items


def vectorize_query(q: str, meta: IndexMeta) -> Dict[str, float]:
    toks = tokenize(q)
    cnt = Counter(toks)
    weights: Dict[str, float] = {}
    max_tf = max(cnt.values()) if cnt else 1
    for t, tf in cnt.items():
        idf = math.log((meta.n_docs + 1) / (1 + meta.df.get(t, 0))) + 1.0
        weights[t] = (0.5 + 0.5 * (tf / max_tf)) * idf
    return weights


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    # compute dot over intersection
    dot = 0.0
    for t, wa in a.items():
        wb = b.get(t)
        if wb:
            dot += wa * wb
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


def semantic_search(q: str, meta: IndexMeta, items: List[IndexItem], top_k: int = 5) -> List[Tuple[float, IndexItem]]:
    vq = vectorize_query(q, meta)
    scored: List[Tuple[float, IndexItem]] = []
    for it in items:
        score = cosine(vq, it.tokens)
        if score > 0:
            scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
