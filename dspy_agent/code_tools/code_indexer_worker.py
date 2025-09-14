from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    from confluent_kafka import Consumer  # type: ignore
except Exception:  # pragma: no cover
    Consumer = None  # type: ignore

from ..agents.knowledge import _py_facts  # type: ignore
from ..embedding.indexer import iter_chunks
import hashlib
import dspy
from ..db.factory import get_storage


class CodeIndexerWorker:
    def __init__(self, bootstrap: str, topic: str = 'code.fs.events', workspace: Optional[Path] = None) -> None:
        if Consumer is None:
            raise RuntimeError('confluent-kafka not installed')
        self.consumer = Consumer({'bootstrap.servers': bootstrap, 'group.id': 'dspy-code-indexer', 'auto.offset.reset': 'latest'})
        self.consumer.subscribe([topic])
        self.workspace = workspace or Path.cwd()
        self.storage = get_storage()

    def _update_python(self, file: Path):
        facts = _py_facts(file)
        if self.storage is None:
            return
        # Persist per-file facts
        try:
            root = self.workspace.resolve()
            rel = str(file.resolve()).replace(str(root), '').lstrip('/')
            safe = (rel or file.name).replace('/', '|')
            self.storage.put(f'code:file:{safe}:facts', {
                'path': facts.path,
                'lines': facts.lines,
                'imports': facts.imports,
                'classes': facts.classes,
                'functions': facts.functions,
            })  # type: ignore
            # Update AST caches for classes/functions
            for c in facts.classes:
                key = f'code:ast:class:{c}'
                try:
                    cur = self.storage.get(key) or []  # type: ignore
                    if facts.path not in cur:
                        cur.append(facts.path)
                        self.storage.put(key, cur)  # type: ignore
                except Exception:
                    pass
            for fn in facts.functions:
                key = f'code:ast:function:{fn}'
                try:
                    cur = self.storage.get(key) or []  # type: ignore
                    if facts.path not in cur:
                        cur.append(facts.path)
                        self.storage.put(key, cur)  # type: ignore
                except Exception:
                    pass
            # Incremental embeddings for this file
            try:
                # Prefer local small HF model; fallback to DSPy embeddings if HF unavailable
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                    embedder = SentenceTransformer('all-MiniLM-L6-v2')
                    def _embed(texts):
                        return embedder.encode(texts)
                except Exception:
                    emb = dspy.Embeddings(model='openai/text-embedding-3-small')
                    def _embed(texts):
                        return emb.embed(texts)  # type: ignore
                chunks = list(iter_chunks(file, lines_per_chunk=200, smart=True))
                vecs = _embed([c.text for c in chunks])
                for ch, vec in zip(chunks, vecs):
                    rec = {"path": ch.path, "start_line": ch.start_line, "end_line": ch.end_line, "vector": list(vec)}
                    self.storage.append('emb.index', rec)  # type: ignore
                    try:
                        text = Path(ch.path).read_text(errors='ignore')
                        lines = text.splitlines()
                        seg = "\n".join(lines[ch.start_line - 1 : ch.end_line])
                        h = hashlib.sha256((ch.path + str(ch.start_line) + str(ch.end_line)).encode('utf-8')).hexdigest()
                        self.storage.append('code.chunks', {"hash": h, "path": ch.path, "start_line": ch.start_line, "end_line": ch.end_line, "text": seg})  # type: ignore
                        self.storage.put(f'code:chunk:{h}', {"path": ch.path, "start_line": ch.start_line, "end_line": ch.end_line, "text": seg})  # type: ignore
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    def run(self) -> None:
        while True:
            msg = self.consumer.poll(0.5)
            if msg is None or msg.error():
                continue
            try:
                obj = json.loads(msg.value().decode('utf-8'))
            except Exception:
                continue
            path = obj.get('path')
            if not path:
                continue
            p = Path(path)
            if p.suffix == '.py':
                self._update_python(p)
