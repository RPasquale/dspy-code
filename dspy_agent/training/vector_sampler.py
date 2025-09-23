from __future__ import annotations

"""
VectorLabelSampler: combine vector features (Parquet) with labels (JSONL) for
supervised fine-tuning. Labels JSONL format (flexible):
  {"text": "...", "label": 1}
  or {"text": "...", "label": "class_name"}

If labels are strings, a label_to_id mapping is built in insertion order.
"""
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Any


class VectorLabelSampler:
    def __init__(self, vectors_dir: str | Path, labels_jsonl: str | Path, batch_size: int = 128) -> None:
        self.dir = Path(vectors_dir)
        self.labels_path = Path(labels_jsonl)
        self.batch_size = int(batch_size)
        self.label_map: Dict[str, Any] = {}
        self.class_to_id: Dict[str, int] = {}
        self._load_labels()

    def _load_labels(self) -> None:
        import json
        if not self.labels_path.exists():
            return
        with self.labels_path.open('r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = rec.get('text')
                label = rec.get('label')
                if not isinstance(text, str):
                    continue
                if isinstance(label, str):
                    if label not in self.class_to_id:
                        self.class_to_id[label] = len(self.class_to_id)
                    label = self.class_to_id[label]
                self.label_map[text] = label

    def iter_batches(self) -> Iterator[Tuple[List[List[float]], List[Any]]]:
        files = [p for p in self.dir.rglob('*.parquet') if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime)
        try:
            import pandas as pd
        except Exception as e:  # pragma: no cover
            raise RuntimeError('pandas is required for VectorLabelSampler') from e
        X: List[List[float]] = []
        y: List[Any] = []
        for f in files:
            try:
                df = pd.read_parquet(str(f), columns=['text', 'vector'])
            except Exception:
                continue
            for _, row in df.iterrows():
                text = row.get('text')
                if not isinstance(text, str):
                    continue
                if text not in self.label_map:
                    continue
                vec = row.get('vector')
                if not isinstance(vec, list):
                    continue
                X.append([float(v) for v in vec])
                y.append(self.label_map[text])
                if len(X) >= self.batch_size:
                    yield X, y
                    X, y = [], []
        if X:
            yield X, y

