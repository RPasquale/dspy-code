from __future__ import annotations

"""
VectorBatchFeeder: small utility to feed RL trainers with vectorized features
produced by the Spark vectorizer. Watches a Parquet directory for new files,
loads batches lazily, and yields fixed-size minibatches.

Usage:
    feeder = VectorBatchFeeder("/workspace/vectorized/embeddings", batch_size=128)
    for X, meta in feeder.iter_batches():
        # X: List[List[float]] (minibatch of vectors)
        # meta: dict with file, offset, ts
        train_step(X)

Optionally supports a Kafka source if kafka-python is installed.
"""

from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Dict, Any, Optional
import time


class VectorBatchFeeder:
    def __init__(self, path: str | Path, batch_size: int = 128, poll_sec: float = 2.0):
        self.dir = Path(path)
        self.batch_size = int(batch_size)
        self.poll_sec = float(poll_sec)
        self._cursor: Dict[str, int] = {}

    def _list_parquet(self) -> List[Path]:
        if not self.dir.exists():
            return []
        files = [p for p in self.dir.rglob('*.parquet') if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime)
        return files

    def _read_rows(self, file: Path, start: int, limit: int) -> Tuple[List[List[float]], int]:
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(str(file))
            # Read a slice
            stop = start + limit
            table = pf.read_row_groups(
                [i for i in range(pf.num_row_groups)],
                columns=['vector']
            )
            col = table.column('vector')
            # Slice at Python level (cheap if small)
            data = col.to_pylist()[start:stop]
            # Ensure list of lists of floats
            batch = [list(map(float, (v or []))) for v in data]
            return batch, min(stop, len(col))
        except Exception:
            # Fallback using pandas if available
            try:
                import pandas as pd
                df = pd.read_parquet(str(file), columns=['vector'])
                data = df['vector'].tolist()[start:start+limit]
                batch = [list(map(float, (v or []))) for v in data]
                return batch, min(start + len(batch), len(df))
            except Exception:
                return [], start

    def iter_batches(self) -> Iterator[Tuple[List[List[float]], Dict[str, Any]]]:
        while True:
            files = self._list_parquet()
            progressed = False
            for f in files:
                key = str(f)
                pos = self._cursor.get(key, 0)
                batch, newpos = self._read_rows(f, pos, self.batch_size)
                if batch:
                    progressed = True
                    self._cursor[key] = newpos
                    yield batch, {'file': key, 'offset': pos, 'ts': f.stat().st_mtime}
            if not progressed:
                time.sleep(self.poll_sec)


def kafka_vector_batches(
    bootstrap: str = 'kafka:9092',
    topic: str = 'embeddings',
    batch_size: int = 128,
) -> Iterator[Tuple[List[List[float]], Dict[str, Any]]]:  # pragma: no cover - optional
    """Optional Kafka source for vector minibatches.

    Requires kafka-python. Expects messages as JSON with key 'vector'.
    """
    try:
        from kafka import KafkaConsumer
        import json
    except Exception:
        raise RuntimeError('kafka-python not installed')

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode('utf-8', errors='ignore'))
    )

    batch: List[List[float]] = []
    for msg in consumer:
        vec = msg.value.get('vector')
        if isinstance(vec, list):
            batch.append([float(x) for x in vec])
        if len(batch) >= batch_size:
            yield batch, {'topic': topic, 'ts': time.time()}
            batch = []

