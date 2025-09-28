#!/usr/bin/env python3
"""
Smoke test: produce a handful of agent.results messages to Kafka, wait briefly
for Spark + embed-worker to process, then count Parquet rows in embeddings dir.

Env:
  KAFKA_BOOTSTRAP  (default: kafka:9092)
  RESULTS_TOPIC    (default: agent.results)
  N_MESSAGES       (default: 5)
  EMBED_PARQUET_DIR (default: /workspace/vectorized/embeddings_imesh)
  SLEEP_SEC        (default: 5)
"""
import json
import os
import time
from typing import List

from kafka import KafkaProducer


def produce(bootstrap: str, topic: str, n: int) -> None:
    p = KafkaProducer(bootstrap_servers=bootstrap, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    now = int(time.time())
    for i in range(n):
        doc_id = f"smoke-{now}-{i}"
        msg = {
            'text': f"Smoke test embedding text {i} at {now}",
            'doc_id': doc_id,
            'topic': topic,
            'kafka_ts': time.time(),
        }
        p.send(topic, msg)
    p.flush()


def parquet_counts(path: str) -> int:
    try:
        import pyarrow.parquet as pq
        import glob
        files = sorted(glob.glob(os.path.join(path, '*.parquet')))
        total = 0
        for f in files[-10:]:  # read up to 10 recent files
            try:
                pf = pq.ParquetFile(f)
                total += pf.metadata.num_rows or 0
            except Exception:
                continue
        return total
    except Exception:
        # best-effort: count files only
        import glob
        return len(glob.glob(os.path.join(path, '*.parquet')))


def main():
    bootstrap = os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092')
    topic = os.getenv('RESULTS_TOPIC', 'agent.results')
    n = int(os.getenv('N_MESSAGES', '5'))
    out_dir = os.getenv('EMBED_PARQUET_DIR', '/workspace/vectorized/embeddings_imesh')
    sleep_sec = float(os.getenv('SLEEP_SEC', '5'))
    print(f"[smoke] bootstrap={bootstrap} topic={topic} n={n} out={out_dir}")
    produce(bootstrap, topic, n)
    print(f"[smoke] produced {n} messages; sleeping {sleep_sec}s for pipeline...")
    time.sleep(max(0.0, sleep_sec))
    total = parquet_counts(out_dir)
    print(f"[smoke] parquet rows (approx recent): {total}")


if __name__ == '__main__':
    main()

