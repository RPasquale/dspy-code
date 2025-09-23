#!/usr/bin/env python3
"""
Kafka → InferMesh → Kafka/Parquet/RedDB embedding worker.

Responsibilities:
- Consume parsed text messages from Kafka (`EMBED_INPUT_TOPIC`), optionally with `doc_id`.
- Batch and call InferMesh `/embed` for embeddings with retries and basic metrics.
- Emit enriched records to Kafka (`EMBED_OUTPUT_TOPIC`), Parquet (`EMBED_PARQUET_DIR`), and optional RedDB stream/KV.
- Expose an internal HTTP endpoint for health/metrics on `EMBED_METRICS_PORT`.

Record schema (JSON):
{
  "topic": str | null,
  "text": str,
  "doc_id": str,            # upstream-provided or sha256(text)
  "vector": number[],
  "kafka_ts": number | null,
  "embedded_ts": number,    # epoch seconds
  "model": str
}

Env vars:
- KAFKA_BOOTSTRAP_SERVERS, EMBED_INPUT_TOPIC, EMBED_OUTPUT_TOPIC, EMBED_GROUP
- INFERMESH_URL, EMBED_MODEL, INFERMESH_API_KEY (optional), INFERMESH_TIMEOUT_SEC, INFERMESH_RETRIES, INFERMESH_BACKOFF_SEC
- EMBED_BATCH_SIZE, EMBED_MAX_WAIT_SEC
- EMBED_WRITE_PARQUET=1, EMBED_PARQUET_DIR
- REDDB_URL, REDDB_NAMESPACE, REDDB_TOKEN, REDDB_STREAM, REDDB_MODE=stream|kv|both
- EMBED_METRICS_PORT (default 9100)
"""
import os
import json
import time
import hashlib
import threading
import signal
from typing import List, Dict, Any, Optional

from kafka import KafkaConsumer, KafkaProducer
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer


def now_ts() -> float:
    return time.time()


class TTLCache:
    def __init__(self, ttl_sec: int = 300, max_items: int = 100_000):
        self.ttl = ttl_sec
        self.max_items = max_items
        self.store: Dict[str, Any] = {}

    def get(self, key: str):
        ent = self.store.get(key)
        if not ent:
            return None
        v, exp = ent
        if time.time() > exp:
            self.store.pop(key, None)
            return None
        return v

    def put(self, key: str, val: Any):
        if len(self.store) >= self.max_items:
            # simple eviction: drop ~10%
            for i, k in enumerate(list(self.store.keys())):
                self.store.pop(k, None)
                if i > self.max_items // 10:
                    break
        self.store[key] = (val, time.time() + self.ttl)


def infermesh_embed(url: str, model: str, inputs: List[str], api_key: Optional[str] = None, timeout: float = 30.0) -> List[List[float]]:
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    payload = {'model': model, 'inputs': inputs}
    t0 = time.time()
    r = requests.post(url.rstrip('/') + '/embed', headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    dt = time.time() - t0
    data = r.json()
    # flexible: accept 'vectors' or 'embeddings'
    vecs = data.get('vectors') or data.get('embeddings') or []
    METRICS['last_infermesh_rtt_ms'] = int(dt * 1000)
    METRICS['total_infermesh_calls'] = METRICS.get('total_infermesh_calls', 0) + 1
    return [[float(x) for x in v] for v in vecs]


def embed_with_retries(url: str, model: str, inputs: List[str], api_key: Optional[str], *, timeout: float, retries: int, backoff: float) -> List[List[float]]:
    last_err: Optional[Exception] = None
    for attempt in range(max(1, retries + 1)):
        try:
            return infermesh_embed(url, model, inputs, api_key, timeout=timeout)
        except Exception as e:
            last_err = e
            METRICS['infermesh_failures'] = METRICS.get('infermesh_failures', 0) + 1
            if attempt >= retries:
                break
            time.sleep(backoff * (2 ** attempt))
    # On persistent failure, return empty vectors to avoid blocking the consumer
    return [[] for _ in inputs]


METRICS: Dict[str, Any] = {
    'started_ts': time.time(),
    'batches': 0,
    'records_in': 0,
    'records_out': 0,
    'last_flush_ts': None,
    'last_infermesh_rtt_ms': None,
    'total_infermesh_calls': 0,
    'infermesh_failures': 0,
    'dlq_records': 0,
}


def start_metrics_server(port: int = 9100):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # type: ignore[override]
            if self.path.startswith('/health'):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'ok', 'timestamp': time.time()}).encode('utf-8'))
                return
            if self.path.startswith('/metrics'):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(METRICS).encode('utf-8'))
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):  # type: ignore[override]
            return  # quiet

    srv = HTTPServer(('0.0.0.0', port), Handler)
    t = threading.Thread(target=srv.serve_forever, name='metrics-http', daemon=True)
    t.start()
    return srv


def main():
    bootstrap = os.getenv('KAFKA_BOOTSTRAP_SERVERS', os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'))
    in_topic = os.getenv('EMBED_INPUT_TOPIC', 'embedding_input')
    out_topic = os.getenv('EMBED_OUTPUT_TOPIC', 'embeddings')
    group = os.getenv('EMBED_GROUP', 'embed-worker')
    mesh_url = os.getenv('INFERMESH_URL', 'http://infermesh:9000')
    model = os.getenv('EMBED_MODEL', os.getenv('MODEL_ID', 'sentence-transformers/all-MiniLM-L6-v2'))
    api_key = os.getenv('INFERMESH_API_KEY')
    batch_size = int(os.getenv('EMBED_BATCH_SIZE', '64'))
    max_wait = float(os.getenv('EMBED_MAX_WAIT_SEC', '0.5'))
    write_parquet = os.getenv('EMBED_WRITE_PARQUET', '0') in ('1','true','yes','on')
    parquet_dir = os.getenv('EMBED_PARQUET_DIR', '/workspace/vectorized/embeddings_imesh')
    os.makedirs(parquet_dir, exist_ok=True)

    # Optional RedDB upsert/append
    REDDB_URL = os.getenv('REDDB_URL', '').strip()
    REDDB_NAMESPACE = os.getenv('REDDB_NAMESPACE', 'dspy').strip()
    REDDB_TOKEN = os.getenv('REDDB_TOKEN', '').strip()
    REDDB_STREAM = os.getenv('REDDB_STREAM', 'embeddings').strip()
    REDDB_MODE = os.getenv('REDDB_MODE', 'stream').strip()  # 'stream', 'kv', or 'both'

    session = requests.Session()
    def reddb_headers():
        h = {'Content-Type': 'application/json'}
        if REDDB_TOKEN:
            h['Authorization'] = f'Bearer {REDDB_TOKEN}'
        return h
    def reddb_append(value: Dict[str, Any]):
        if not REDDB_URL:
            return
        try:
            url = f"{REDDB_URL.rstrip('/')}/api/streams/{REDDB_NAMESPACE}/{REDDB_STREAM}/append"
            session.post(url, headers=reddb_headers(), data=json.dumps(value), timeout=3)
        except Exception:
            pass
    def reddb_put(key: str, value: Dict[str, Any]):
        if not REDDB_URL:
            return
        try:
            url = f"{REDDB_URL.rstrip('/')}/api/kv/{REDDB_NAMESPACE}/{key}"
            session.put(url, headers=reddb_headers(), data=json.dumps(value), timeout=3)
        except Exception:
            pass

    consumer = KafkaConsumer(
        in_topic,
        bootstrap_servers=bootstrap,
        group_id=group,
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode('utf-8', errors='ignore'))
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    dlq_topic = os.getenv('EMBED_DLQ_TOPIC', '').strip()

    cache = TTLCache(ttl_sec=int(os.getenv('EMBED_CACHE_TTL', '300')),
                     max_items=int(os.getenv('EMBED_CACHE_MAX', '200000')))

    # Start metrics HTTP server
    metrics_port = int(os.getenv('EMBED_METRICS_PORT', '9100'))
    start_metrics_server(metrics_port)

    # Optional: cache warmup from recent Parquet files to reduce duplicate embeds
    warmup_n = int(os.getenv('EMBED_CACHE_WARMUP', '0') or '0')
    if warmup_n > 0 and write_parquet:
        try:
            import pyarrow.parquet as pq
            import glob
            files = sorted(glob.glob(os.path.join(parquet_dir, '*.parquet')), key=lambda p: os.path.getmtime(p), reverse=True)
            loaded = 0
            for f in files:
                if loaded >= warmup_n:
                    break
                try:
                    pf = pq.ParquetFile(f)
                    batches = pf.iter_batches(batch_size=512)
                    for b in batches:
                        d = b.to_pydict()
                        texts = d.get('text') or []
                        vecs = d.get('vector') or []
                        for t, v in zip(texts, vecs):
                            if isinstance(t, str) and isinstance(v, list) and v:
                                cache.put(hashlib.sha256(t.encode('utf-8')).hexdigest(), v)
                                loaded += 1
                                if loaded >= warmup_n:
                                    break
                        if loaded >= warmup_n:
                            break
                except Exception:
                    continue
        except Exception:
            pass

    batch: List[Dict[str, Any]] = []
    last_flush = time.time()
    def flush():
        nonlocal batch, last_flush
        if not batch:
            return
        texts = [rec['text'] for rec in batch]
        # cache lookup
        keys = [hashlib.sha256(t.encode('utf-8')).hexdigest() for t in texts]
        results: List[List[float]] = [None] * len(texts)  # type: ignore
        to_query: List[int] = []
        for i, k in enumerate(keys):
            cached = cache.get(k)
            if cached is not None:
                results[i] = cached
            else:
                to_query.append(i)
        if to_query:
            # call infermesh for uncached with retries
            inputs = [texts[i] for i in to_query]
            vecs = embed_with_retries(
                mesh_url,
                model,
                inputs,
                api_key,
                timeout=float(os.getenv('INFERMESH_TIMEOUT_SEC', '30') or '30'),
                retries=int(os.getenv('INFERMESH_RETRIES', '2') or '2'),
                backoff=float(os.getenv('INFERMESH_BACKOFF_SEC', '0.5') or '0.5'),
            )
            failed = []
            for idx, i in enumerate(to_query):
                v = vecs[idx]
                # optional normalization and metadata
                dim = len(v)
                norm = 0.0
                try:
                    norm = (sum(float(x)*float(x) for x in v)) ** 0.5 if v else 0.0
                except Exception:
                    norm = 0.0
                normalize = os.getenv('EMBED_NORMALIZE', '0').lower() in ('1','true','yes','on')
                if normalize and norm > 0:
                    v = [float(x)/norm for x in v]
                results[i] = v
                cache.put(keys[i], results[i])
                if not results[i]:
                    failed.append(i)

        # publish to Kafka and optionally Parquet / RedDB
        now = now_ts()
        out_records = []
        for rec, vec in zip(batch, results):
            out = {
                'topic': rec.get('topic'),
                'text': rec['text'],
                'doc_id': rec.get('doc_id'),
                'vector': vec,
                'kafka_ts': rec.get('kafka_ts'),
                'embedded_ts': now,
                'model': model,
                'dim': len(vec) if isinstance(vec, list) else 0,
                'normalized': os.getenv('EMBED_NORMALIZE', '0').lower() in ('1','true','yes','on')
            }
            out_records.append(out)
            producer.send(out_topic, out)
        producer.flush()

        # DLQ handling for failed embeddings (empty vectors)
        if dlq_topic or REDDB_URL:
            for out in out_records:
                if not out['vector']:
                    dlq = {
                        'topic': out.get('topic'),
                        'text': out.get('text'),
                        'doc_id': out.get('doc_id'),
                        'kafka_ts': out.get('kafka_ts'),
                        'ts': now,
                        'model': model,
                        'reason': 'infermesh_failed_or_empty'
                    }
                    if dlq_topic:
                        try:
                            producer.send(dlq_topic, dlq)
                        except Exception:
                            pass
                    if REDDB_URL:
                        try:
                            url = f"{REDDB_URL.rstrip('/')}/api/streams/{REDDB_NAMESPACE}/{os.getenv('REDDB_DLQ_STREAM','embeddings_dlq').strip()}/append"
                            session.post(url, headers=reddb_headers(), data=json.dumps(dlq), timeout=3)
                        except Exception:
                            pass
                    METRICS['dlq_records'] = METRICS.get('dlq_records', 0) + 1

        if write_parquet:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pylist(out_records)
                fname = os.path.join(parquet_dir, f"part-{int(now)}-{len(out_records)}.parquet")
                pq.write_table(table, fname)
            except Exception:
                pass

        if REDDB_URL:
            for out in out_records:
                if REDDB_MODE in ('stream', 'both'):
                    reddb_append(out)
                if REDDB_MODE in ('kv', 'both'):
                    doc_key = (out.get('doc_id') or hashlib.sha256(out['text'].encode('utf-8')).hexdigest())
                    reddb_put(f"embedding:{doc_key}", out)

        batch = []
        last_flush = time.time()
        METRICS['batches'] += 1
        METRICS['records_out'] += len(out_records)
        METRICS['last_flush_ts'] = last_flush

    stop = False

    def handle_sigterm(signum, frame):  # flush before exiting
        nonlocal stop
        stop = True
        try:
            flush()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    for msg in consumer:
        if stop:
            break
        val = msg.value
        # Expect JSON with 'text', optionally 'topic', 'kafka_ts', 'doc_id'
        text = val.get('text') if isinstance(val, dict) else None
        if isinstance(text, str) and text.strip():
            # derive doc_id if upstream provided different keys
            doc_id = None
            if isinstance(val, dict):
                for k in ('doc_id', 'id', 'document_id', 'key'):
                    if isinstance(val.get(k), str) and val.get(k).strip():
                        doc_id = val.get(k).strip()
                        break
            if not doc_id:
                # fallback to text hash
                doc_id = hashlib.sha256(text.encode('utf-8')).hexdigest()
            batch.append({'text': text, 'doc_id': doc_id, 'topic': val.get('topic'), 'kafka_ts': val.get('kafka_ts')})
            METRICS['records_in'] += 1
        if len(batch) >= batch_size or (time.time() - last_flush) >= max_wait:
            flush()


if __name__ == '__main__':
    main()
