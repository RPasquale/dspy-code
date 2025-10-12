#!/usr/bin/env python3
import os
import json
import time
import hashlib
import logging
from typing import Dict, Any

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable, KafkaError
import requests
import math

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [emb-indexer] %(message)s",
)
logger = logging.getLogger("dspy.embeddings.indexer")


def main():
    bootstrap = os.getenv('KAFKA_BOOTSTRAP_SERVERS', os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'))
    topic = os.getenv('EMB_INDEX_TOPIC', 'embeddings')
    group = os.getenv('EMB_INDEX_GROUP', 'emb-indexer')

    REDDB_URL = os.getenv('REDDB_URL', '').strip()
    REDDB_NAMESPACE = os.getenv('REDDB_NAMESPACE', 'dspy').strip()
    REDDB_TOKEN = os.getenv('REDDB_TOKEN', '').strip()
    REDDB_INDEX_STREAM = os.getenv('REDDB_INDEX_STREAM', 'emb_index').strip()
    SHARDS = int(os.getenv('EMB_INDEX_SHARDS', '32') or '32')
    SHARD_MAX_IDS = int(os.getenv('EMB_SHARD_MAX_IDS', '5000') or '5000')

    sess = requests.Session()
    headers = {'Content-Type': 'application/json'}
    if REDDB_TOKEN:
        headers['Authorization'] = f'Bearer {REDDB_TOKEN}'

    def reddb_append(value: Dict[str, Any]):
        if not REDDB_URL:
            return
        try:
            url = f"{REDDB_URL.rstrip('/')}/api/streams/{REDDB_NAMESPACE}/{REDDB_INDEX_STREAM}/append"
            sess.post(url, headers=headers, data=json.dumps(value), timeout=3)
        except Exception:
            pass

    def reddb_put(key: str, value: Dict[str, Any]):
        if not REDDB_URL:
            return
        try:
            url = f"{REDDB_URL.rstrip('/')}/api/kv/{REDDB_NAMESPACE}/{key}"
            sess.put(url, headers=headers, data=json.dumps(value), timeout=3)
        except Exception:
            pass

    def reddb_get(key: str):
        if not REDDB_URL:
            return None
        try:
            url = f"{REDDB_URL.rstrip('/')}/api/kv/{REDDB_NAMESPACE}/{key}"
            r = sess.get(url, headers=headers, timeout=3)
            if r.status_code == 200 and r.text:
                try:
                    return r.json()
                except Exception:
                    return None
        except Exception:
            return None
        return None

    def build_consumer() -> KafkaConsumer:
        delay = 1.0
        while True:
            try:
                consumer = KafkaConsumer(
                    topic,
                    bootstrap_servers=bootstrap,
                    group_id=group,
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    value_deserializer=lambda v: json.loads(v.decode('utf-8', errors='ignore'))
                )
                logger.info("Connected to Kafka at %s (topic=%s, group=%s)", bootstrap, topic, group)
                return consumer
            except NoBrokersAvailable:
                logger.warning(
                    "Kafka brokers not yet available at %s (topic=%s); retrying in %.1fs",
                    bootstrap,
                    topic,
                    delay,
                )
            except KafkaError as exc:
                logger.warning(
                    "Kafka error while connecting to %s (topic=%s): %s; retrying in %.1fs",
                    bootstrap,
                    topic,
                    exc,
                    delay,
                )
            except Exception as exc:  # pragma: no cover - safety log
                logger.exception(
                    "Unexpected error creating Kafka consumer for %s; retrying in %.1fs",
                    bootstrap,
                    delay,
                )
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

    consumer = build_consumer()

    try:
        while True:
            try:
                msg = next(consumer)
            except StopIteration:  # pragma: no cover - kafka iterator contract
                continue
            except (KafkaError, NoBrokersAvailable) as exc:
                logger.warning("Kafka consumer error: %s; reconnecting", exc)
                consumer.close(autocommit=False)
                consumer = build_consumer()
                continue
            except Exception as exc:
                logger.exception("Unexpected consumer error, continuing: %s", exc)
                continue

            val = msg.value if isinstance(msg.value, dict) else {}
            text = val.get('text')
            vec = val.get('vector')
            if not isinstance(text, str) or not isinstance(vec, list) or not vec:
                continue
            doc_id = val.get('doc_id')
            if not isinstance(doc_id, str) or not doc_id.strip():
                doc_id = hashlib.sha256(text.encode('utf-8')).hexdigest()
            # compute norm and unit vector for fast cosine
            norm = math.sqrt(sum(float(x) * float(x) for x in vec)) or 1.0
            unit = [float(x) / norm for x in vec]
            record = {
                'doc_id': doc_id,
                'vector': vec,
                'norm': norm,
                'unit': unit,
                'topic': val.get('topic'),
                'kafka_ts': val.get('kafka_ts'),
                'embedded_ts': val.get('embedded_ts'),
                'model': val.get('model'),
            }
            reddb_append(record)
            reddb_put(f'embvec:{doc_id}', record)

            # shard assignment
            try:
                shard = int(hashlib.sha256(doc_id.encode('utf-8')).hexdigest(), 16) % max(1, SHARDS)
            except Exception:
                shard = 0
            # maintain shard id list for simple kNN candidate search
            ids_key = f'shard:{shard}:ids'
            ids = reddb_get(ids_key)
            if not isinstance(ids, list):
                ids = []
            if doc_id not in ids:
                ids.append(doc_id)
                if len(ids) > SHARD_MAX_IDS:
                    ids = ids[-SHARD_MAX_IDS:]
                try:
                    reddb_put(ids_key, ids)
                except Exception:
                    pass
    except KeyboardInterrupt:
        logger.info("Received interrupt; shutting down embedding indexer")
    finally:
        try:
            consumer.close(autocommit=False)
        except Exception:
            pass


if __name__ == '__main__':
    main()
