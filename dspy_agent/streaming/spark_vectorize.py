#!/usr/bin/env python3
"""
Spark Structured Streaming job to parse and/or vectorize Kafka topic messages.

Default (recommended): parse-only + publish to 'embedding_input', so a separate
embedding microservice (e.g., InferMesh) handles scalable embeddings.

When publishing to `SINK_INPUT_TOPIC`, each record is JSON with:
  { "text": str, "topic": str, "kafka_ts": ts, "doc_id": str }
doc_id will be propagated from upstream if present (doc_id/id/document_id/key),
otherwise it's derived as sha256(text).

If you need built-in vectors, enable hashing/ST embeddings and vector sinks.

Environment variables:
  KAFKA_BOOTSTRAP      - default 'kafka:9092'
  SPARK_KAFKA_TOPICS   - comma-separated topics, default 'agent.results'
  VEC_OUTPUT_DIR       - Parquet output for vectors, default '/workspace/vectorized/embeddings'
  VEC_CHECKPOINT       - checkpoint directory, default '/workspace/.dspy_checkpoints/vectorizer'
  SINK_INPUT_TOPIC     - publish parsed JSON {text, topic, kafka_ts} to this Kafka topic (e.g., 'embedding_input')
  SINK_TO_KAFKA        - if '1', publish vectors JSON to Kafka topic 'embeddings'
  USE_SENTENCE_TRANSFORMERS - if '1', use sentence-transformers; else hashing vectors
  ST_MODEL             - sentence-transformers model (default 'all-MiniLM-L6-v2')

"""
import asyncio
import hashlib
import json
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType

from ..infra.runtime import ensure_infra, ensure_infra_sync


def ensure_kafka_topics(bootstrap: str, topics, partitions: int = 3, replication_factor: int = 1) -> None:
    """Best-effort creation of required Kafka topics before streaming starts."""
    try:
        from kafka.admin import KafkaAdminClient, NewTopic  # type: ignore
        from kafka.errors import TopicAlreadyExistsError  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[spark-vectorizer] Kafka admin client unavailable ({exc}); skipping topic creation")
        return

    deduped = sorted(set(t.strip() for t in topics if t))
    if not deduped:
        return

    try:
        admin = KafkaAdminClient(bootstrap_servers=bootstrap, client_id="dspy-spark-vectorizer-init")
    except Exception as exc:
        print(f"[spark-vectorizer] Unable to init KafkaAdminClient: {exc}")
        return

    new_topics = [NewTopic(name=name, num_partitions=partitions, replication_factor=replication_factor) for name in deduped]
    try:
        futures = admin.create_topics(new_topics, validate_only=False)
    except Exception as exc:
        print(f"[spark-vectorizer] create_topics failed: {exc}")
        admin.close()
        return

    for name, future in futures.items():
        try:
            future.result()
            print(f"[spark-vectorizer] Created topic {name}")
        except TopicAlreadyExistsError:
            print(f"[spark-vectorizer] Topic {name} already exists")
        except Exception as exc:
            print(f"[spark-vectorizer] Topic {name} create failed: {exc}")

    admin.close()


def _resolve_bootstrap(raw: str) -> str:
    """Normalize Kafka bootstrap list for container + host environments."""
    alias = os.getenv('DSPY_KAFKA_LOCAL_ALIAS', 'kafka')
    hosts = []
    for token in (raw or '').split(','):
        token = token.strip()
        if not token:
            continue
        scheme = ''
        rest = token
        if '://' in token:
            scheme, rest = token.split('://', 1)
        host = rest
        port = ''
        if rest.startswith('[') and ']' in rest:
            bracket, after = rest.split(']', 1)
            host = bracket[1:]
            if after.startswith(':'):
                port = after[1:]
        elif rest.count(':') == 1:
            host, port = rest.split(':', 1)
        local_hosts = {'localhost', '127.0.0.1', '0.0.0.0'}
        if host in local_hosts and alias:
            host = alias
        rebuilt = host
        if port:
            rebuilt = f"{host}:{port}"
        if scheme:
            rebuilt = f"{scheme}://{rebuilt}"
        hosts.append(rebuilt)
    return ','.join(hosts) if hosts else raw


def wait_for_kafka(bootstrap, max_retries=30, retry_delay=2):
    """Wait for Kafka to be ready before starting Spark"""
    import socket
    import time
    
    print(f"[spark-vectorizer] Waiting for Kafka to be ready at {bootstrap}...")
    
    for attempt in range(max_retries):
        try:
            # Parse bootstrap servers
            servers = bootstrap.split(',')
            for server in servers:
                host, port = server.strip().split(':')
                port = int(port)
                
                # Test connection to each server
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result != 0:
                    print(f"[spark-vectorizer] Attempt {attempt + 1}/{max_retries}: {host}:{port} not ready")
                    time.sleep(retry_delay)
                    break
            else:
                print(f"[spark-vectorizer] Kafka is ready at {bootstrap}")
                return True
        except Exception as e:
            print(f"[spark-vectorizer] Attempt {attempt + 1}/{max_retries}: Error checking Kafka - {e}")
            time.sleep(retry_delay)
    
    print(f"[spark-vectorizer] WARNING: Kafka not ready after {max_retries} attempts, proceeding anyway...")
    return False

def main():
    # Ensure the unified infrastructure layer is online before the Spark job starts.
    try:
        ensure_infra_sync()
    except RuntimeError:
        asyncio.get_running_loop().create_task(ensure_infra(auto_start_services=True))

    bootstrap = os.getenv('KAFKA_BOOTSTRAP') or os.getenv('KAFKA_BOOTSTRAP_SERVERS') or 'kafka:9092'
    print(f"[spark-vectorizer] Original bootstrap: {bootstrap}")
    bootstrap = _resolve_bootstrap(bootstrap)
    print(f"[spark-vectorizer] Resolved bootstrap: {bootstrap}")
    topics = os.getenv('SPARK_KAFKA_TOPICS', 'agent.results')
    out_dir = os.getenv('VEC_OUTPUT_DIR', '/workspace/vectorized/embeddings')
    ckpt_dir = os.getenv('VEC_CHECKPOINT', '/workspace/.dspy_checkpoints/vectorizer')
    sink_kafka = os.getenv('SINK_TO_KAFKA', '0') in ('1', 'true', 'yes', 'on')
    input_topic = os.getenv('SINK_INPUT_TOPIC')
    embeddings_topic = os.getenv('VECTOR_SINK_TOPIC', 'embeddings')
    partitions = int(os.getenv('VECTOR_TOPIC_PARTITIONS', os.getenv('EMBED_TOPIC_PARTITIONS', '3')) or '3')
    replication = int(os.getenv('VECTOR_TOPIC_REPLICATION', '1') or '1')

    topic_names = {t.strip() for t in topics.split(',') if t.strip()}
    if input_topic:
        topic_names.add(input_topic.strip())
    if sink_kafka and embeddings_topic:
        topic_names.add(embeddings_topic.strip())

    # Wait for Kafka to be ready
    wait_for_kafka(bootstrap)
    ensure_kafka_topics(bootstrap, topic_names, partitions=partitions, replication_factor=replication)

    spark = (
        SparkSession.builder.appName('kafka_vectorizer')
        .config('spark.sql.shuffle.partitions', '4')
        # Add Kafka timeout configurations
        .config('spark.sql.streaming.kafka.consumer.poll.timeoutMs', '30000')
        .config('spark.sql.streaming.kafka.consumer.request.timeoutMs', '30000')
        .config('spark.sql.streaming.kafka.consumer.metadata.max.age.ms', '300000')
        .config('spark.sql.streaming.kafka.consumer.reconnect.backoff.max.ms', '10000')
        .config('spark.sql.streaming.kafka.consumer.reconnect.backoff.ms', '1000')
        .getOrCreate()
    )

    # Read from Kafka with enhanced timeout and retry configuration
    df = (
        spark.readStream.format('kafka')
        .option('kafka.bootstrap.servers', bootstrap)
        .option('subscribe', topics)
        .option('startingOffsets', 'earliest')
        .option('failOnDataLoss', 'false')
        # Enhanced timeout and retry configuration
        .option('kafka.request.timeout.ms', '30000')
        .option('kafka.metadata.max.age.ms', '300000')
        .option('kafka.retries', '10')
        .option('kafka.retry.backoff.ms', '1000')
        .option('kafka.reconnect.backoff.ms', '1000')
        .option('kafka.reconnect.backoff.max.ms', '10000')
        .option('kafka.connections.max.idle.ms', '300000')
        .option('kafka.socket.timeout.ms', '30000')
        .option('kafka.socket.connection.setup.timeout.ms', '30000')
        .option('kafka.socket.connection.setup.timeout.max.ms', '30000')
        .load()
    )

    # Parse messages (JSON if possible)
    def parse_value(b):
        try:
            print(f"[DEBUG] Processing message: {b}")
            s = b.decode('utf-8', errors='ignore')
            print(f"[DEBUG] Decoded string: {s}")
            j = json.loads(s)
            print(f"[DEBUG] Parsed JSON: {j}")
            # Common fields for text-like payloads
            for k in ('text', 'message', 'content', 'prompt', 'body'):
                if isinstance(j, dict) and k in j and isinstance(j[k], str):
                    result = j[k]
                    print(f"[DEBUG] Found text field '{k}': {result}")
                    return result
            # Handle code.fs.events messages - use path as text content
            if isinstance(j, dict) and 'path' in j:
                result = f"File {j['path']} was {j.get('event', 'modified')}"
                print(f"[DEBUG] Code.fs.events message: {result}")
                return result
            print(f"[DEBUG] No recognized fields, returning raw string: {s}")
            return s
        except Exception as e:
            print(f"[DEBUG] JSON parsing failed: {e}")
            try:
                result = b.decode('utf-8', errors='ignore')
                print(f"[DEBUG] Fallback to raw decode: {result}")
                return result
            except Exception as e2:
                print(f"[DEBUG] Raw decode also failed: {e2}")
                return ''

    parse_value_udf = F.udf(lambda b: parse_value(b) if b is not None else '', StringType())

    # Simple tokenizer
    def tokenize(s: str):
        if not s:
            return []
        return [t for t in F.regexp_replace(F.lit(s), r'[^A-Za-z0-9]+', ' ').cast('string')._jc.toString().split() if t]

    # Hashing-based vectorizer (deterministic)
    import math
    import hashlib

    def hash_vector(tokens, dim=256):
        vec = [0.0] * dim
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    hash_vec_udf = F.udf(lambda s: hash_vector((s or '').split()), ArrayType(DoubleType()))


    # Derive doc_id:
    # 1) If upstream provided JSON containing doc_id/id/document_id/key, try to parse from raw value
    # 2) Fallback to sha256(text)
    def extract_doc_id(raw: bytes) -> str:
        try:
            s = raw.decode('utf-8', errors='ignore')
            j = json.loads(s)
            if isinstance(j, dict):
                for k in ('doc_id', 'id', 'document_id', 'key'):
                    v = j.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except Exception:
            pass
        return ''

    doc_from_raw_udf = F.udf(lambda b: extract_doc_id(b) if b is not None else '', StringType())
    sha_udf = F.udf(lambda s: hashlib.sha256((s or '').encode('utf-8')).hexdigest(), StringType())

    # We need to access the original 'value' column from the base DataFrame
    # Let's modify the base selection to include both value and text
    print("[DEBUG] Creating base DataFrame...")
    base = df.select(
        F.col('topic'),
        F.col('timestamp').alias('kafka_ts'),
        F.col('value'),  # Keep original value for doc_id extraction
        parse_value_udf(F.col('value')).alias('text')
    )
    print("[DEBUG] Base DataFrame created, adding doc_id columns...")
    
    typed = base.withColumn('doc_id_raw', doc_from_raw_udf(F.col('value')))
    typed = typed.withColumn('doc_id', F.when(F.length(F.col('doc_id_raw')) > 0, F.col('doc_id_raw')).otherwise(sha_udf(F.col('text'))))
    print("[DEBUG] Doc ID columns added")

    # Optional: learned embeddings via sentence-transformers
    use_st = os.getenv('USE_SENTENCE_TRANSFORMERS', '0').lower() in ('1', 'true', 'yes', 'on')
    vec_col = None
    if use_st:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(os.getenv('ST_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
            import numpy as np
            def st_embed(s: str):
                try:
                    if not s:
                        return []
                    v = _model.encode([s], normalize_embeddings=True)
                    return [float(x) for x in (v[0].tolist() if hasattr(v, 'tolist') else list(v[0]))]
                except Exception:
                    return []
            st_udf = F.udf(lambda s: st_embed(s), ArrayType(DoubleType()))
            vec_col = st_udf(F.col('text'))
        except Exception:
            vec_col = hash_vec_udf(F.col('text'))
    else:
        vec_col = hash_vec_udf(F.col('text'))

    # Build vectors
    print("[DEBUG] Building vectors...")
    vec = typed.withColumn('vector', vec_col)
    print("[DEBUG] Vectors built")

    # Write to Parquet sink for training
    print(f"[DEBUG] Starting Parquet sink to {out_dir} with checkpoint {ckpt_dir}")
    (vec
        .writeStream
        .format('parquet')
        .option('path', out_dir)
        .option('checkpointLocation', ckpt_dir)
        .outputMode('append')
        .start()
    )
    print("[DEBUG] Parquet sink started")

    # Optional: also publish parsed text to an input topic for external embed service
    tx_base = os.getenv('KAFKA_TX_ID_BASE', 'spark-vectorizer')
    if input_topic:
        print(f"[DEBUG] Kafka sink disabled - using separate parquet-to-kafka script instead")
        # Kafka sink disabled due to transaction issues - using separate script
        # The parquet-to-kafka.py script will read from Parquet files and publish to Kafka

    # Optional: also publish to Kafka embeddings topic (vector serialized as JSON)
    if sink_kafka:
        print(f"[DEBUG] Second Kafka sink disabled for debugging - focusing on Parquet sink only")
        # Temporarily disable second Kafka sink to debug the issue
        # out_kafka = (vec
        #              .select(F.to_json(F.struct('topic', 'kafka_ts', 'text', 'doc_id', 'vector')).alias('value'))
        #              .writeStream
        #              .format('kafka')
        #              .option('kafka.bootstrap.servers', bootstrap)
        #              .option('topic', embeddings_topic)
        #              .option('kafka.acks', 'all')
        #              .option('checkpointLocation', ckpt_dir + '_kafka')
        #              .outputMode('append')
        #              .start())
        # out_kafka.awaitTermination()

    spark.streams.awaitAnyTermination()


if __name__ == '__main__':
    main()
