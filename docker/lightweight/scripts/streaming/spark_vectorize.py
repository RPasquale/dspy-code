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
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType
import os
import json
import hashlib


def main():
    bootstrap = os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092')
    topics = os.getenv('SPARK_KAFKA_TOPICS', 'agent.results')
    out_dir = os.getenv('VEC_OUTPUT_DIR', '/workspace/vectorized/embeddings')
    ckpt_dir = os.getenv('VEC_CHECKPOINT', '/workspace/.dspy_checkpoints/vectorizer')
    sink_kafka = os.getenv('SINK_TO_KAFKA', '0') in ('1', 'true', 'yes', 'on')

    spark = (
        SparkSession.builder.appName('kafka_vectorizer')
        .config('spark.sql.shuffle.partitions', '4')
        .getOrCreate()
    )

    # Read from Kafka
    df = (
        spark.readStream.format('kafka')
        .option('kafka.bootstrap.servers', bootstrap)
        .option('subscribe', topics)
        .option('startingOffsets', 'latest')
        .load()
    )

    # Parse messages (JSON if possible)
    def parse_value(b):
        try:
            s = b.decode('utf-8', errors='ignore')
            j = json.loads(s)
            # Common fields for text-like payloads
            for k in ('text', 'message', 'content', 'prompt', 'body'):
                if isinstance(j, dict) and k in j and isinstance(j[k], str):
                    return j[k]
            return s
        except Exception:
            try:
                return b.decode('utf-8', errors='ignore')
            except Exception:
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

    base = df.select(
        F.col('topic'),
        F.col('timestamp').alias('kafka_ts'),
        parse_value_udf(F.col('value')).alias('text')
    )

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

    typed = base.withColumn('doc_id_raw', doc_from_raw_udf(F.col('value')))
    typed = typed.withColumn('doc_id', F.when(F.length(F.col('doc_id_raw')) > 0, F.col('doc_id_raw')).otherwise(sha_udf(F.col('text'))))

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
    vec = typed.withColumn('vector', vec_col)

    # Write to Parquet sink for training
    (vec
        .writeStream
        .format('parquet')
        .option('path', out_dir)
        .option('checkpointLocation', ckpt_dir)
        .outputMode('append')
        .start()
    )

    # Optional: also publish parsed text to an input topic for external embed service
    input_topic = os.getenv('SINK_INPUT_TOPIC')
    tx_base = os.getenv('KAFKA_TX_ID_BASE', 'spark-vectorizer')
    if input_topic:
        parsed = typed.select(F.to_json(F.struct('topic', 'kafka_ts', 'text', 'doc_id')).alias('value'))
        (parsed
            .writeStream
            .format('kafka')
            .option('kafka.bootstrap.servers', bootstrap)
            .option('topic', input_topic)
            .option('kafka.transactional.id', f"{tx_base}-input")
            .option('kafka.enable.idempotence', 'true')
            .option('kafka.acks', 'all')
            .option('checkpointLocation', ckpt_dir + '_input')
            .outputMode('append')
            .start())

    # Optional: also publish to Kafka embeddings topic (vector serialized as JSON)
    if sink_kafka:
        out_kafka = (vec
                     .select(F.to_json(F.struct('topic', 'kafka_ts', 'text', 'doc_id', 'vector')).alias('value'))
                     .writeStream
                     .format('kafka')
                     .option('kafka.bootstrap.servers', bootstrap)
                     .option('topic', 'embeddings')
                     .option('kafka.transactional.id', f"{tx_base}-embeddings")
                     .option('kafka.enable.idempotence', 'true')
                     .option('kafka.acks', 'all')
                     .option('checkpointLocation', ckpt_dir + '_kafka')
                     .outputMode('append')
                     .start())
        out_kafka.awaitTermination()

    spark.streams.awaitAnyTermination()


if __name__ == '__main__':
    main()
