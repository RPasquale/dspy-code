from __future__ import annotations

import os
from pyspark.sql import functions as F
from streaming.utils.spark import create_spark


def _persist_partition(rows):
    # Runs on executor
    try:
        from dspy_agent.db.enhanced_storage import get_enhanced_data_manager  # type: ignore
        from dspy_agent.db.data_models import create_action_record, ActionType, Environment  # type: ignore
        dm = get_enhanced_data_manager()
    except Exception:
        dm = None
    count = 0
    if dm is None:
        return 0
    for r in rows:
        try:
            record = {
                'source_file': r['source_file'],
                'ext': r.get('ext'),
                'char_count': int(r.get('char_count') or 0),
                'timestamp': float(r.get('ingest_ts').timestamp()) if r.get('ingest_ts') else None,
            }
            dm.storage.append('ingested_docs', record)  # type: ignore
            # Action record for learning
            try:
                rec = create_action_record(
                    action_type=ActionType.TOOL_SELECTION,
                    state_before={'source_file': r['source_file'], 'ext': r.get('ext')},
                    state_after={'char_count': record['char_count']},
                    parameters={'mode': 'doc_ingest'},
                    result={'selected': 'doc_ingest'},
                    reward=1.0 if record['char_count'] and record['char_count'] > 0 else 0.0,
                    confidence=1.0,
                    execution_time=0.0,
                    environment=Environment.DEVELOPMENT,
                )
                dm.record_action(rec)  # type: ignore
            except Exception:
                pass
            count += 1
        except Exception:
            continue
    return count


def main() -> None:
    spark = create_spark('persist_docs_to_reddb')
    base = os.getenv('WAREHOUSE_BASE', './warehouse')
    src = os.path.join(base, 'silver', 'docs_normalized')
    if not os.path.exists(src):
        print('No docs_normalized found at', src)
        return
    df = spark.read.parquet(src).select('source_file','ext','char_count','ingest_ts')
    df = df.dropna(subset=['source_file'])
    # Dedup to reduce chatter
    df = df.dropDuplicates(['source_file'])
    # Persist per-partition
    total = df.rdd.mapPartitions(_persist_partition).sum()
    print('Persisted docs:', total)


if __name__ == '__main__':
    main()

