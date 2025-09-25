from __future__ import annotations

import os
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from streaming.utils.spark import create_spark
from streaming.utils.extract_text import extract_text_from_bytes


def main() -> None:
    spark = create_spark("file_ingest_docs")

    src_dir = os.getenv("INGEST_DOCS_SRC", "/data/landing/docs")
    out_base = os.getenv("WAREHOUSE_BASE", "./warehouse")
    bronze = os.path.join(out_base, "bronze", "docs_raw")
    silver = os.path.join(out_base, "silver", "docs_normalized")
    pending_dir = os.path.join(out_base, "silver", "docs_pending")
    require_approval = os.getenv("REQUIRE_APPROVAL", "false").lower() == "true"
    max_files = int(os.getenv("MAX_FILES_PER_TRIGGER", "50"))

    # Binary file stream: returns path, modificationTime, length, content
    stream_src = (
        spark.readStream.format("binaryFile")
        .option("path", src_dir)
        .option("recursiveFileLookup", "true")
        .option("maxFilesPerTrigger", max_files)
        .load()
    )

    # UDF extraction: use path + content
    @F.udf(StringType())
    def extract_udf(path: str, content: bytes) -> str:
        try:
            return extract_text_from_bytes(path, content)[:1_000_000]
        except Exception:
            return ''

    def process_batch(df, epoch_id):
        if df.rdd.isEmpty():
            return
        # bronze write (no content to save space; keep length/mtime/path)
        bronze_df = df.select(
            F.col('path').alias('source_file'),
            F.col('length').alias('size_bytes'),
            F.col('modificationTime').alias('mtime'),
        ).withColumn('ingest_ts', F.current_timestamp())
        bronze_df.write.mode('append').parquet(bronze)

        # Extract text to silver/pending
        enriched = df.select(
            F.col('path').alias('source_file'),
            F.col('length').alias('size_bytes'),
            F.col('modificationTime').alias('mtime'),
            F.lower(F.element_at(F.split(F.col('path'), '\\.'), -1)).alias('ext'),
            extract_udf(F.col('path'), F.col('content')).alias('text'),
        )
        if require_approval:
            pending = (
                enriched
                .groupBy('source_file')
                .agg(F.max('size_bytes').alias('size_bytes'))
                .withColumn('status', F.lit('pending'))
                .withColumn('ingest_ts', F.current_timestamp())
            )
            pending.write.mode('append').parquet(pending_dir)
        else:
            normalized = (
                enriched
                .withColumn('char_count', F.length(F.col('text')))
                .withColumn('ingest_ts', F.current_timestamp())
            )
            normalized.write.mode('append').parquet(silver)

    (
        stream_src.writeStream.outputMode('append')
        .foreachBatch(process_batch)
        .option('checkpointLocation', os.path.join(out_base, 'checkpoints', 'file_ingest_docs'))
        .start()
        .awaitTermination()
    )


if __name__ == '__main__':
    main()

