from __future__ import annotations

import os
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
from streaming.utils.spark import create_spark
from streaming.utils.normalize import normalize_frame


def main() -> None:
    spark = create_spark("file_ingest_csv")

    src_dir = os.getenv("INGEST_SRC", "/data/landing/csv")
    out_base = os.getenv("WAREHOUSE_BASE", "./warehouse")
    out_bronze = os.path.join(out_base, "bronze", "files_raw")
    out_silver = os.path.join(out_base, "silver", "files_normalized")
    out_pending = os.path.join(out_base, "silver", "files_pending")
    max_files = int(os.getenv("MAX_FILES_PER_TRIGGER", "100"))
    header = os.getenv("CSV_HEADER", "true")
    sep = os.getenv("CSV_SEP", ",")

    # Use a permissive schema: all strings; cast later
    try:
        ncols = int(os.getenv("CSV_COLUMNS", "0"))
    except Exception:
        ncols = 0
    if ncols > 0:
        fields = [F.lit("").cast("string").alias(f"col{i}") for i in range(ncols)]
        schema = StructType.fromJson({"type": "struct", "fields": [{"name": f"col{i}", "type": "string", "nullable": True, "metadata": {}} for i in range(ncols)]})  # type: ignore
    else:
        schema = None  # infer columns count dynamically; Spark requires a schema for streaming; fallback to batch in foreachBatch.

    # Pattern: use foreachBatch to load newly discovered files as batch CSV with inferSchema
    stream_src = (
        spark.readStream.format("text")
        .option("path", src_dir)
        .option("recursiveFileLookup", "true")
        .option("maxFilesPerTrigger", max_files)
        .load()
        .withColumn("_path", F.input_file_name())
    )

    require_approval = os.getenv("REQUIRE_APPROVAL", "false").lower() == "true"

    def process_batch(df, epoch_id):
        files = [r[0] for r in df.select("_path").distinct().collect()]
        if not files:
            return
        batch = (
            spark.read.format("csv")
            .option("header", header)
            .option("inferSchema", "true")
            .option("sep", sep)
            .load(files)
            .withColumn("source_file", F.input_file_name())
        )
        # bronze
        (
            batch.write.mode("append").parquet(out_bronze)
        )
        if require_approval:
            pending = (
                batch
                .groupBy("source_file")
                .agg(F.count(F.lit(1)).alias("rows"))
                .withColumn("status", F.lit("pending"))
                .withColumn("ingest_ts", F.current_timestamp())
            )
            pending.write.mode("append").parquet(out_pending)
        else:
            # normalize to silver immediately
            normalized = normalize_frame(batch).withColumn("ingest_ts", F.current_timestamp())
            (
                normalized.write.mode("append").parquet(out_silver)
            )

    (
        stream_src.writeStream.outputMode("update")
        .foreachBatch(process_batch)
        .option("checkpointLocation", os.path.join(out_base, "checkpoints", "file_ingest_csv"))
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    main()
