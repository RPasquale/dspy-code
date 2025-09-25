from __future__ import annotations

import os
from pyspark.sql import functions as F
from streaming.utils.spark import create_spark
from streaming.utils.normalize import normalize_frame


def main() -> None:
    spark = create_spark("file_ingest_parquet")

    src_dir = os.getenv("INGEST_PARQUET_SRC", "/data/landing/parquet")
    out_base = os.getenv("WAREHOUSE_BASE", "./warehouse")
    out_bronze = os.path.join(out_base, "bronze", "parquet_raw")
    out_silver = os.path.join(out_base, "silver", "parquet_normalized")
    out_pending = os.path.join(out_base, "silver", "parquet_pending")
    max_files = int(os.getenv("MAX_FILES_PER_TRIGGER", "100"))
    require_approval = os.getenv("REQUIRE_APPROVAL", "false").lower() == "true"

    stream_src = (
        spark.readStream.format("parquet")
        .option("path", src_dir)
        .option("recursiveFileLookup", "true")
        .option("maxFilesPerTrigger", max_files)
        .load()
        .withColumn("source_file", F.input_file_name())
    )

    def process_batch(df, epoch_id):
        if df.rdd.isEmpty():
            return
        df.write.mode("append").parquet(out_bronze)
        if require_approval:
            pending = (
                df.groupBy("source_file").agg(F.count(F.lit(1)).alias("rows")).withColumn("status", F.lit("pending")).withColumn("ingest_ts", F.current_timestamp())
            )
            pending.write.mode("append").parquet(out_pending)
        else:
            normalized = normalize_frame(df).withColumn("ingest_ts", F.current_timestamp())
            normalized.write.mode("append").parquet(out_silver)

    (
        stream_src.writeStream.outputMode("append")
        .foreachBatch(process_batch)
        .option("checkpointLocation", os.path.join(out_base, "checkpoints", "file_ingest_parquet"))
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    main()

