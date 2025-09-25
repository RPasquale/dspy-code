from __future__ import annotations

import os
from pyspark.sql import functions as F

from streaming.utils.spark import create_spark
from .schemas import tool_to_diff_map_schema, kafka_json_options


def main() -> None:
    spark = create_spark("tool_map_ingest")

    bootstrap = os.getenv("KAFKA_BOOTSTRAP")
    topic_map = os.getenv("TOPIC_TOOL_TO_DIFF_MAP", "tool_to_diff_map")
    checkpoint_base = os.getenv("CHECKPOINT_BASE", "./checkpoints")
    warehouse_base = os.getenv("WAREHOUSE_BASE", "./warehouse")

    if not bootstrap:
        raise SystemExit("KAFKA_BOOTSTRAP not set")

    opts = kafka_json_options(bootstrap)
    raw = spark.readStream.format("kafka").options(**opts).option("subscribe", topic_map).load()
    js = F.from_json(F.col("value").cast("string"), tool_to_diff_map_schema)
    df = raw.select(js.alias("j")).select("j.*")
    df = df.withWatermark("ts", "30 minutes").dropDuplicates([
        "tenant_id", "task_id", "tool_group_id", "seq_id", "group_id", "sample_id"
    ])

    out_path = os.path.join(warehouse_base, "silver", "tool_to_diff_map")
    ckpt_path = os.path.join(checkpoint_base, "tool_map_ingest")

    (
        df.writeStream.outputMode("append")
        .format("parquet")
        .option("checkpointLocation", ckpt_path)
        .option("path", out_path)
        .trigger(processingTime="30 seconds")
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    main()

