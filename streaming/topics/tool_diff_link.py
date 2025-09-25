from __future__ import annotations

import os
from pyspark.sql import functions as F

from streaming.utils.spark import create_spark
from .schemas import tool_to_diff_map_schema


def main() -> None:
    spark = create_spark("tool_diff_link")

    warehouse_base = os.getenv("WAREHOUSE_BASE", "./warehouse")
    # Read mapping (parquet or Kafka); here assume parquet persisted elsewhere
    mapping_src = os.path.join(warehouse_base, "silver", "tool_to_diff_map")
    diff_joined_src = os.path.join(warehouse_base, "silver", "grpo_joined")
    out_path = os.path.join(warehouse_base, "silver", "tool_to_diff_linked")

    # Fallback: if no parquet mapping exists, try to read from Kafka JSON (optional)
    if os.path.exists(mapping_src):
        link = spark.read.parquet(mapping_src)
    else:
        # Minimal skeleton: empty frame with expected columns
        link = spark.createDataFrame([], schema=tool_to_diff_map_schema)

    # Join with global grpo_joined to fetch downstream diff scores
    try:
        grpo = spark.read.parquet(diff_joined_src)
    except Exception:
        grpo = spark.createDataFrame([], schema=link.schema)

    # Expect grpo joined to have (tenant_id, task_id, group_id, sample_id, score_blend)
    cols = ["tenant_id", "task_id", "group_id", "sample_id", "score_blend"]
    for c in cols:
        if c not in grpo.columns:
            # If missing, create placeholder columns
            grpo = grpo.withColumn(c, F.lit(None).cast("string" if c != "score_blend" else "double"))

    linked = (
        link.join(grpo.select(cols), ["tenant_id", "task_id", "group_id", "sample_id"], "left")
        .withColumn("downstream_proxy", F.coalesce(F.col("score_blend"), F.col("diff_quality_proxy").cast("double")))
        .withColumn("ingest_ts", F.current_timestamp())
    )

    linked.write.mode("overwrite").parquet(out_path)


if __name__ == "__main__":
    main()

