from __future__ import annotations

import os
from pyspark.sql import functions as F
from pyspark.sql import Window

from streaming.utils.spark import create_spark


def main() -> None:
    spark = create_spark("tool_vectorize_tokens")

    warehouse_base = os.getenv("WAREHOUSE_BASE", "./warehouse")
    features_base = os.getenv("FEATURES_BASE", os.path.join(warehouse_base, "features"))
    max_len = int(os.getenv("MAX_LEN", "2048"))

    # Batch job over silver/tool_joined (append-only parquet)
    src = os.path.join(warehouse_base, "silver", "tool_joined")
    df = spark.read.parquet(src)

    # Compute group baselines and advantages per (tenant, task, tool_group)
    wgrp = Window.partitionBy("tenant_id", "task_id", "tool_group_id")
    df2 = df.withColumn("score", F.col("score_blend").cast("double"))
    df2 = df2.withColumn("baseline", F.avg("score").over(wgrp))
    df2 = df2.withColumn("A", F.coalesce(F.col("score") - F.col("baseline"), F.lit(0.0)))
    # Normalize within group and clip
    std = F.stddev_pop("A").over(wgrp)
    df2 = df2.withColumn("A_norm", F.when(std > 0, F.col("A") / std).otherwise(F.col("A")))
    clip = float(os.getenv("GRPO_CLIP", "5.0"))
    df2 = df2.withColumn("A_clip", F.when(F.col("A_norm") > clip, clip).when(F.col("A_norm") < -clip, -clip).otherwise(F.col("A_norm")))

    # Build a minimal token/features row per sequence (aggregate_tokens field)
    # For training, broadcast A_clip over tokens; here we keep a compact record per seq.
    out = df2.select(
        "tenant_id", "task_id", "tool_group_id", "seq_id", "aggregate_tokens", F.col("A_clip").alias("advantage"),
    ).withColumn("token_count", F.size(F.col("aggregate_tokens")))

    out_path = os.path.join(features_base, "grpo_tool_tokens")
    out.write.mode("overwrite").parquet(out_path)


if __name__ == "__main__":
    main()

