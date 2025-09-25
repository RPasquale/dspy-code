from __future__ import annotations

import os
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from streaming.utils.spark import create_spark


def main() -> None:
    spark = create_spark("tool_metrics")

    warehouse_base = os.getenv("WAREHOUSE_BASE", "./warehouse")
    metrics_base = os.getenv("METRICS_BASE", os.path.join(warehouse_base, "metrics"))
    src = os.path.join(warehouse_base, "silver", "tool_joined")

    df = spark.read.parquet(src)
    # Simple time-bucket metrics (5-minute windows) by tenant
    metrics = (
        df.groupBy(
            "tenant_id",
            F.window(F.col("ts"), "5 minutes").alias("w"),
        )
        .agg(
            F.countDistinct("tool_group_id").alias("groups"),
            F.countDistinct("seq_id").alias("sequences"),
            F.avg("score_blend").alias("score_avg"),
        )
        .select(
            "tenant_id",
            F.col("w.start").alias("ts_start"),
            F.col("w.end").alias("ts_end"),
            "groups",
            "sequences",
            "score_avg",
        )
    )

    out_path = os.path.join(metrics_base, "grpo_tool_5m")
    metrics.write.mode("overwrite").parquet(out_path)


if __name__ == "__main__":
    main()

