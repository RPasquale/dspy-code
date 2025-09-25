from __future__ import annotations

import os
from pyspark.sql import functions as F
from pyspark.sql import types as T

from streaming.utils.spark import create_spark
from .schemas import (
    tool_groups_schema, tool_samples_schema, tool_scores_schema, kafka_json_options,
)


def main() -> None:
    spark = create_spark("tool_join_groups")

    bootstrap = os.getenv("KAFKA_BOOTSTRAP")
    topic_groups = os.getenv("TOPIC_TOOL_GROUPS", "tool_groups")
    topic_samples = os.getenv("TOPIC_TOOL_SAMPLES", "tool_samples")
    topic_scores = os.getenv("TOPIC_TOOL_SCORES", "tool_scores")
    checkpoint_base = os.getenv("CHECKPOINT_BASE", "./checkpoints")
    warehouse_base = os.getenv("WAREHOUSE_BASE", "./warehouse")

    if not bootstrap:
        raise SystemExit("KAFKA_BOOTSTRAP not set")

    opts = kafka_json_options(bootstrap)

    # Read Kafka JSON streams
    groups_raw = (
        spark.readStream.format("kafka").options(**opts).option("subscribe", topic_groups).load()
    )
    samples_raw = (
        spark.readStream.format("kafka").options(**opts).option("subscribe", topic_samples).load()
    )
    scores_raw = (
        spark.readStream.format("kafka").options(**opts).option("subscribe", topic_scores).load()
    )

    def parse(df, schema):
        js = F.from_json(F.col("value").cast("string"), schema)
        return df.select(js.alias("j")).select("j.*")

    groups = parse(groups_raw, tool_groups_schema)
    samples = parse(samples_raw, tool_samples_schema)
    scores = parse(scores_raw, tool_scores_schema)

    # Watermark + dedupe
    groups_w = groups.withWatermark("ts", "30 minutes").dropDuplicates(["tenant_id", "task_id", "tool_group_id"])  # noqa: E501
    samples_w = samples.withWatermark("ts", "30 minutes").dropDuplicates(["tenant_id", "task_id", "tool_group_id", "seq_id"])  # noqa: E501
    scores_w = scores.withWatermark("ts", "30 minutes").dropDuplicates(["tenant_id", "task_id", "tool_group_id", "seq_id", "source"])  # noqa: E501

    # Blend scores per sequence (mean over sources)
    score_blend = (
        scores_w.groupBy("tenant_id", "task_id", "tool_group_id", "seq_id")
        .agg(F.avg("score").alias("score_blend"), F.max("ts").alias("ts_last"))
    )

    joined = (
        samples_w.join(groups_w, ["tenant_id", "task_id", "tool_group_id"], "left")
        .join(score_blend, ["tenant_id", "task_id", "tool_group_id", "seq_id"], "left")
        .withColumn("ingest_ts", F.current_timestamp())
    )

    out_path = os.path.join(warehouse_base, "silver", "tool_joined")
    ckpt_path = os.path.join(checkpoint_base, "tool_join_groups")

    (
        joined.writeStream.outputMode("append")
        .format("parquet")
        .option("checkpointLocation", ckpt_path)
        .option("path", out_path)
        .trigger(processingTime="30 seconds")
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    main()

