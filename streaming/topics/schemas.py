from __future__ import annotations

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType, ArrayType, DoubleType, TimestampType, MapType
)


tool_step_schema = StructType([
    StructField("t_idx", IntegerType(), True),
    StructField("tool_name", StringType(), True),
    StructField("args_json", StringType(), True),
    StructField("result_kind", StringType(), True),
    StructField("tokens", ArrayType(LongType()), True),
    StructField("logprobs_token", ArrayType(DoubleType()), True),
    StructField("latency_ms", LongType(), True),
    StructField("error_code", StringType(), True),
    StructField("artifacts", ArrayType(StringType()), True),
])


tool_groups_schema = StructType([
    StructField("tenant_id", StringType(), False),
    StructField("task_id", StringType(), False),
    StructField("tool_group_id", StringType(), False),
    StructField("K", IntegerType(), True),
    StructField("step_budget", IntegerType(), True),
    StructField("sampler_cfg", StringType(), True),
    StructField("ts", TimestampType(), True),
])


tool_samples_schema = StructType([
    StructField("tenant_id", StringType(), False),
    StructField("task_id", StringType(), False),
    StructField("tool_group_id", StringType(), False),
    StructField("seq_id", StringType(), False),
    StructField("steps", ArrayType(tool_step_schema), True),
    StructField("aggregate_tokens", ArrayType(LongType()), True),
    StructField("ts", TimestampType(), True),
])


tool_scores_schema = StructType([
    StructField("tenant_id", StringType(), False),
    StructField("task_id", StringType(), False),
    StructField("tool_group_id", StringType(), False),
    StructField("seq_id", StringType(), False),
    StructField("source", StringType(), True),
    StructField("score", DoubleType(), True),
    StructField("ts", TimestampType(), True),
])


tool_to_diff_map_schema = StructType([
    StructField("tenant_id", StringType(), False),
    StructField("task_id", StringType(), False),
    StructField("tool_group_id", StringType(), False),
    StructField("seq_id", StringType(), False),
    StructField("group_id", StringType(), False),  # diff group id
    StructField("sample_id", StringType(), False),  # diff sample id
    StructField("diff_quality_proxy", DoubleType(), True),
    StructField("ts", TimestampType(), True),
])


def kafka_json_options(bootstrap: str) -> dict:
    return {
        "kafka.bootstrap.servers": bootstrap,
        "startingOffsets": "earliest",
        "maxOffsetsPerTrigger": "10000",
    }

