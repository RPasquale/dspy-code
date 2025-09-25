from __future__ import annotations

import os
from pyspark.sql import SparkSession


def create_spark(app_name: str) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    )
    shuffle_parts = os.getenv("SPARK_SHUFFLE_PARTS")
    if shuffle_parts:
        builder = builder.config("spark.sql.shuffle.partitions", shuffle_parts)
    return builder.getOrCreate()

