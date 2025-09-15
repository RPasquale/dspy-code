#!/usr/bin/env python3
import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr


def parse_args():
    p = argparse.ArgumentParser(description="Stream logs from Kafka and print to console")
    p.add_argument("--bootstrap", required=True, help="Kafka bootstrap servers")
    p.add_argument("--pattern", required=True, help="Topic subscribe pattern (regex)")
    p.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    return p.parse_args()


def main():
    args = parse_args()

    # Fallbacks for containerized envs
    os.environ.setdefault("HADOOP_USER_NAME", "spark")

    spark = (
        SparkSession.builder.appName("SparkLogs")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

    df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap)
        .option("subscribePattern", args.pattern)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    out = (
        df.select(
            col("topic"),
            col("partition"),
            col("offset"),
            expr("CAST(key AS STRING) AS key"),
            expr("CAST(value AS STRING) AS value"),
            col("timestamp"),
        )
    )

    query = (
        out.writeStream.outputMode("append")
        .format("console")
        .option("truncate", "false")
        .option("numRows", "20")
        .option("checkpointLocation", args.checkpoint)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
