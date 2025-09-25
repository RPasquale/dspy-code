from __future__ import annotations

import os
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from streaming.utils.spark import create_spark


def summarize(df: DataFrame, tag: str) -> DataFrame:
    # Basic profile: row count, columns, null fractions per column
    if df.rdd.isEmpty():
        return df.sparkSession.createDataFrame([], schema="source_file string, tag string, rows long")
    cols = df.columns
    agg_exprs = [F.count(F.lit(1)).alias("rows")]
    for c in cols:
        agg_exprs.append((F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) / F.count(F.lit(1))).alias(f"null_frac__{c}"))
    out = df.groupBy("source_file").agg(*agg_exprs).withColumn("tag", F.lit(tag))
    # Keep reasonable number of columns
    return out


def main() -> None:
    spark = create_spark("feature_build_ingest_files")
    base = os.getenv("WAREHOUSE_BASE", "./warehouse")
    out = os.path.join(base, "features", "ingest_files")

    sources = [
        (os.path.join(base, "silver", "files_normalized"), "csv"),
        (os.path.join(base, "silver", "excel_normalized"), "excel"),
        (os.path.join(base, "silver", "json_normalized"), "json"),
        (os.path.join(base, "silver", "parquet_normalized"), "parquet"),
        (os.path.join(base, "silver", "avro_normalized"), "avro"),
    ]

    outs = []
    for path, tag in sources:
        if not os.path.exists(path):
            continue
        df = spark.read.parquet(path)
        if "source_file" not in df.columns:
            df = df.withColumn("source_file", F.lit(tag))
        outs.append(summarize(df, tag))

    if not outs:
        print("No normalized sources found.")
        return
    allf = outs[0]
    for f in outs[1:]:
        allf = allf.unionByName(f, allowMissingColumns=True)

    (allf.write.mode("overwrite").parquet(out))


if __name__ == "__main__":
    main()

