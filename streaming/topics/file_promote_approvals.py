from __future__ import annotations

import os
from pyspark.sql import functions as F
from streaming.utils.spark import create_spark
from streaming.utils.normalize import normalize_frame


def main() -> None:
    spark = create_spark("file_promote_approvals")
    base = os.getenv("WAREHOUSE_BASE", "./warehouse")
    approvals_dir = os.getenv("APPROVALS_DIR", os.path.join(base, "approvals", "files_decisions"))
    bronze_csv = os.path.join(base, "bronze", "files_raw")
    bronze_xlsx = os.path.join(base, "bronze", "excel_raw")
    out_csv = os.path.join(base, "silver", "files_normalized")
    out_xlsx = os.path.join(base, "silver", "excel_normalized")

    # Read approvals decisions JSONL (schema: source_file, decision, ts)
    if not os.path.exists(approvals_dir):
        print("No approvals directory:", approvals_dir)
        return
    approvals = spark.read.json(approvals_dir)
    approved = approvals.filter(F.lower(F.col("decision")) == F.lit("approved")).select("source_file").distinct()

    if approved.rdd.isEmpty():
        print("No approved files to promote.")
        return

    # Promote CSV bronze → silver
    if os.path.exists(bronze_csv):
        csv_src = spark.read.parquet(bronze_csv)
        csv_promote = csv_src.join(approved, ["source_file"], "inner")
        if not csv_promote.rdd.isEmpty():
            normalized = normalize_frame(csv_promote).withColumn("promote_ts", F.current_timestamp())
            normalized.write.mode("append").parquet(out_csv)

    # Promote Excel bronze → silver
    if os.path.exists(bronze_xlsx):
        xlsx_src = spark.read.parquet(bronze_xlsx)
        xlsx_promote = xlsx_src.join(approved, ["source_file"], "inner")
        if not xlsx_promote.rdd.isEmpty():
            normalized = normalize_frame(xlsx_promote).withColumn("promote_ts", F.current_timestamp())
            normalized.write.mode("append").parquet(out_xlsx)


if __name__ == "__main__":
    main()

