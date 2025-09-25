from __future__ import annotations

import os
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from streaming.utils.spark import create_spark
from streaming.utils.normalize import normalize_frame


def excel_available(spark: SparkSession) -> bool:
    try:
        # Check if spark-excel datasource is available
        spark._jvm.org.apache.spark.sql.execution.datasources.DataSource.lookupDataSource("com.crealytics.spark.excel", spark._jsparkSession.sessionState().conf())  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


def pandas_excel_to_spark(spark: SparkSession, path: str):
    import pandas as pd  # type: ignore
    xls = pd.ExcelFile(path)
    frames = []
    for sheet in xls.sheet_names:
        pdf = xls.parse(sheet_name=sheet)
        pdf['__sheet__'] = sheet
        pdf['source_file'] = path
        frames.append(pdf)
    if not frames:
        return spark.createDataFrame([], schema=None)
    big = pd.concat(frames, ignore_index=True)
    return spark.createDataFrame(big)


def main() -> None:
    spark = create_spark("file_ingest_excel")
    src_dir = os.getenv("INGEST_XLSX_SRC", "/data/landing/excel")
    out_base = os.getenv("WAREHOUSE_BASE", "./warehouse")
    out_bronze = os.path.join(out_base, "bronze", "excel_raw")
    out_silver = os.path.join(out_base, "silver", "excel_normalized")
    header = os.getenv("XLSX_HEADER", "true")
    infer = os.getenv("XLSX_INFER", "true")
    data_addr = os.getenv("XLSX_DATA_ADDRESS", None)

    # Batch ingestion (run periodically or via SparkOperator)
    files = [str(p) for p in Path(src_dir).rglob("*.xlsx")]
    if not files:
        print("No Excel files found in", src_dir)
        return

    use_excel_ds = excel_available(spark)
    dfs = []
    for f in files:
        if use_excel_ds:
            reader = spark.read.format("com.crealytics.spark.excel").option("header", header).option("inferSchema", infer)
            if data_addr:
                reader = reader.option("dataAddress", data_addr)
            try:
                df = reader.load(f).withColumn("source_file", F.lit(f))
            except Exception:
                # fallback to pandas
                df = pandas_excel_to_spark(spark, f)
        else:
            df = pandas_excel_to_spark(spark, f)
        dfs.append(df)

    if not dfs:
        return

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.unionByName(d, allowMissingColumns=True)

    # bronze
    merged.write.mode("append").parquet(out_bronze)
    # silver
    normalized = normalize_frame(merged).withColumn("ingest_ts", F.current_timestamp())
    normalized.write.mode("append").parquet(out_silver)


if __name__ == "__main__":
    main()

