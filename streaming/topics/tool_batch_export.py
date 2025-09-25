from __future__ import annotations

import json
import os
from pathlib import Path
from pyspark.sql import functions as F

from streaming.utils.spark import create_spark


def main() -> None:
    spark = create_spark("tool_batch_export")

    warehouse_base = os.getenv("WAREHOUSE_BASE", "./warehouse")
    features_base = os.getenv("FEATURES_BASE", os.path.join(warehouse_base, "features"))
    dataset_dir = Path(os.getenv("DATASET_DIR", os.path.join(warehouse_base, "datasets", "grpo_tool_batches")))
    shard_size = int(os.getenv("GRPO_SHARD_SIZE", "4096"))

    src = os.path.join(features_base, "grpo_tool_tokens")
    df = spark.read.parquet(src)

    # Create a shard id per N rows for export
    df = df.withColumn("row_id", F.monotonically_increasing_id())
    df = df.withColumn("shard_id", (F.col("row_id") / shard_size).cast("long"))

    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    shards = [r[0] for r in df.select("shard_id").distinct().collect()]
    for sid in shards:
        out_path = dataset_dir / f"shard_{sid:05d}.parquet"
        df.filter(F.col("shard_id") == sid).drop("row_id", "shard_id").write.mode("overwrite").parquet(str(out_path))
        cnt = spark.read.parquet(str(out_path)).count()
        manifests.append({"path": str(out_path), "rows": int(cnt), "schema_version": 1})

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()

