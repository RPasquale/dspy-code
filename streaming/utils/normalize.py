from __future__ import annotations

import re
from typing import List
from pyspark.sql import DataFrame, functions as F


def snake_case(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", name.strip())
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()


def normalize_columns(df: DataFrame) -> DataFrame:
    mapping = {c: snake_case(c) for c in df.columns}
    for k, v in mapping.items():
        if k != v:
            df = df.withColumnRenamed(k, v)
    return df


def trim_strings(df: DataFrame) -> DataFrame:
    for c, t in df.dtypes:
        if t == 'string':
            df = df.withColumn(c, F.trim(F.col(c)))
    return df


def cast_numeric_like(df: DataFrame, cols: List[str] | None = None) -> DataFrame:
    """Cast numeric-like string columns to double by stripping commas.

    If cols is None, attempts to cast columns with names ending in _amount,_qty,_count,_num,_value.
    """
    candidates = cols or [c for c in df.columns if re.search(r"(_amount|_qty|_count|_num|_value)$", c)]
    for c in candidates:
        if c in df.columns:
            df = df.withColumn(c, F.regexp_replace(F.col(c), r",", ""))
            df = df.withColumn(c, F.col(c).cast('double'))
    return df


def parse_timestamps(df: DataFrame, patterns: List[str] | None = None) -> DataFrame:
    pats = patterns or [
        'yyyy-MM-dd HH:mm:ss', 'yyyy/MM/dd HH:mm:ss', 'yyyy-MM-dd', 'MM/dd/yyyy', 'dd/MM/yyyy'
    ]
    date_cols = [c for c in df.columns if re.search(r"(date|_dt|timestamp)$", c)]
    for c in date_cols:
        if c in df.columns:
            # try multiple patterns
            expr = None
            for p in pats:
                ts = F.to_timestamp(F.col(c), p)
                expr = ts if expr is None else F.coalesce(expr, ts)
            df = df.withColumn(c, expr)
    return df


def normalize_frame(df: DataFrame) -> DataFrame:
    df = normalize_columns(df)
    df = trim_strings(df)
    df = cast_numeric_like(df)
    df = parse_timestamps(df)
    return df

