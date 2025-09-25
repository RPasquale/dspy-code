from __future__ import annotations

import json
import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T


def create_spark(app_name: str) -> SparkSession:
    return (SparkSession.builder.appName(app_name).getOrCreate())


def main() -> None:
    spark = create_spark("code_log_build_dataset")

    events_dir = os.getenv("EVENTS_DIR") or os.getenv("EVENTBUS_LOG_DIR") or "/warehouse/events"
    code_root = Path(os.getenv("CODE_ROOT", "/opt/app"))
    dataset_dir = Path(os.getenv("DATASET_DIR", "/warehouse/datasets/code_log_predict"))
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Read all topic jsonl files
    # Each line is a JSON object with keys: topic, ts/timestamp, event (dict), caller (file,function,line) inside meta.
    schema = T.StringType()
    df = spark.read.text(events_dir)
    # Parse JSON lines
    def parse_json(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None
    js_udf = F.udf(parse_json, T.MapType(T.StringType(), T.StringType(), True))  # loose
    # Better: parse with from_json on a permissive schema, but we accept Map for portability.
    j = df.select(F.col("value").alias("line"))

    # Use a python udf to extract needed fields robustly
    def extract(line: str):
        try:
            obj = json.loads(line)
            topic = obj.get("topic")
            ev = obj.get("event") or {}
            ts = obj.get("ts") or obj.get("timestamp")
            caller = (obj.get("caller") or {}) if isinstance(obj, dict) else {}
            if not caller and isinstance(obj, dict):
                # Some events may nest meta differently; try within obj
                caller = (obj.get("meta") or {}).get("caller") if isinstance(obj.get("meta"), dict) else {}
            cfile = caller.get("file") if isinstance(caller, dict) else None
            func = caller.get("function") if isinstance(caller, dict) else None
            line_no = caller.get("line") if isinstance(caller, dict) else None
            # Build a compact log text
            log_text = None
            if isinstance(ev, dict):
                for k in ("message", "text", "stdout", "status", "action", "name", "event"):
                    v = ev.get(k)
                    if isinstance(v, str) and v.strip():
                        log_text = v; break
                if not log_text:
                    log_text = json.dumps({k:v for k,v in ev.items() if isinstance(v, (str,int,float))})
            else:
                log_text = str(ev)
            return {
                "topic": topic,
                "ts": float(ts) if ts is not None else None,
                "code_path": cfile,
                "function": func,
                "line": int(line_no) if line_no is not None else None,
                "log_text": log_text,
            }
        except Exception:
            return None
    extract_udf = F.udf(extract, T.MapType(T.StringType(), T.StringType()))
    rows = j.select(extract_udf(F.col("line")).alias("rec")).where(F.col("rec").isNotNull())

    # Flatten map to columns
    cols = [
        F.col("rec")["topic"].alias("topic"),
        F.col("rec")["ts"].alias("ts"),
        F.col("rec")["code_path"].alias("code_path"),
        F.col("rec")["function"].alias("function"),
        F.col("rec")["line"].alias("line"),
        F.col("rec")["log_text"].alias("log_text"),
    ]
    base = rows.select(*cols).where(F.col("code_path").isNotNull() & F.col("log_text").isNotNull())

    # UDF to read code content; truncate to max_len
    max_len = int(os.getenv("CODE_MAX_CHARS", "8000"))
    snippet_mode = os.getenv("CODE_SNIPPET_MODE", "local").lower()  # local|function|file
    snippet_window = int(os.getenv("CODE_SNIPPET_WINDOW", "80"))

    def read_code(path: str, line: int | None = None):
        try:
            p = (code_root / path).resolve()
            if p.exists() and p.is_file():
                txt = p.read_text(encoding='utf-8', errors='ignore')
                if snippet_mode == 'file' or line is None:
                    return txt[:max_len]
                # local window around line
                lines = txt.splitlines()
                ln = max(1, int(line))
                if snippet_mode == 'local':
                    start = max(0, ln - 1 - snippet_window//2)
                    end = min(len(lines), ln - 1 + snippet_window//2)
                    snippet = "\n".join(lines[start:end])
                    return snippet[:max_len]
                if snippet_mode == 'function' and p.suffix == '.py':
                    try:
                        import ast
                        tree = ast.parse(txt)
                        func_start = 1
                        func_end = len(lines)
                        class Finder(ast.NodeVisitor):
                            def __init__(self):
                                self.s = 1; self.e = len(lines)
                            def visit_FunctionDef(self, node):  # type: ignore
                                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                                    if node.lineno <= ln <= node.end_lineno:
                                        self.s = node.lineno; self.e = node.end_lineno
                                self.generic_visit(node)
                        f = Finder(); f.visit(tree)
                        func_start, func_end = f.s, f.e
                        snippet = "\n".join(lines[func_start-1:func_end])
                        return snippet[:max_len]
                    except Exception:
                        # fallback to local window
                        start = max(0, ln - 1 - snippet_window//2)
                        end = min(len(lines), ln - 1 + snippet_window//2)
                        snippet = "\n".join(lines[start:end])
                        return snippet[:max_len]
                # default fallback
                return txt[:max_len]
        except Exception:
            pass
        return None
    def read_code_wrapper(path: str, ln: str | None = None):
        try:
            line_num = int(ln) if ln is not None else None
        except Exception:
            line_num = None
        return read_code(path, line_num)

    read_code_udf = F.udf(read_code_wrapper, T.StringType())
    out = base.withColumn("code_text", read_code_udf(F.col("code_path"), F.col("line"))).where(F.col("code_text").isNotNull())

    # Write shard with partition by day
    out = out.withColumn("day", F.from_unixtime(F.col("ts"), 'yyyy-MM-dd'))
    (out.select("code_path","function","line","code_text","log_text","topic","ts","day")
        .write.mode("append").partitionBy("day").parquet(str(dataset_dir)))


if __name__ == '__main__':
    main()
