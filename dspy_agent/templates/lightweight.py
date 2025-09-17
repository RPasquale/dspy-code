from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Iterable, List, Optional

try:
    from dspy_agent import __version__ as _DSPY_VERSION
except Exception:  # pragma: no cover
    _DSPY_VERSION = None


@dataclass(frozen=True)
class TemplateAsset:
    """Represents a file emitted by lightweight_init."""

    relative_path: str
    content: str
    executable: bool = False


def _default_pip_spec() -> str:
    if _DSPY_VERSION:
        return f"dspy-code=={_DSPY_VERSION}"
    try:
        from importlib import metadata

        version = metadata.version("dspy-code")
        if version:
            return f"dspy-code=={version}"
    except Exception:  # pragma: no cover - fallback below
        pass
    return "dspy-code"


def render_dockerfile(install_source: str = "pip", pip_spec: Optional[str] = None) -> str:
    """Render a Dockerfile for the lightweight stack."""

    if install_source not in {"pip", "local"}:
        raise ValueError(f"unsupported install_source '{install_source}'")

    base = dedent(
        """
        FROM python:3.11-slim

        ENV PYTHONDONTWRITEBYTECODE=1 \\
            PYTHONUNBUFFERED=1 \\
            PIP_NO_CACHE_DIR=1 \\
            PIP_DISABLE_PIP_VERSION_CHECK=1

        RUN apt-get update && apt-get install -y --no-install-recommends \\
            bash \\
            curl \\
            gcc \\
            git \\
            librdkafka-dev \\
            pkg-config \\
            procps \\
            tini && rm -rf /var/lib/apt/lists/*

        WORKDIR /app

        COPY entrypoints /entrypoints
        RUN chmod +x /entrypoints/*.sh
        """
    ).strip()

    if install_source == "pip":
        spec = pip_spec or _default_pip_spec()
        spec_literal = json.dumps(spec)
        pip_section = dedent(
            f"""
            ARG DSPY_PIP_SPEC={spec_literal}

            RUN pip install --no-cache-dir uv && \\
                uv pip install --system "${{DSPY_PIP_SPEC}}"
            """
        ).strip()
    else:
        pip_section = dedent(
            """
            COPY pyproject.toml README.md /app/
            COPY dspy_agent /app/dspy_agent

            RUN pip install --no-cache-dir uv && \\
                uv pip install --system .
            """
        ).strip()

    return f"{base}\n\n{pip_section}\n\nENTRYPOINT [\"dspy-agent\"]\n"


def render_compose(image: str, host_ws: Path, host_logs: Optional[Path], db_backend: str) -> str:
    ws_path = host_ws.resolve()
    logs_part = ""
    if host_logs:
        logs_part = f"\n      - {host_logs.resolve()}:/workspace/logs:ro"

    compose = dedent(
        f"""
        services:
          dspy-agent:
            image: {image}
            build:
              context: .
              dockerfile: Dockerfile
            environment:
              - LOCAL_MODE=false
              - USE_OLLAMA=true
              - DB_BACKEND={db_backend}
              - REDDB_URL
              - REDDB_NAMESPACE=dspy
              - REDDB_TOKEN
              - MODEL_NAME=qwen3:1.7b
              - OPENAI_API_KEY=
              - OPENAI_BASE_URL=http://ollama:11434
              - OLLAMA_MODEL=qwen3:1.7b
              - OLLAMA_API_KEY=
              - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
              - KAFKA_CLIENT_ID=dspy-agent
              - KAFKA_TOPIC_PREFIX
            entrypoint: ["/entrypoints/run_dspy_agent.sh"]
            volumes:
              - {ws_path}:/workspace:rw{logs_part}
              - ./entrypoints:/entrypoints:ro
            ports:
              - "127.0.0.1:8765:8765"
            restart: unless-stopped
            depends_on:
              ollama:
                condition: service_healthy
              kafka:
                condition: service_healthy
              spark:
                condition: service_healthy

          ollama:
            image: ollama/ollama:latest
            entrypoint: ["/bin/sh", "-lc"]
            command: |
              ollama serve &
              sleep 3;
              ollama pull qwen3:1.7b || true;
              wait
            ports:
              - "127.0.0.1:11435:11434"
            volumes:
              - ollama:/root/.ollama
            healthcheck:
              test: ["CMD-SHELL", "curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 || exit 1"]
              interval: 10s
              timeout: 3s
              retries: 20
              start_period: 5s

          zookeeper:
            image: bitnami/zookeeper:3.9
            environment:
              - ALLOW_ANONYMOUS_LOGIN=yes
            ports:
              - "127.0.0.1:2181:2181"
            healthcheck:
              test: ["CMD-SHELL", "echo > /dev/tcp/localhost/2181 || exit 1"]
              interval: 10s
              timeout: 3s
              retries: 20
              start_period: 5s

          kafka:
            image: bitnami/kafka:3.6
            depends_on:
              - zookeeper
            environment:
              - KAFKA_CFG_ZOOKEEPER_CONNECT=zookeeper:2181
              - ALLOW_PLAINTEXT_LISTENER=yes
              - KAFKA_LISTENERS=PLAINTEXT://:9092
              - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:9092
              - KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
              - KAFKA_INTER_BROKER_LISTENER_NAME=PLAINTEXT
            ports:
              - "127.0.0.1:9092:9092"
            healthcheck:
              test: ["CMD-SHELL", "echo > /dev/tcp/localhost/9092 || exit 1"]
              interval: 10s
              timeout: 3s
              retries: 20
              start_period: 10s

          spark:
            image: bitnami/spark:3.5
            depends_on:
              kafka:
                condition: service_healthy
            volumes:
              - {ws_path}:/workspace
              - ./scripts:/app/scripts:ro
              - ./entrypoints:/entrypoints:ro
            entrypoint: ["/entrypoints/run_spark.sh"]
            healthcheck:
              test: ["CMD-SHELL", "pgrep -f 'spark_logs.py' >/dev/null 2>&1 || exit 1"]
              interval: 30s
              timeout: 5s
              retries: 10
              start_period: 20s

          dspy-worker:
            image: {image}
            depends_on:
              kafka:
                condition: service_healthy
            entrypoint: ["/entrypoints/run_worker.sh", "app"]
            volumes:
              - {ws_path}:/workspace:rw
              - ./entrypoints:/entrypoints:ro
            healthcheck:
              test: ["CMD-SHELL", "pgrep -f 'dspy-agent worker --topic app' >/dev/null 2>&1 || exit 1"]
              interval: 15s
              timeout: 5s
              retries: 10
              start_period: 10s

          dspy-worker-backend:
            image: {image}
            depends_on:
              kafka:
                condition: service_healthy
            entrypoint: ["/entrypoints/run_worker.sh", "backend"]
            volumes:
              - {ws_path}:/workspace:rw
              - ./entrypoints:/entrypoints:ro
            healthcheck:
              test: ["CMD-SHELL", "pgrep -f 'dspy-agent worker --topic backend' >/dev/null 2>&1 || exit 1"]
              interval: 15s
              timeout: 5s
              retries: 10
              start_period: 10s

          dspy-worker-frontend:
            image: {image}
            depends_on:
              kafka:
                condition: service_healthy
            entrypoint: ["/entrypoints/run_worker.sh", "frontend"]
            volumes:
              - {ws_path}:/workspace:rw
              - ./entrypoints:/entrypoints:ro
            healthcheck:
              test: ["CMD-SHELL", "pgrep -f 'dspy-agent worker --topic frontend' >/dev/null 2>&1 || exit 1"]
              interval: 15s
              timeout: 5s
              retries: 10
              start_period: 10s

          dspy-code-watch:
            image: {image}
            depends_on:
              kafka:
                condition: service_healthy
            entrypoint: ["/bin/bash", "-lc"]
            command: >-
              python - <<'PY'
        from dspy_agent.code_tools.code_watch import CodeWatcher
        from pathlib import Path
        CodeWatcher(Path('/workspace')).run()
        PY
            volumes:
              - {ws_path}:/workspace:rw
            healthcheck:
              test: ["CMD-SHELL", "pgrep -f 'code_watch' >/dev/null 2>&1 || exit 1"]
              interval: 15s
              timeout: 5s
              retries: 10
              start_period: 10s

          dspy-code-indexer:
            image: {image}
            depends_on:
              kafka:
                condition: service_healthy
            entrypoint: ["/bin/bash", "-lc"]
            command: >-
              python - <<'PY'
        from dspy_agent.code_tools.code_indexer_worker import CodeIndexerWorker
        CodeIndexerWorker('kafka:9092').run()
        PY
            volumes:
              - {ws_path}:/workspace:rw
            healthcheck:
              test: ["CMD-SHELL", "pgrep -f 'code_indexer_worker' >/dev/null 2>&1 || exit 1"]
              interval: 15s
              timeout: 5s
              retries: 10
              start_period: 10s

          dspy-router:
            image: {image}
            depends_on:
              kafka:
                condition: service_healthy
            entrypoint: ["/entrypoints/run_router.sh"]
            volumes:
              - ./entrypoints:/entrypoints:ro
            healthcheck:
              test: ["CMD-SHELL", "pgrep -f 'router_worker' >/dev/null 2>&1 || exit 1"]
              interval: 15s
              timeout: 5s
              retries: 10
              start_period: 10s

        volumes:
          ollama: {{}}
        """
    ).strip()
    return compose + "\n"


def entrypoint_assets() -> List[TemplateAsset]:
    return [
        TemplateAsset(
            "entrypoints/run_dspy_agent.sh",
            dedent(
                """
                #!/usr/bin/env bash
                set -euo pipefail

                echo "[entrypoint] starting dspy-agent and waiting for deps"

                until curl -sf http://ollama:11434/api/tags >/dev/null 2>&1; do
                  echo "[entrypoint] waiting for ollama..."; sleep 2
                done

                until (echo > /dev/tcp/kafka/9092) >/dev/null 2>&1; do
                  echo "[entrypoint] waiting for kafka..."; sleep 2
                done

                dspy-agent stream-topics-create --bootstrap kafka:9092 || true

                exec dspy-agent up --workspace /workspace --db auto --status --status-port 8765
                """
            ).strip() + "\n",
            executable=True,
        ),
        TemplateAsset(
            "entrypoints/run_router.sh",
            dedent(
                """
                #!/usr/bin/env bash
                set -euo pipefail

                echo "[entrypoint] waiting for kafka, then starting router"

                until (echo > /dev/tcp/kafka/9092) >/dev/null 2>&1; do
                  echo "[entrypoint] waiting for kafka..."; sleep 2
                done

                python - <<'PY'
                from dspy_agent.agents.router_worker import RouterWorker
                RouterWorker('kafka:9092').run()
                PY
                """
            ).strip() + "\n",
            executable=True,
        ),
        TemplateAsset(
            "entrypoints/run_spark.sh",
            dedent(
                """
                #!/usr/bin/env bash
                set -euo pipefail

                echo "[entrypoint] starting spark log stream"

                spark-submit \
                  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
                  /app/scripts/streaming/spark_logs.py \
                  --bootstrap kafka:9092 \
                  --pattern 'logs.raw.*' \
                  --checkpoint /workspace/.dspy_checkpoints/spark_logs
                """
            ).strip() + "\n",
            executable=True,
        ),
        TemplateAsset(
            "entrypoints/run_worker.sh",
            dedent(
                """
                #!/usr/bin/env bash
                set -euo pipefail

                topic="${1:-app}"
                echo "[entrypoint] starting dspy-agent worker for topic: ${topic}"

                until (echo > /dev/tcp/kafka/9092) >/dev/null 2>&1; do
                  echo "[entrypoint] waiting for kafka..."; sleep 2
                done

                exec dspy-agent worker --topic "${topic}" --bootstrap kafka:9092
                """
            ).strip() + "\n",
            executable=True,
        ),
    ]


def extra_lightweight_assets() -> Iterable[TemplateAsset]:
    yield from entrypoint_assets()
    yield TemplateAsset(
        "scripts/streaming/spark_logs.py",
        dedent(
            """
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
            """
        ).strip() + "\n",
        executable=True,
    )
