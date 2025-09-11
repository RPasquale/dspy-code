FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends     gcc git curl librdkafka-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY dspy_agent /app/dspy_agent

RUN pip install --no-cache-dir uv &&     uv pip install --system . &&     pip install --no-cache-dir confluent-kafka || true

ENTRYPOINT ["dspy-agent"]