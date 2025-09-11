FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

# Install required system dependencies.
# - bash is needed because docker-compose uses "/bin/bash -lc" for service commands
# - curl and librdkafka-dev already present for runtime needs
# - gcc and git for building/installing the Python package
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash gcc git curl librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY dspy_agent /app/dspy_agent
COPY docker/lightweight/entrypoints /app/entrypoints

RUN chmod +x /app/entrypoints/*.sh && \
    pip install --no-cache-dir uv && \
    uv pip install --system . && \
    pip install --no-cache-dir confluent-kafka || true

ENTRYPOINT ["dspy-agent"]
