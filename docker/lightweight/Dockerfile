FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    gcc \
    git \
    librdkafka-dev \
    pkg-config \
    procps \
    tini && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY entrypoints /entrypoints
RUN chmod +x /entrypoints/*.sh

COPY pyproject.toml README.md /app/
COPY dspy_agent /app/dspy_agent

RUN pip install --no-cache-dir uv && \
    uv pip install --system .

ENTRYPOINT ["dspy-agent"]
