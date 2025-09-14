FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in separate layer for better caching
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    gcc \
    git \
    curl \
    librdkafka-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml README.md /app/

# Install uv and Python dependencies in separate layer
RUN pip install --no-cache-dir uv

# Copy source code
COPY dspy_agent /app/dspy_agent
COPY docker/lightweight/entrypoints /app/entrypoints

# Install Python package and dependencies
RUN chmod +x /app/entrypoints/*.sh && \
    uv pip install --system .

ENTRYPOINT ["dspy-agent"]
