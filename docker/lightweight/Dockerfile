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
    tini \
    wget \
    ca-certificates \
    xz-utils \
    tar && rm -rf /var/lib/apt/lists/*

# Install ast-grep. Prefer official binaries when available; fall back to cargo build (arm64).
ENV AST_GREP_VERSION=0.33.0
RUN set -eux; \
    arch="$(uname -m)"; \
    if [ "$arch" = "x86_64" ]; then \
      target="x86_64-unknown-linux-gnu"; \
      url="https://github.com/ast-grep/ast-grep/releases/download/v${AST_GREP_VERSION}/ast-grep-v${AST_GREP_VERSION}-${target}.tar.gz"; \
      curl -fsSL "$url" -o /tmp/ast-grep.tar.gz; \
      tar -xzf /tmp/ast-grep.tar.gz -C /tmp; \
      install /tmp/ast-grep/sg /usr/local/bin/sg; \
      install /tmp/ast-grep/ast-grep /usr/local/bin/ast-grep; \
      rm -rf /tmp/ast-grep*; \
    else \
      echo "[ast-grep] binary release not available for $arch; building via cargo"; \
      apt-get update; \
      apt-get install -y --no-install-recommends cargo rustc; \
      cargo install --locked ast-grep; \
      install /root/.cargo/bin/sg /usr/local/bin/sg; \
      install /root/.cargo/bin/ast-grep /usr/local/bin/ast-grep; \
      rm -rf /root/.cargo/registry /root/.cargo/git; \
      apt-get purge -y cargo rustc; \
      apt-get autoremove -y; \
      rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

COPY entrypoints /entrypoints
RUN chmod +x /entrypoints/*.sh

COPY pyproject.toml README.md /app/
COPY dspy_agent /app/dspy_agent

RUN pip install --no-cache-dir uv && \
    uv pip install --system .

ENTRYPOINT ["dspy-agent"]
