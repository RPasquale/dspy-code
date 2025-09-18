FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    gcc \
    g++ \
    build-essential \
    cmake \
    git \
    librdkafka-dev \
    pkg-config \
    procps \
    tini \
    wget \
    ca-certificates \
    xz-utils \
    tar \
    ninja-build \
    libasound2-dev \
    libx11-dev \
    libxrandr-dev \
    libxi-dev \
    libxcursor-dev \
    libxinerama-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    mesa-common-dev \
    libopenal-dev \
    libvorbis-dev \
    libogg-dev && rm -rf /var/lib/apt/lists/*

# Build raylib from source for the current architecture (pufferlib dep)
RUN git clone --depth 1 --branch 5.5 https://github.com/raysan5/raylib.git /tmp/raylib \
 && cd /tmp/raylib/src \
 && make PLATFORM=PLATFORM_DESKTOP BUILD_MODE=RELEASE -j"$(nproc)" \
 && make install PLATFORM=PLATFORM_DESKTOP RAYLIB_LIB_PATH=/usr/local/lib RAYLIB_INCLUDE_PATH=/usr/local/include \
 && rm -rf /tmp/raylib

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

# Install uv and base dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system .

# Install RL dependencies conditionally based on architecture
RUN set -eux; \
    arch="$(uname -m)"; \
    if [ "$arch" = "x86_64" ]; then \
      echo "Installing full RL stack for x86_64..."; \
      uv pip install --system .[rl]; \
    else \
      echo "Installing ARM64-compatible RL dependencies..."; \
      uv pip install --system .[rl-arm64]; \
      echo "Note: PufferLib not available on ARM64, using alternative RL components"; \
    fi

ENTRYPOINT ["dspy-agent"]
