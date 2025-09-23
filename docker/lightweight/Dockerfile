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
    libogg-dev \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    liblzma-dev \
    libedit-dev \
    llvm \
    libomp-dev && rm -rf /var/lib/apt/lists/*

# Build raylib from source for the current architecture (pufferlib dep)
# This ensures compatibility with PufferLib's requirements
RUN git clone --depth 1 --branch 5.5 https://github.com/raysan5/raylib.git /tmp/raylib \
 && cd /tmp/raylib/src \
 && make PLATFORM=PLATFORM_DESKTOP BUILD_MODE=RELEASE -j"$(nproc)" \
 && make install PLATFORM=PLATFORM_DESKTOP RAYLIB_LIB_PATH=/usr/local/lib RAYLIB_INCLUDE_PATH=/usr/local/include \
 && ldconfig \
 && rm -rf /tmp/raylib

# Set environment variables to help PufferLib find the correct raylib
ENV RAYLIB_PATH=/usr/local \
    RAYLIB_LIB_PATH=/usr/local/lib \
    RAYLIB_INCLUDE_PATH=/usr/local/include \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

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

# Install RL dependencies if available (optional)
RUN set -eux; \
    echo "Installing RL extras (optional)..."; \
    # Set environment variables to help with PufferLib build \
    export RAYLIB_PATH=/usr/local; \
    export RAYLIB_LIB_PATH=/usr/local/lib; \
    export RAYLIB_INCLUDE_PATH=/usr/local/include; \
    export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH; \
    # Install PufferLib from source to avoid pre-built raylib issues \
    if pip install --no-cache-dir --no-binary=pufferlib pufferlib>=3.0.0; then \
      echo "PufferLib installed successfully from source"; \
      # Install other RL dependencies \
      uv pip install --system .[rl]; \
      echo "RL extras installed"; \
    else \
      echo "[warn] RL extras failed to install (optional). Continuing."; \
    fi

# Create non-root user and set permissions
RUN useradd -m -u 10001 -s /usr/sbin/nologin agent \
    && mkdir -p /home/agent/.dspy_cache \
    && chown -R agent:agent /app /entrypoints /home/agent

USER agent

ENTRYPOINT ["dspy-agent"]
