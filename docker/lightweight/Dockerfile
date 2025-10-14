FROM rust:1.82 AS rust-builder

# Resource limits to prevent system crashes
ARG CARGO_BUILD_JOBS=2
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}
ENV CARGO_INCREMENTAL=0
ENV RUSTFLAGS="-C codegen-units=4"

WORKDIR /src

# Copy dependency files first for better caching
COPY env_runner_rs/Cargo.toml env_runner_rs/Cargo.lock ./env_runner_rs/
COPY reddb_rs/Cargo.toml ./reddb_rs/

# Fetch dependencies
RUN cd env_runner_rs && cargo fetch
RUN cd reddb_rs && cargo fetch

# Copy source code
COPY proto/ ./proto
COPY env_runner_rs/ ./env_runner_rs
COPY reddb_rs/ ./reddb_rs

# Build with limited parallelism to prevent system crashes
RUN cd env_runner_rs && cargo build --release -j ${CARGO_BUILD_JOBS}
RUN cd reddb_rs && cargo build --release -j ${CARGO_BUILD_JOBS}

FROM python:3.11-slim AS runtime

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

# Optionally build raylib from source for x86_64 only (pufferlib dependency)
RUN set -eux; \
    arch="$(uname -m)"; \
    if [ "$arch" = "x86_64" ]; then \
      git clone --depth 1 --branch 5.5 https://github.com/raysan5/raylib.git /tmp/raylib; \
      cd /tmp/raylib/src; \
      make PLATFORM=PLATFORM_DESKTOP BUILD_MODE=RELEASE -j2; \
      make install PLATFORM=PLATFORM_DESKTOP RAYLIB_LIB_PATH=/usr/local/lib RAYLIB_INCLUDE_PATH=/usr/local/include; \
      ldconfig; \
      rm -rf /tmp/raylib; \
      echo "[raylib] built for $arch"; \
    else \
      echo "[raylib] skipping build on $arch"; \
    fi

ENV RAYLIB_PATH=/usr/local \
    RAYLIB_LIB_PATH=/usr/local/lib \
    RAYLIB_INCLUDE_PATH=/usr/local/include \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# Skip ast-grep installation for now (optional dependency)
# ENV AST_GREP_VERSION=0.33.0
# RUN set -eux; \
#     arch="$(uname -m)"; \
#     if [ "$arch" = "x86_64" ]; then \
#       target="x86_64-unknown-linux-gnu"; \
#       url="https://github.com/ast-grep/ast-grep/releases/download/v${AST_GREP_VERSION}/ast-grep-v${AST_GREP_VERSION}-${target}.tar.gz"; \
#       curl -fsSL "$url" -o /tmp/ast-grep.tar.gz; \
#       tar -xzf /tmp/ast-grep.tar.gz -C /tmp; \
#       install /tmp/ast-grep/sg /usr/local/bin/sg; \
#       install /tmp/ast-grep/ast-grep /usr/local/bin/ast-grep; \
#       rm -rf /tmp/ast-grep*; \
#     else \
#       echo "[ast-grep] binary release not available for $arch; building via cargo"; \
#       apt-get update; \
#       apt-get install -y --no-install-recommends cargo rustc; \
#       cargo install --locked ast-grep; \
#       install /root/.cargo/bin/sg /usr/local/bin/sg; \
#       install /root/.cargo/bin/ast-grep /usr/local/bin/ast-grep; \
#       rm -rf /root/.cargo/registry /root/.cargo/git; \
#       apt-get purge -y cargo rustc; \
#       apt-get autoremove -y; \
#       rm -rf /var/lib/apt/lists/*; \
#     fi

WORKDIR /app

COPY docker/lightweight/entrypoints /entrypoints
RUN find /entrypoints -name "*.sh" -exec sed -i 's/\r$//' {} \; && chmod +x /entrypoints/*.sh

COPY pyproject.toml README.md /app/
COPY dspy_agent /app/dspy_agent
COPY scripts /app/scripts

RUN pip install --no-cache-dir uv && \
    uv pip install --system . && \
    pip install --no-cache-dir fastapi uvicorn

# Install RL dependencies on x86_64 only (optional)
RUN set -eux; \
    arch="$(uname -m)"; \
    if [ "$arch" = "x86_64" ]; then \
      echo "Installing RL extras (optional)..."; \
      export RAYLIB_PATH=/usr/local; \
      export RAYLIB_LIB_PATH=/usr/local/lib; \
      export RAYLIB_INCLUDE_PATH=/usr/local/include; \
      export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH; \
      if pip install --no-cache-dir --no-binary=pufferlib pufferlib>=3.0.0; then \
        echo "PufferLib installed successfully"; \
        uv pip install --system .[rl]; \
        echo "RL extras installed"; \
      else \
        echo "[warn] RL extras failed to install (optional). Continuing."; \
      fi; \
    else \
      echo "[warn] Skipping RL extras on $arch"; \
    fi

COPY orchestrator-linux /usr/local/bin/orchestrator
COPY stream_supervisor-linux /usr/local/bin/stream_supervisor
COPY --from=rust-builder /src/env_runner_rs/target/release/env_runner /usr/local/bin/env_runner
COPY --from=rust-builder /src/reddb_rs/target/release/reddb /usr/local/bin/reddb

RUN useradd -m -u 10001 -s /usr/sbin/nologin agent \
    && mkdir -p /home/agent/.dspy_cache \
    && chown -R agent:agent /app /entrypoints /home/agent

USER agent

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["dspy-agent"]
