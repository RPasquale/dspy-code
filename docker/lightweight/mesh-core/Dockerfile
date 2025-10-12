FROM rust:1.82 as builder

# Resource limits to prevent system crashes
ARG CARGO_BUILD_JOBS=2
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}
ENV CARGO_INCREMENTAL=0
ENV RUSTFLAGS="-C codegen-units=4"

ARG MESH_VERSION=main
WORKDIR /src

# Install system dependencies including protobuf compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    git \
    protobuf-compiler \
    protobuf-compiler-grpc \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install protobuf well-known types
RUN mkdir -p /usr/include/google/protobuf && \
    curl -L https://raw.githubusercontent.com/protocolbuffers/protobuf/main/src/google/protobuf/empty.proto \
    -o /usr/include/google/protobuf/empty.proto

# Clone the repository
RUN git clone --depth 1 --branch ${MESH_VERSION} https://github.com/redbco/mesh-core.git .

# Copy local proto files
COPY proto/ ./proto/

# Clean any existing Cargo.lock to avoid version conflicts
RUN rm -f Cargo.lock

# Update Rust to latest stable and install protobuf codegen
RUN rustup update stable && rustup default stable
RUN cargo install protobuf-codegen -j ${CARGO_BUILD_JOBS}

# Set protobuf include path to include Google well-known types
ENV PROTOBUF_INCLUDE_PATH=/usr/include

# Build the project with limited parallelism to prevent system crashes
RUN cargo build --release --bin mesh -j ${CARGO_BUILD_JOBS}

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates netcat-openbsd bash && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/mesh
COPY --from=builder /src/target/release/mesh /opt/mesh/bin/mesh
COPY docker/lightweight/entrypoints/run_mesh.sh /entrypoints/run_mesh.sh
RUN sed -i 's/\r$//' /entrypoints/run_mesh.sh && chmod +x /entrypoints/run_mesh.sh && chmod +x /opt/mesh/bin/mesh
ENTRYPOINT ["/entrypoints/run_mesh.sh"]
