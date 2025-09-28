FROM rust:1.79 AS builder
WORKDIR /src
COPY reddb_rs/Cargo.toml ./reddb_rs/
RUN cd reddb_rs && cargo fetch
COPY reddb_rs/ ./reddb_rs
RUN cd reddb_rs && cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV REDDB_DATA_DIR=/data \
    REDDB_HOST=0.0.0.0 \
    REDDB_PORT=8080
RUN mkdir -p ${REDDB_DATA_DIR}
COPY --from=builder /src/reddb_rs/target/release/reddb /usr/local/bin/reddb
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:8080/health || exit 1
ENTRYPOINT ["/usr/local/bin/reddb"]
