# Proto Tooling Guide

This repository uses a shared `proto/` workspace with [Buf](https://buf.build) to produce Go
bindings and descriptor sets, while the Rust environment runner compiles the same `.proto`
contracts at build time via `prost`/`tonic`.

## Layout

- `proto/` – canonical definitions (`mesh.proto`, `runner.proto`).
- `proto/buf.yaml` – Buf module configuration (imports Google well-known types).
- `buf.gen.yaml` – generation template for all targets.
- `scripts/generate_protos.sh` – helper that runs Buf and formats regenerated Go bindings.
- `env_runner_rs/build.rs` – compiles the same protos for Rust at build time, watching `proto/`.

## Regeneration Workflow

1. Install Buf locally (`brew install bufbuild/buf/buf` or see Buf docs).
2. Run `make proto.generate` from the repo root. This will:
   - Regenerate Go bindings into `orchestrator/internal/pb`.
   - Emit an updated descriptor set in `proto/bin/descriptor.bin`.
   - Reformat the generated Go sources.
3. Build the Rust environment runner (`cargo build -p env_runner_rs`) to refresh the
   prost/tonic bindings in `target/`.

When editing proto files:

- Keep changes isolated to `proto/` and re-run `make proto.generate`.
- The Rust bindings are produced on the fly during compilation via `env_runner_rs/build.rs`. If
  you need the generated Rust code materialised on disk (e.g. for IDE support), run
  `PROTO_ROOT=proto cargo build -p env_runner_rs` and inspect `target/`.

## Troubleshooting

- If Buf is unavailable, `scripts/generate_protos.sh` will exit with guidance rather than silently
  failing.
- The Rust build script honours the `PROTO_ROOT` environment variable, enabling alternate proto
  directories for experiments (e.g. `PROTO_ROOT=/tmp/proto cargo test -p env_runner_rs`).
