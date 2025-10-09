# Workflow Composer and Orchestrator Integration

This guide explains how workflow graphs created in the dashboard map to the Go orchestrator and the Rust environment runner.

## Directory Structure

```
${REPO_ROOT}/data/workflows/           # JSON definitions managed by the orchestrator
${REPO_ROOT}/data/workflows/<id>.json  # Latest revision of a workflow
${REPO_ROOT}/data/workflows/<id>/history/20240203T120102Z.json
                                      # Archived revision captured on each update
```

Run `scripts/prepare_workflow_store.sh` to bootstrap the store directory or
point `WORKFLOW_STORE_DIR` to a custom location when starting the orchestrator.
Use `scripts/show_workflow_history.sh <workflow-id>` to inspect archived
revisions via the HTTP API.

## Workflow Schema (abridged)

```json
{
  "id": "wf-go-build",
  "name": "Go Build Curriculum",
  "description": "Compile and verify Go agent builds",
  "tenant": "default",
  "version": "v1",
  "tags": ["go", "build"],
  "nodes": [
    {
      "id": "sig-code",
      "type": "signature",
      "name": "Codegen Signature",
      "signature": {
        "prompt": "// go build pipeline...",
        "runtime": "go",
        "tools": ["build", "fmt"],
        "temperature": 0.0
      }
    },
    {
      "id": "ver-test",
      "type": "verifier",
      "name": "Unit Tests",
      "verifier": {
        "command": "go test ./...",
        "weight": 1.0
      }
    }
  ],
  "edges": [
    { "id": "edge-1", "source": "sig-code", "target": "ver-test", "kind": "control" }
  ]
}
```

Validation rules enforced server-side:

- Non-empty name, at least one node and edge
- Unique node/edge IDs and valid references
- Type-specific requirements (e.g., signature prompts, verifier commands, deployment tenant/domain/channel)

## Metrics & Observability

The orchestrator now exposes a Prometheus gauge `workflows_total` via the
existing `/metrics` endpoint. Each save refreshes the gauge and archives the
previous revision to a timestamped JSON file under `history/`.

Front-end badges classify workflows as:

- **Active**: Updated within the last 15 minutes
- **Warm**: Updated in the last 3 hours
- **Stale**: Older than 3 hours

## Runner Context Propagation

When a queue submission includes `payload.workflow_id`, the orchestrator
embeds:

```json
"workflow_context": {
  "id": "wf-go-build",
  "name": "Go Build Curriculum",
  "tenant": "default",
  "version": "v1",
  "nodes": [...],
  "edges": [...],
  "node_index": {
    "signature": ["sig-code"],
    "verifier": ["ver-test"]
  }
}
```

The Rust runner re-emits this metadata with embeddings and uses it to enrich
error labels and reward routing hooks.

## Operational Checklist

1. Execute `scripts/prepare_workflow_store.sh` (or supply your own path) before running the orchestrator.
2. Set `WORKFLOW_STORE_DIR` in the orchestrator environment (defaults to `data/workflows`).
3. Configure queue submissions to include `workflow_id` when routing tasks through the runner.
4. Monitor `workflows_total` on the orchestrator metrics endpoint for visibility into expected workflows.
5. Inspect historical revisions under `data/workflows/<id>/history` or via
   `GET /workflows/<id>/history` (helper script provided in `scripts/`).
