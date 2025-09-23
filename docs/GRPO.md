GRPO: Group Relative Preference Optimization (Torch)

Overview
- Purpose: learn a policy that increases preference-aligned outputs from the coding agent using group-wise relative rewards.
- Components:
  - Trainer: Torch-based implementation with group-wise standardized advantages and KL to a reference model.
  - Miner: Builds grouped preference datasets from RedDB actions and network signals.
  - Service: Background training with live metrics exposed in the dashboard.

Data format (JSONL)
- One JSON object per group:
  - prompt: string
  - candidates: array of { text: string, reward: number }
  - meta: optional dict (context_hash, signature_name, env, hits, …)

Example
  {"prompt": "Refactor function for clarity", "candidates": [{"text": "Solution A...", "reward": 0.2}, {"text": "Solution B...", "reward": 0.8}], "meta": {"context_hash": "..."}}

Mining from RedDB
- Command: `dspy-agent grpo mine --hours 48 --out docs/samples/grpo_mined.jsonl`
- Behavior:
  - Groups recent actions by inferred user prompt (from parameters/result/state).
  - Uses rewards logged in ActionRecord to weight candidates.
  - Enriches meta with InferMesh retrieval hits when available, plus context_hash/signature_name.
  - Works across DB nodes; reads from RedDB streams (`rl_actions`, `retrieval_events`).

Training
- One-off CLI: `dspy-agent grpo train --dataset docs/samples/grpo_example.jsonl --steps 1000 --model gpt2`
- Dashboard UI: on Training page, “GRPO Training” panel supports start/stop and displays metrics.
- Checkpoints and metrics: `.grpo/checkpoints`, `.grpo/metrics.jsonl`.
- Schedules (optional): `--lr-step N --lr-gamma 0.9` for StepLR; `--kl-warmup N --kl-target 0.03` to anneal KL from 0→target over N steps.

Policy Backends
- HF Transformers: when installed, uses `AutoModelForCausalLM` to compute log-probs for (prompt+response).
- Fallback scorer: a light BoW linear model to enable GRPO without heavy dependencies.

Mesh / InferMesh / Network Integration
- InferMesh: recent retrieval events populate `meta.imesh_hits` per group to surface context.
- Mesh/DB nodes: mining traverses RedDB streams regardless of deployment topology; if multiple nodes feed `rl_actions`, groups include cross-node behavior.
- Future hooks (suggested):
  - Use embedding neighborhoods to synthesize hard negatives per prompt (k-NN across InferMesh Parquet shards).
  - Trigger mining jobs from Kafka (e.g., compact per-query windows) and write datasets to an artifacts topic.

Operational Notes
- Torch required for training; Transformers optional.
- Large HF models should be paired with GPU (set `--device cuda`).
- For production, run via the dashboard or containerize the GRPO service alongside RedDB and the embed worker.

Auto Mode and Hierarchical GRPO
- Auto: Start background mining+training cycles via dashboard (Start Auto) or backend API `/api/grpo/auto/start`. Configure cadence (`period_sec`), window (`hours`), and min group threshold.
- Hierarchical: Auto mode can run in `hierarchical` mode to mine multiple datasets per cycle:
  - global.jsonl (all actions), signature.jsonl (per-signature behaviors), patch.jsonl (patch/edit actions).
  - The service trains sequentially per level and persists level-specific checkpoints in `.grpo/auto/ckpts_<level>/`.
  - This approximates hierarchical credit assignment: the global controller learns high-level routing preferences; signature-level learns finer behaviors; patch-level focuses on code edits.

Policy Nudges (apply-policy)
- CLI: `dspy-agent grpo apply-policy --hours 24 --top-k 3 --bottom-k 1 --workspace <repo-root>`
- Auto mode: include `{ "apply_policy": true, "workspace": "/app/project" }` in `/api/grpo/auto/start` payload.
- Derives safe nudges from recent actions: prefers top-performing tools/signatures and (optionally) denies very poor performers with simple regex triggers built from frequent prompt keywords. Writes `.dspy_policy.yaml`.

Capacity Guard and Live Stats
- Live system stats: the dashboard training card now shows CPU, RAM, Disk, GPU, and top container usage (refreshes every 5s).
- Guard API: backend enforces a resource guard before GRPO auto cycles and when starting training.
  - Endpoint: `POST /api/system/guard` with `{ "min_free_gb": 5, "min_ram_gb": 2, "min_vram_mb": 0 }`.
  - UI: “Guard: OK/Insufficient capacity” appears on the GRPO card and disables Start buttons if failing.
- Auto thresholds: include `{ "min_free_gb": 5, "min_ram_gb": 2, "min_vram_mb": 0 }` in `/api/grpo/auto/start` to gate cycles.
