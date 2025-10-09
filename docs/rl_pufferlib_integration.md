# RL + PufferLib Integration (Scaffold)

This page captures the agreed design for wiring a bandit/RL loop around the coding toolchain using PufferLib and the verifiers module, with Kafka/Spark optional context. It also includes editorial notes based on your outline to tighten clarity and make implementation predictable.

## Editorial Notes

- Prereqs: Pin a minimal working set (PufferLib v0.5+, Gymnasium v0.27, Torch 2.x). Install extras with `pip install .[rl]` which now pulls in Pyro (`pyro-ppl`) and CARBS so the Protein/Carbs sweep strategies run without manual wiring.
- Architecture note: PufferLib’s prebuilt raylib bundle presently targets `x86_64`. On Apple Silicon/ARM the `.[rl]` extras will install everything except PufferLib; either skip the Puffer vectorization paths or build/link PufferLib manually against an ARM raylib toolchain.
- Action space: Call out the initial discrete tools explicitly (run_tests, lint, build) and how to extend. Keep a table mapping action→tool args.
- Observations: Define the exact order of features: `[verifier_scores..., context_features...]`. Document each `verifier.kind` and its expected scale.
- Rewards: Specify weight semantics and shaping for penalties. For unbounded metrics (e.g., blast_radius), either min–max scale to [0,1] (with rolling window) or treat as a penalty whose magnitude is subtracted.
- Bandits: Clarify the learning target is tool selection only. For contextual bandits, list the context features and their normalization strategy.
- Vectorization: Note that vector env episodes can be 1 step (single action→reward) to align with bandits.
- Kafka/Spark: Include topic names and message schema (at least fields for `features: float[]` and `ts`). Start read-only; all writes remain in existing pipelines.
- Logging: Log per-step (tool, reward, verifier_details), per-epoch averages, and bandit value estimates.

## Directory Layout

- RL toolkit is consolidated under `dspy_agent/rl` (re-exported from `dspy_agent/rlkit.py`).

CLI commands:
- `dspy-agent rl train`: Hybrid GEPA + PufferLib training with the live toolchain executor (tests/lint/build). Tunable via `--steps`, `--n-envs`, `--lr`, `--entropy`, etc.
- `dspy-agent rl eval`: Evaluate with the same wiring (stub; extend for checkpoints).
- `dspy-agent rl tune`: Simple epsilon sweep for epsilon-greedy.
- `dspy-agent rl sweep`: Run a PufferLib-backed hyperparameter search and persist the best config per repo.
- `dspy-agent rl guide`: Print research-backed hyperparameter ranges (temperature, entropy, clip settings, curriculum stages).

Legacy flags such as `--puffer`, `--neural`, and separate `rl ppo`/`rl async-train`
entry points have been deprecated; the unified `rl train` command now runs the
PufferLib-based trainer by default.

### Supervisor Integration

- Set `RL_RESULTS_TOPIC` so the Rust env runner mirrors transitions onto a
  dedicated Kafka topic that the supervisor buffer consumes (`RL_BUFFER_DIR` by
  default).
- Start a training epoch via `POST /training/rl/start` on the supervisor; check
  progress at `/training/rl/status/<task>` or watch the new
  `supervisor_training_*` Prometheus gauges.
- Mesh workers mark training traffic with the `skill` header; the supervisor’s
  skill planner will auto-trigger focused RL jobs when failure rates breach the
  configured threshold.

### Running on Slurm

- Submit the GPU job with the new template: `sbatch deploy/slurm/train_puffer_rl.sbatch`
- Override knobs via environment variables (e.g. `RL_STEPS`, `RL_N_ENVS`, `RL_LR`,
  `RL_ENTROPY`, `RL_SKIP_GEPA`, `RL_GEPA_MODULES="context code"`).
- When `MESH_ENDPOINT` is provided the Slurm script will launch the Rust
  `env_runner` alongside the trainer so generations flow through the mesh stack.

## Toolchain Environment

The environment wraps the toolchain behind a discrete action space:
- Actions: `run_tests (0)`, `lint (1)`, `build (2)`
- Step: Execute tool → produce `AgentResult(metrics=...)` → compute reward via verifiers → return `(obs, reward, done, truncated, info)`.

Observation vector: `obs = [verifier_scores...] + context_features` where:
- `verifier_scores` are scaled/clamped per `RewardConfig`.
- `context_features` combine Kafka/Spark (optional) counters **and** knowledge-graph
  retrieval signals sourced from `.dspy_agentic` and RedDB (precision, coverage,
  average retrieval score, query volume). All features are normalized before
  reaching the policy.

Extend the action set by adding new enum values in `ToolAction` and updating your executor mapping.

Real executor wiring:
- The CLI uses `ToolchainExecutor` (see `dspy_agent/rl`) which maps:
  - `run_tests` → `pytest -q` by default (configurable via `RL_TEST_CMD`).
  - `lint` → `ruff check --output-format json .` (configurable via `RL_LINT_CMD`).
  - `build` → `python -m compileall -q .` (configurable via `RL_BUILD_CMD`).
- Metrics produced: `pass_rate`, `tests_passed`, `tests_failed`, `tests_total`, `lint_issues`, `lint_ok`, `build_ok`, and optional `blast_radius` if provided upstream.
- Customize defaults by passing `--workspace` and setting env vars.

## Verifiers and Reward

Each verifier follows:
```python
class BaseVerifier:
    kind: str
    def __call__(self, result: AgentResult) -> float: ...
```

Reward aggregation is a weighted sum with penalty support:
- `weights[kind]`: contribution multiplier
- `penalty_kinds`: kinds subtracted as `-abs(scaled_value)`
- `clamp01_kinds`: clamp to [0,1]
- `scales[kind] = (min, max)`: min–max scaling to [0,1]

See `dspy_agent/rl` for `RewardConfig` and `aggregate_reward`.

### Graph memory verifier weights

Graph/RAG signals ship as first-class verifiers (`graph_signal`, `graph_prefetch`,
`graph_mcts_alignment`, `memory_precision`, `memory_coverage`). Their defaults are
registered inside `dspy_agent.cli._rl_build_make_env`, so existing RL/stream runs
automatically score the new feedback. Adjust them inline or via the workspace
settings file, for example:

```
dspy-agent rl train --workspace . \
  --weights graph_signal=1.2 graph_mcts_alignment=1.3 memory_precision=0.6
```

Or persist into `.dspy/stream_settings.json` with the helper:

```
dspy-agent settings set reward_weights.graph_mcts_alignment 1.1
```

Higher weights emphasise alignment with graph memory and retrieval precision,
allowing dashboards and RL loops to prioritise memory health alongside code
quality.

Loading your verifiers:
- Provide a Python module via `--verifiers-module` or env var and expose:
  - `get_verifiers() -> Iterable[VerifierProtocol]`, or
  - module-level instances with `kind` + `__call__`, or
  - classes with `kind` and a no-arg constructor.
Loader: provided via `dspy_agent/rl.load_from_module`.

## Bandit Policy

Start with simple bandits over the discrete actions:
- `epsilon-greedy`: exploration via epsilon; tracks incremental means
- `ucb1`: optimism in face of uncertainty
- `thompson`: Beta sampling suitable for [0,1] rewards

Longer-term, wire a contextual bandit by passing normalized feature vectors (the current scaffold forwards `obs` to `policy.select(ctx)` and `policy.update(...)`).

## Vectorized Training

Two options:
- Built-in Python vectorization: `VectorRunner` manages a list of envs and loops over steps.
- PufferLib: `--puffer` uses `pufferlib.emulation.GymnasiumPufferEnv` + `pufferlib.vector.make` for high-throughput vectorization.

Examples:
```
pip install .[rl]  # On ARM, install pufferlib separately once raylib is available

# Initialize a config file (no env vars needed)
dspy-agent rl config init --out .dspy_rl.json --verifiers-module verifiers --puffer

# Hybrid GEPA + PufferLib training across 8 envs
RL_VERIFIERS_MODULE=verifiers dspy-agent rl train --steps 1000 --n-envs 8 --lr 5e-4 --entropy 0.02

# Lightweight single-env experiment (skips GEPA warmup)
dspy-agent rl train --workspace . --steps 200 --n-envs 1 --skip-gepa
```

### Hyperparameter Sweep Quickstart

Leverage the sweep wrapper to tune bandit/RL hyperparameters automatically:

```
dspy-agent rl sweep --workspace . --iterations 20 --update-config
```

The command evaluates candidate configurations with the live toolchain,
persists the best result to `.dspy/rl/best.json`, and (with
`--update-config`) refreshes `.dspy_rl.json` so future runs inherit the
optimised settings. When `method` is set to `protein` or `carbs`, ensure the
`.[rl]` extras are installed so Pyro (for Gaussian-process scoring) and CARBS
are available; the CLI will surface a clear runtime error if either dependency
is missing.

## Kafka/Spark Context

- Subscribe: `logs.ctx.<topic>` (e.g., `logs.ctx.backend`)
- Publish: `agent.result.<topic>` and `agent.feedback.<topic>` (not used by the read-only context in v1)
- The `KafkaContextSource` is an optional dependency; if unavailable it behaves as a no-op provider with empty features.
- Recommended schema (JSON): `{ "features": [float], "ts": int, "meta": {...} }`

## Future: Neural Policies (PufferLib)

The scaffold keeps imports lazy. When ready to train neural policies:
1. Install extras: `pip install .[rl]`
2. `train_puffer_policy` implements a lightweight torch REINFORCE loop and uses PufferLib vectorization when available.
3. The unified `dspy-agent rl train` command already drives the PufferLib trainer; adjust `--lr`, `--entropy`, and replay settings to explore different behaviours.

PuffeRL PPO shell:
See `dspy_agent/rl.run_puffer_ppo` for a sketch that builds a PufferLib vectorized env and instantiates PuffeRL with a simple actor-critic model. Replace the policy factory with a proper PufferLib policy and adjust config to your version.

## Configuration and Logging

- Keep RL settings (weights, scales, bandit params) in config (YAML/TOML/JSON) and/or env vars.
- Log step-level details: tool, reward, verifier raw and scaled scores, and bandit estimates.
- Periodically persist value estimates and best-tool summaries.

Prefer a JSON config file over env vars. Example `.dspy_rl.json`:
```
{
  "n_envs": 4,
  "verifiers_module": "verifiers",
  "weights": {"pass_rate": 1.0, "blast_radius": 1.0},
  "penalty_kinds": ["blast_radius"],
  "clamp01_kinds": ["pass_rate"],
  "scales": {"blast_radius": [0, 1]},
  "test_cmd": "pytest -q",
  "lint_cmd": "ruff check --output-format json .",
  "build_cmd": "python -m compileall -q .",
  "timeout_sec": 180
}
```

Training-specific hyperparameters (learning rate, entropy coefficient,
replay capacity, etc.) are now supplied via CLI flags on `dspy-agent rl train`.

Use with CLI:
```
dspy-agent rl train --workspace . --rl-config .dspy_rl.json --steps 1000 --n-envs 4
```

Default verifiers resolution order:
- If `RL_VERIFIERS_MODULE` is set, use it.
- Else, try importing the external `verifiers` package (installed via extras).
- Else, fallback to `dspy_agent.rl.sample_verifiers`.

Install the external verifiers package:
```
pip install 'verifiers @ git+https://github.com/willccbb/verifiers.git'
# or: pip install .[rl]  # includes verifiers in extras
```

## Integration Checklist

- Define/verifiers with stable `kind` names.
- Map tool actions to concrete invocations and args.
- Decide reward scaling for unbounded metrics (e.g., blast radius).
- Normalize context features; document units and ranges.
- Choose a bandit policy and initial hyperparameters.
- Add per-project weights and policy settings to your config.
