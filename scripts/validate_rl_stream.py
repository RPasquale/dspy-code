from __future__ import annotations

import json
import os
import time
from pathlib import Path
import sys

# Ensure repository root is importable when executed as scripts/validate_rl_stream.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_bandit_training() -> dict:
    """Run a short bandit RL training loop using build-only action (fast/safe)."""
    from dspy_agent.rl.rlkit import TrainerConfig, bandit_trainer
    from dspy_agent.cli import _rl_build_make_env  # type: ignore

    ws = Path.cwd()
    # Restrict to a light action to avoid external tooling (e.g., ruff)
    make_env = _rl_build_make_env(
        ws,
        verifiers_module=None,
        weights={"pass_rate": 1.0, "blast_radius": 1.0},
        penalty_kinds=["blast_radius"],
        clamp01_kinds=["pass_rate"],
        scales={"blast_radius": (0.0, 1.0)},
        test_cmd=None,
        lint_cmd="python -c 'print(0)'",
        build_cmd="python -m compileall -q .",
        timeout_sec=60,
        actions=["build"],
    )
    cfg = TrainerConfig(steps=8, n_envs=1, policy="epsilon-greedy", policy_kwargs={"epsilon": 0.1})
    stats = bandit_trainer(make_env, cfg)
    rewards = stats.rewards or []
    return {"steps": len(rewards), "avg_reward": (sum(rewards) / len(rewards)) if rewards else 0.0}


def run_stream_stack() -> dict:
    """Start the local streaming stack, emit a few synthetic errors, and report health."""
    from dspy_agent.streaming.streaming_runtime import start_local_stack
    from dspy_agent.streaming.streamkit import Aggregator, LocalBus

    ws = Path.cwd()
    # Ensure a backend log exists and has some error lines to pick up
    log_dir = ws / "logs" / "backend"
    log_dir.mkdir(parents=True, exist_ok=True)
    test_log = log_dir / "test_backend.log"
    with test_log.open("a") as f:
        f.write("2025-09-21 10:00:00 INFO start\n")
        f.write("2025-09-21 10:00:01 ERROR something failed\n")

    threads, bus = start_local_stack(ws, None, storage=None, kafka=None)

    # Emit another error after startup to ensure the tailer sees new lines
    time.sleep(1.0)
    with test_log.open("a") as f:
        f.write("2025-09-21 10:00:02 ERROR secondary failure\n")
    time.sleep(1.0)

    # Try to force an immediate flush on aggregators
    for t in threads:
        try:
            if isinstance(t, Aggregator):
                t.flush_now()
        except Exception:
            pass

    # Give the pipeline a moment to propagate and vectorize
    time.sleep(1.0)

    # Collect metrics and snapshots
    bus_metrics = {}
    try:
        bus_metrics = bus.metrics()  # type: ignore[attr-defined]
    except Exception:
        bus_metrics = {}

    feature_snap = None
    try:
        feature_snap = bus.feature_snapshot()  # type: ignore[attr-defined]
    except Exception:
        feature_snap = None

    srl_path = ws / ".dspy_stream_rl.json"
    srl = None
    if srl_path.exists():
        try:
            srl = json.loads(srl_path.read_text())
        except Exception:
            srl = None

    # Stop threads gracefully
    for t in threads:
        try:
            stop = getattr(t, "stop", None)
            if callable(stop):
                stop()
        except Exception:
            pass

    # Quick health summary
    topics = list((bus_metrics.get("topics") or {}).keys()) if isinstance(bus_metrics, dict) else []
    return {
        "topics": topics,
        "dlq_total": (bus_metrics or {}).get("dlq_total", 0),
        "feature_snapshot": feature_snap,
        "stream_rl": srl,
    }


def main() -> None:
    out = {"rl": None, "stream": None}
    try:
        out["rl"] = run_bandit_training()
    except Exception as e:
        out["rl"] = {"error": str(e)}
    try:
        out["stream"] = run_stream_stack()
    except Exception as e:
        out["stream"] = {"error": str(e)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
