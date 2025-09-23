from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root import
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from dspy_agent.rl.rlkit import TrainerConfig, bandit_trainer
    from dspy_agent.cli import _rl_build_make_env  # relies on stubbed dspy indirectly

    ws = REPO_ROOT
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
    cfg = TrainerConfig(steps=12, n_envs=1, policy="epsilon-greedy", policy_kwargs={"epsilon": 0.1})
    stats = bandit_trainer(make_env, cfg)
    rewards = stats.rewards or []
    print(f"rl_ok steps={len(rewards)} avg={ (sum(rewards)/len(rewards)) if rewards else 0.0 :.3f}")


if __name__ == "__main__":
    main()

