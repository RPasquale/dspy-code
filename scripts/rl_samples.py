from __future__ import annotations

"""Tiny RL samples for quick validation.

Runs both the bandit trainer and the neural REINFORCE trainer on a dummy
environment matching tests/test_rl_tooling.py.

Usage:
  python -m scripts.rl_samples --steps 100 --neural-steps 200
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple, List, Optional

from dspy_agent.rl.rlkit import (
    AgentResult,
    EnvConfig,
    RLToolEnv,
    ToolAction,
    aggregate_reward,
    RewardConfig,
    TrainerConfig,
    bandit_trainer,
    train_puffer_policy,
)


@dataclass
class DummyVerifier:
    kind: str
    def __call__(self, result: AgentResult) -> float:
        return float(result.metrics.get(self.kind, 0.0))


def build_dummy_env() -> RLToolEnv:
    reward_cfg = RewardConfig(
        weights={"pass_rate": 1.0, "blast_radius": 0.5},
        penalty_kinds=("blast_radius",),
    )
    verifiers: List[DummyVerifier] = [DummyVerifier("pass_rate"), DummyVerifier("blast_radius")]

    def reward_fn(result: AgentResult, verifiers_list: Iterable[DummyVerifier], weights_map: Mapping[str, float]) -> Tuple[float, List[float], Dict[str, float]]:
        return aggregate_reward(result, verifiers_list, reward_cfg)

    def executor(action: ToolAction, _args: dict) -> AgentResult:
        if action == ToolAction.PATCH:
            metrics = {"pass_rate": 0.92, "blast_radius": 0.05}
        else:
            metrics = {"pass_rate": 0.45, "blast_radius": 0.20}
        return AgentResult(metrics=metrics, info={})

    env_cfg = EnvConfig(
        verifiers=verifiers,
        reward_fn=reward_fn,
        weights=reward_cfg.weights,
        action_args=None,
        allowed_actions=["run_tests", "patch"],
    )
    return RLToolEnv(executor=executor, cfg=env_cfg, episode_len=1)


def make_env_factory():
    def make_env() -> RLToolEnv:
        return build_dummy_env()
    return make_env


def main(steps: int = 100, neural_steps: int = 200, n_envs: int = 2) -> None:
    # Bandit trainer (epsilon-greedy)
    cfg = TrainerConfig(steps=int(steps), policy="epsilon-greedy", policy_kwargs={"epsilon": 0.2}, n_envs=1)
    stats = bandit_trainer(make_env_factory(), cfg)
    print(f"[bandit] steps={len(stats.rewards)} avg_reward={sum(stats.rewards)/max(1,len(stats.rewards)):.3f}")

    # Neural REINFORCE trainer (requires torch)
    try:
        import torch  # noqa: F401
    except Exception:
        print("[neural] torch not available; skipping neural sample")
        return
    stats2 = train_puffer_policy(
        make_env=make_env_factory(),
        steps=int(neural_steps),
        n_envs=int(n_envs),
        entropy_coef=0.02,
        replay_capacity=1024,
        replay_batch=64,
        grad_clip=1.0,
        checkpoint_interval=0,
    )
    rewards = stats2.rewards or []
    print(f"[neural] steps={len(rewards)} avg_reward={sum(rewards)/max(1,len(rewards)):.3f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--neural-steps", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=2)
    args = p.parse_args()
    main(steps=args.steps, neural_steps=args.neural_steps, n_envs=args.n_envs)

