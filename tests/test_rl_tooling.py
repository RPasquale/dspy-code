from collections import Counter
import importlib

import pytest

from dspy_agent.rl.rlkit import (
    AgentResult,
    EnvConfig,
    RewardConfig,
    RLToolEnv,
    ToolAction,
    bandit_trainer,
    bandit_trainer_puffer,
    aggregate_reward,
    TrainerConfig,
)


class DummyVerifier:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def __call__(self, result: AgentResult) -> float:  # pragma: no cover - simple adapter
        return float(result.metrics.get(self.kind, 0.0))


def _build_env_factory() -> tuple:
    reward_cfg = RewardConfig(
        weights={"pass_rate": 1.0, "blast_radius": 0.5},
        penalty_kinds=("blast_radius",),
    )

    verifiers = [DummyVerifier("pass_rate"), DummyVerifier("blast_radius")]

    def reward_fn(result, verifiers_list, weights_map):  # type: ignore[no-redef]
        return aggregate_reward(result, verifiers_list, reward_cfg)

    def executor(action: ToolAction, _args: dict) -> AgentResult:
        if action == ToolAction.PATCH:
            metrics = {"pass_rate": 0.92, "blast_radius": 0.05}
        else:
            metrics = {"pass_rate": 0.45, "blast_radius": 0.20}
        return AgentResult(metrics=metrics, info={})

    def make_env() -> RLToolEnv:
        env_cfg = EnvConfig(
            verifiers=verifiers,
            reward_fn=reward_fn,
            weights=reward_cfg.weights,
            action_args=None,
            allowed_actions=["run_tests", "patch"],
        )
        return RLToolEnv(executor=executor, cfg=env_cfg, episode_len=1)

    return make_env


def test_rl_tool_env_reward_and_info():
    make_env = _build_env_factory()
    env = make_env()

    obs, info = env.reset()
    assert obs == []
    assert info["t"] == 0

    patch_idx = env.action_names.index("patch")
    obs_after, reward, terminated, truncated, step_info = env.step(patch_idx)

    expected_reward = 0.92 - 0.5 * 0.05
    assert reward == pytest.approx(expected_reward, rel=1e-3)
    assert terminated is True
    assert truncated is False
    assert step_info["tool"] == "patch"
    assert len(obs_after) == 2
    assert obs_after[0] == pytest.approx(0.92, rel=1e-3)
    assert obs_after[1] == pytest.approx(0.05, rel=1e-3)

    env.reset()
    run_idx = env.action_names.index("run_tests")
    _obs2, reward_run, _term2, _trunc2, info_run = env.step(run_idx)
    assert reward_run < reward
    assert info_run["tool"] == "run_tests"


def test_bandit_trainer_prefers_high_reward_action():
    make_env = _build_env_factory()
    cfg = TrainerConfig(steps=200, policy="epsilon-greedy", policy_kwargs={"epsilon": 0.2, "seed": 7}, n_envs=1)

    stats = bandit_trainer(make_env, cfg)

    tools = [info.get("tool") for info in stats.infos if isinstance(info, dict) and info.get("tool")]
    counts = Counter(tools)

    assert len(stats.rewards) == cfg.steps
    patch_count = counts.get("patch", 0)
    run_count = counts.get("run_tests", 0)

    assert patch_count > run_count
    assert patch_count >= int(cfg.steps * 0.6)


@pytest.mark.skipif(importlib.util.find_spec("pufferlib") is None, reason="pufferlib not installed")
def test_bandit_trainer_puffer_vectorizes_and_learns():
    make_env = _build_env_factory()
    cfg = TrainerConfig(steps=40, policy="epsilon-greedy", policy_kwargs={"epsilon": 0.2, "seed": 11}, n_envs=2)

    stats = bandit_trainer_puffer(make_env, cfg)

    assert len(stats.rewards) == cfg.steps * cfg.n_envs
    tools = [info.get("tool") for info in stats.infos if isinstance(info, dict) and info.get("tool")]
    counts = Counter(tools)
    patch_count = counts.get("patch", 0)
    run_count = counts.get("run_tests", 0)

    assert patch_count > 0
    assert patch_count > run_count
