from __future__ import annotations

"""RL Showcase Script

Runs a small, self-contained demo of the different RL options available:
1) Bandit trainer (single-env)
2) Bandit trainer with PufferLib vectorization (if available)
3) Neural REINFORCE trainer (if torch available)
4) PuffeRL PPO (if pufferlib available)

The environment is a tiny, deterministic RLToolEnv that exposes two actions:
 - run_tests (lower reward)
 - patch (higher reward)

Usage:
  blampert-rl-showcase --steps 100 --neural-steps 200 --puffer-steps 40
"""

import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple, List

from dspy_agent.rl.rlkit import (
    AgentResult,
    EnvConfig,
    RLToolEnv,
    ToolAction,
    aggregate_reward,
    RewardConfig,
    TrainerConfig,
    bandit_trainer,
    bandit_trainer_puffer,
    train_puffer_policy,
    run_puffer_ppo,
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


def _summarize_infos(infos: List[Dict[str, object]]) -> Dict[str, int]:
    tools = [str(it.get("tool")) for it in infos if isinstance(it, dict) and it.get("tool")]
    return dict(Counter(tools))


def main(steps: int = 120, puffer_steps: int = 60, neural_steps: int = 200, n_envs: int = 2) -> None:
    print("=== RL Showcase ===")
    env_factory = make_env_factory()

    # 1) Bandit trainer (single env)
    print("\n[1] Bandit trainer (epsilon-greedy, 1 env)")
    t0 = time.time()
    cfg = TrainerConfig(steps=int(steps), policy="epsilon-greedy", policy_kwargs={"epsilon": 0.2}, n_envs=1)
    stats = bandit_trainer(env_factory, cfg)
    dt = time.time() - t0
    avg = sum(stats.rewards)/max(1,len(stats.rewards)) if stats.rewards else 0.0
    print(f"  steps={len(stats.rewards)} avg_reward={avg:.3f} time={dt:.2f}s tools={_summarize_infos(stats.infos)}")

    # 2) Bandit trainer with PufferLib vectorization
    print("\n[2] Bandit trainer (PufferLib vectorized) — optional")
    try:
        import pufferlib  # noqa: F401
        t0 = time.time()
        cfg = TrainerConfig(steps=int(puffer_steps), policy="epsilon-greedy", policy_kwargs={"epsilon": 0.2}, n_envs=max(2, n_envs))
        stats_pf = bandit_trainer_puffer(env_factory, cfg)
        dt = time.time() - t0
        avg = sum(stats_pf.rewards)/max(1,len(stats_pf.rewards)) if stats_pf.rewards else 0.0
        print(f"  steps={len(stats_pf.rewards)} avg_reward={avg:.3f} time={dt:.2f}s tools={_summarize_infos(stats_pf.infos)}")
    except Exception as e:
        print(f"  [skipped] pufferlib not available: {e}")

    # 3) Neural REINFORCE trainer (torch)
    print("\n[3] Neural REINFORCE trainer — optional")
    try:
        import torch  # noqa: F401
        t0 = time.time()
        stats_nn = train_puffer_policy(
            make_env=env_factory,
            steps=int(neural_steps),
            n_envs=int(n_envs),
            entropy_coef=0.02,
            replay_capacity=1024,
            replay_batch=64,
            grad_clip=1.0,
            checkpoint_interval=0,
            verbose=False,
        )
        dt = time.time() - t0
        avg = sum(stats_nn.rewards)/max(1,len(stats_nn.rewards)) if stats_nn.rewards else 0.0
        print(f"  steps={len(stats_nn.rewards)} avg_reward={avg:.3f} time={dt:.2f}s")
    except Exception as e:
        print(f"  [skipped] torch not available: {e}")

    # 4) Puffer PPO (vectorized PPO) — optional
    print("\n[4] Puffer PPO — optional")
    try:
        import pufferlib  # noqa: F401
        print("  Running short PPO sample (may take a while)…")
        run_puffer_ppo(make_env=env_factory, n_envs=max(2, n_envs), total_steps=5_000)
        print("  PPO finished (short run)")
    except Exception as e:
        print(f"  [skipped] pufferlib not available: {e}")

    print("\n=== Done ===")


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _unified_diff(path: str, old: str, new: str) -> str:
    import difflib
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile=path, tofile=path, lineterm='')
    return "\n".join(diff)


def real_patch_demo() -> None:
    print("\n[5] Real PATCH demo with policy/preferences (monkeypatched)")
    from pathlib import Path
    import types
    import json
    from dspy_agent.rl.rlkit import ToolchainConfig, ToolchainExecutor, ToolAction

    # Setup temp workspace
    ws = Path('.smoke_ws') / f'showcase_{int(time.time())}'
    src = ws / 'src'
    mod = src / 'mod.py'
    base = """def calc(x: int) -> int:\n    return x + 1\n"""
    _write(mod, base)

    # Preferences: forbid broad except-pass pattern
    prefs = ws / '.dspy_preferences.json'
    prefs_obj = {
        "forbidden_commands": [],
        "forbidden_code_patterns": [r"except Exception:\\s*pass"],
        "max_blast_radius": 1000
    }
    _write(prefs, json.dumps(prefs_obj, indent=2))

    # Monkeypatch DSPy skills used during PATCH
    # FileLocator: emit the target file as candidate
    import dspy_agent.skills.file_locator as FL
    import dspy_agent.skills.code_edit as CE
    import dspy_agent.skills.patch_verifier as PV
    import dspy_agent.skills.test_planner as TP

    class MockLocator:
        def __call__(self, *args, **kwargs):
            return types.SimpleNamespace(file_candidates=str(mod), notes="demo")

    # Two variants of CodeEdit to show violation vs compliance
    def make_violation_patch() -> str:
        new = """def calc(x: int) -> int:\n    try:\n        return x + 1\n    except Exception:\n        pass\n"""
        return _unified_diff(str(mod), base, new)

    def make_clean_patch() -> str:
        new = """def calc(x: int) -> int:\n    # safer early return\n    return (x or 0) + 1\n"""
        return _unified_diff(str(mod), base, new)

    class MockEditorViolation:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return types.SimpleNamespace(patch=make_violation_patch())

    class MockEditorClean:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return types.SimpleNamespace(patch=make_clean_patch())

    class MockVerifier:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return types.SimpleNamespace(verdict='pass', reasons='ok')

    class MockPlanner:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return types.SimpleNamespace(commands='')  # no tests

    # Install mocks
    FL.FileLocator = MockLocator  # type: ignore
    PV.PatchVerifier = MockVerifier  # type: ignore
    TP.TestPlanner = MockPlanner  # type: ignore

    # Executor
    execu = ToolchainExecutor(ToolchainConfig(workspace=ws))

    def run_once(editor_cls) -> Dict[str, object]:
        CE.CodeEdit = editor_cls  # type: ignore
        result = execu(ToolAction.PATCH, {
            'task': 'Insert change',
            'context': 'demo',
            'revert_always': True,
        })
        return result.metrics  # type: ignore

    # Violation patch (triggers preferences)
    m1 = run_once(MockEditorViolation)
    print(f"  violation: pass_rate={m1.get('pass_rate')} blast={m1.get('blast_radius')} policy={m1.get('quality_policy')} prefs={m1.get('quality_preferences', 1.0)}")
    # Clean patch
    m2 = run_once(MockEditorClean)
    print(f"  clean:     pass_rate={m2.get('pass_rate')} blast={m2.get('blast_radius')} policy={m2.get('quality_policy')} prefs={m2.get('quality_preferences', 1.0)}")




if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--puffer-steps", type=int, default=60)
    p.add_argument("--neural-steps", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=2)
    args = p.parse_args()
    main(steps=args.steps, puffer_steps=args.puffer_steps, neural_steps=args.neural_steps, n_envs=args.n_envs)
