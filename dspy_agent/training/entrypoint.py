from __future__ import annotations

"""
Simple training entrypoint that wires RL + verifiers + analytics persistence.

Usage:
  python -m dspy_agent.training.entrypoint \
    --workspace /path/to/ws \
    --signature CodeContextSig \
    --verifiers dspy_agent.verifiers.custom \
    --steps 200 --env production

This will run a bandit trainer with the specified verifiers and persist
verifier_scores + rewards labeled with the signature name.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

from dspy_agent.rl.rlkit import (
    RLToolEnv, EnvConfig, RewardConfig, aggregate_reward,
    bandit_trainer, make_bandit, make_default_analytics_cb,
)
from dspy_agent.rl.rlkit import detect_toolchain, ToolchainExecutor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', type=str, required=True)
    ap.add_argument('--signature', type=str, required=True, help='Signature name for labeling analytics')
    ap.add_argument('--verifiers', type=str, default='dspy_agent.verifiers.custom', help='Module path exporting get_verifiers()')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--n-envs', type=int, default=1)
    ap.add_argument('--policy', type=str, default='epsilon-greedy')
    ap.add_argument('--env', type=str, default='development', help='Environment label for analytics')
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    tcfg = detect_toolchain(ws)
    execu = ToolchainExecutor(tcfg)

    # Load verifiers dynamically
    import importlib
    mod = importlib.import_module(args.verifiers)
    if hasattr(mod, 'get_verifiers'):
        verifiers = list(mod.get_verifiers())
    else:
        raise RuntimeError(f"Verifier module {args.verifiers} missing get_verifiers()")

    # Reward config: default weights = 1.0 for all kinds
    weights = {getattr(v, 'kind', f'v{i}'): 1.0 for i, v in enumerate(verifiers)}
    rc = RewardConfig(weights=weights)

    def reward_fn(result, vlist, wmap):
        return aggregate_reward(result, vlist, rc)

    env_cfg = EnvConfig(
        verifiers=verifiers,
        reward_fn=reward_fn,
        weights=weights,
        context_provider=None,
        action_args=None,
        allowed_actions=None,
    )

    def executor(tool, args) -> Any:
        # Bridge ToolchainExecutor to RL AgentResult
        return execu(tool, args)

    def make_env() -> RLToolEnv:
        env = RLToolEnv(executor, env_cfg, episode_len=1)
        # Tag env with signature for analytics; VectorRunner will include this in info
        try:
            env.set_signature_name(args.signature)
        except Exception:
            pass
        return env

    # Analytics callback for labeled persistence
    cb = make_default_analytics_cb(args.signature, env=args.env)

    from dspy_agent.rl.rlkit import TrainerConfig
    cfg = TrainerConfig(steps=int(args.steps), policy=args.policy, policy_kwargs={}, n_envs=max(1, int(args.n_envs)))
    bandit_trainer(make_env, cfg, analytics_cb=cb)


if __name__ == '__main__':
    main()
