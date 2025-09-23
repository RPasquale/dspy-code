from __future__ import annotations

"""PufferRL PPO (compat wrapper)

This module is now a thin compatibility wrapper that delegates to
``dspy_agent.rl.rlkit.run_puffer_ppo`` — the canonical implementation used by
the CLI and tests. It avoids diverging behavior between two similar entry points
and removes placeholder code paths.

Usage:
    from dspy_agent.rl.puffer_ppo_shell import run_puffer_ppo
    run_puffer_ppo(make_env, n_envs=8, total_steps=100_000)

Installation:
    Requires pufferlib>=3.0.0 for vectorization and PuffeRL PPO.
"""

from typing import Callable

from .rlkit import run_puffer_ppo as _run_puffer_ppo


def run_puffer_ppo(make_env: Callable[[], object], n_envs: int = 8, total_steps: int = 100_000):  # pragma: no cover - optional
    """Run PuffeRL PPO using the canonical rlkit implementation.

    See :func:`dspy_agent.rl.rlkit.run_puffer_ppo` for details and exceptions
    (raises if pufferlib is not installed).
    """
    return _run_puffer_ppo(make_env=make_env, n_envs=n_envs, total_steps=total_steps)
