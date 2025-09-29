from __future__ import annotations

# Consolidated RL toolkit: env, verifiers adapter, bandits, runner, trainer,
# executor, verifiers loader, RL config, and PuffeRL shell.
#
# This single module replaces prior small files to keep the codebase compact
# while preserving functionality. Public symbols are re-exported via
# dspy_agent.rl.__init__ for stable imports.

import json
import os
import math
import random
import shlex
import time
import logging
from dataclasses import dataclass, field
import sys
import shutil
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


class ToolAction(IntEnum):
    RUN_TESTS = 0
    LINT = 1
    BUILD = 2
    PATCH = 3
    SHELL_LS = 4
    SHELL_PWD = 5
    SHELL_CAT = 6
    SHELL_CD = 7
    SHELL_RUN = 8

    @classmethod
    def names(cls) -> List[str]:
        return [action.name.lower() for action in cls]

    @classmethod
    def from_any(cls, value: object) -> "ToolAction":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, str):
            key = value.strip().lower()
            if not key:
                raise ValueError('Empty ToolAction name')
            normalized = key.replace('-', '_').replace(' ', '_')
            for action in cls:
                base = action.name.lower()
                if normalized == base:
                    return action
                if normalized == base.replace('run_', ''):
                    return action
            aliases = {
                'tests': cls.RUN_TESTS,
                'test': cls.RUN_TESTS,
                'ls': cls.SHELL_LS,
                'list': cls.SHELL_LS,
                'pwd': cls.SHELL_PWD,
                'cat': cls.SHELL_CAT,
                'view': cls.SHELL_CAT,
                'cd': cls.SHELL_CD,
                'chdir': cls.SHELL_CD,
                'shell': cls.SHELL_RUN,
                'sh': cls.SHELL_RUN,
                'run': cls.SHELL_RUN,
                'cli': cls.SHELL_RUN,
            }
            if normalized in aliases:
                return aliases[normalized]
        raise ValueError(f'Unknown ToolAction: {value!r}')


@dataclass
class AgentResult:
    metrics: Mapping[str, Any]
    info: Optional[Mapping[str, Any]] = None


class VerifierProtocol(Protocol):
    kind: str
    def __call__(self, result: AgentResult) -> float: ...  # pragma: no cover


ActionExecutor = Callable[[ToolAction, Dict[str, Any]], AgentResult]
ContextProvider = Callable[[], List[float]]


@dataclass
class EnvConfig:
    verifiers: Iterable[VerifierProtocol]
    reward_fn: Callable[[AgentResult, Iterable[VerifierProtocol], Mapping[str, float]], Tuple[float, List[float], Dict[str, float]]]
    weights: Mapping[str, float]
    context_provider: Optional[ContextProvider] = None
    action_args: Optional[Mapping[str, Dict[str, Any]]] = None
    allowed_actions: Optional[Iterable[str]] = None


# Optional gym/numpy are disabled by default in restricted environments to avoid
# import-time crashes from native extensions. Enable via DSPY_ENABLE_GYM=1.
_HAS_GYM = False
_spaces = None  # type: ignore
_np = None  # type: ignore
if os.getenv("DSPY_ENABLE_GYM", "0").lower() in {"1", "true", "yes"}:
    try:  # optional gymnasium for vectorized env compatibility
        import gymnasium as _gym  # type: ignore
        from gymnasium import spaces as _spaces  # type: ignore
        import numpy as _np  # type: ignore
        _HAS_GYM = True
    except Exception:  # pragma: no cover - optional
        _HAS_GYM = False
        _spaces = None  # type: ignore
        _np = None  # type: ignore


class RLToolEnv:
    metadata = {"render.modes": ["human"]}

    def __init__(self, executor: ActionExecutor, cfg: EnvConfig, episode_len: int = 1) -> None:
        self._exec = executor
        self._cfg = cfg
        self._episode_len = max(1, episode_len)
        self._t = 0
        self._last_obs: List[float] = []
        self._signature_name: Optional[str] = None
        actions: List[ToolAction] = []
        seen: set[ToolAction] = set()
        if cfg.allowed_actions:
            for spec in cfg.allowed_actions:
                try:
                    action = ToolAction.from_any(spec)
                except Exception:
                    continue
                if action in seen:
                    continue
                seen.add(action)
                actions.append(action)
        if not actions:
            actions = [action for action in ToolAction]
        self._actions = actions
        self._action_names = [action.name.lower() for action in self._actions]
        aa: Dict[int, Dict[str, Any]] = {}
        if cfg.action_args:
            for idx, action in enumerate(self._actions):
                key = action.name.lower()
                if key in cfg.action_args:
                    aa[idx] = dict(cfg.action_args[key])
        self._action_args = aa
        if _HAS_GYM:
            base = len(list(self._cfg.verifiers))
            ctx_len = 0
            try:
                ctx_len = len(self._cfg.context_provider() or []) if self._cfg.context_provider else 0
            except Exception:
                ctx_len = 0
            obs_dim = max(1, base + ctx_len)
            self.observation_space = _spaces.Box(low=-1e9, high=1e9, shape=(obs_dim,), dtype=_np.float32)  # type: ignore[attr-defined]
            self.action_space = _spaces.Discrete(len(self._actions))  # type: ignore[attr-defined]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[List[float], Dict[str, Any]]:
        self._t = 0
        ctx = self._cfg.context_provider() if self._cfg.context_provider else []
        # Initialize with zero verifier scores + context for proper observation dimension
        verifier_scores = [0.0] * len(list(self._cfg.verifiers))
        self._last_obs = verifier_scores + list(ctx)
        return self._last_obs, {"t": self._t}

    def step(self, action: int) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
        idx = int(action)
        if idx < 0 or idx >= len(self._actions):
            raise IndexError(f'Action index out of range: {action!r}')
        tool = self._actions[idx]
        args = self._action_args.get(idx, {})
        result = self._exec(tool, args)
        reward, vvec, details = self._cfg.reward_fn(result, self._cfg.verifiers, self._cfg.weights)
        ctx = self._cfg.context_provider() if self._cfg.context_provider else []
        obs = vvec + ctx
        self._last_obs = obs
        self._t += 1
        terminated = self._t >= self._episode_len
        truncated = False
        info: Dict[str, Any] = {"tool": self._action_names[idx], "verifier_scores": details}
        if self._signature_name:
            info["signature_name"] = self._signature_name
        return obs, float(reward), bool(terminated), bool(truncated), info

    @property
    def action_dim(self) -> int:
        return len(self._actions)

    @property
    def action_names(self) -> List[str]:
        return list(self._action_names)

    @property
    def obs_size_hint(self) -> int:
        base = len(list(self._cfg.verifiers))
        ctx = len(self._cfg.context_provider() or []) if self._cfg.context_provider else 0
        return base + ctx

    # Allow attaching external feature provider post-init (for vector feeder integration)
    def set_context_provider(self, provider: ContextProvider) -> None:
        self._cfg.context_provider = provider
        # Update gym observation space if available
        if _HAS_GYM:
            try:
                base = len(list(self._cfg.verifiers))
                ctx_len = len(self._cfg.context_provider() or []) if self._cfg.context_provider else 0
                obs_dim = max(1, base + ctx_len)
                self.observation_space = _spaces.Box(low=-1e9, high=1e9, shape=(obs_dim,), dtype=_np.float32)  # type: ignore[attr-defined]
            except Exception:
                pass

    # Optional: tag the env with a signature for analytics persistence
    def set_signature_name(self, name: Optional[str]) -> None:
        self._signature_name = (name or "").strip() or None


# ------------------------
# Verifiers → Reward
# ------------------------

@dataclass
class RewardConfig:
    weights: Mapping[str, float]
    penalty_kinds: Iterable[str] = ()
    clamp01_kinds: Iterable[str] = ()
    scales: Mapping[str, Tuple[float, float]] = ()


def _scale(kind: str, val: float, cfg: RewardConfig) -> float:
    if kind in cfg.scales:
        lo, hi = cfg.scales[kind]
        if hi == lo:
            return 0.0
        val = (val - lo) / (hi - lo)
    if kind in cfg.clamp01_kinds:
        val = 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)
    return float(val)


def aggregate_reward(
    result: AgentResult,
    verifiers: Iterable[VerifierProtocol],
    weights: Mapping[str, float] | RewardConfig,
) -> Tuple[float, List[float], Dict[str, float]]:
    cfg = weights if isinstance(weights, RewardConfig) else RewardConfig(weights=weights)
    vec: List[float] = []
    details: Dict[str, float] = {}
    total = 0.0
    pk = set(cfg.penalty_kinds)
    for v in verifiers:
        raw = float(v(result))
        w = float(cfg.weights.get(v.kind, 1.0))
        val = _scale(v.kind, raw, cfg)
        contrib = -abs(val) if v.kind in pk else val
        details[v.kind] = raw
        vec.append(val)
        total += w * contrib
    return float(total), vec, details


# -------------
# Bandits
# -------------

class BaseBandit:
    def __init__(self, n_actions: int, seed: Optional[int] = None) -> None:
        self.n = int(n_actions)
        self.rng = random.Random(seed)
    def select(self, ctx: Optional[List[float]] = None) -> int:  # pragma: no cover
        raise NotImplementedError
    def update(self, action: int, reward: float, ctx: Optional[List[float]] = None) -> None:  # pragma: no cover
        raise NotImplementedError


class EpsilonGreedy(BaseBandit):
    def __init__(self, n_actions: int, epsilon: float = 0.1, seed: Optional[int] = None) -> None:
        super().__init__(n_actions, seed)
        self.epsilon = float(epsilon)
        self.counts = [0] * n_actions
        self.values = [0.0] * n_actions
    def select(self, ctx: Optional[List[float]] = None) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n)
        return int(max(range(self.n), key=lambda a: self.values[a]))
    def update(self, action: int, reward: float, ctx: Optional[List[float]] = None) -> None:
        a = int(action); self.counts[a] += 1; n = self.counts[a]; q = self.values[a]
        self.values[a] = q + (reward - q) / n


class UCB1(BaseBandit):
    def __init__(self, n_actions: int, c: float = 2.0, seed: Optional[int] = None) -> None:
        super().__init__(n_actions, seed)
        self.c = float(c)
        self.counts = [0] * n_actions
        self.values = [0.0] * n_actions
        self.t = 0
    def select(self, ctx: Optional[List[float]] = None) -> int:
        self.t += 1
        for a in range(self.n):
            if self.counts[a] == 0:
                return a
        scores = [self.values[a] + self.c * math.sqrt(math.log(self.t) / self.counts[a]) for a in range(self.n)]
        return int(max(range(self.n), key=lambda a: scores[a]))
    def update(self, action: int, reward: float, ctx: Optional[List[float]] = None) -> None:
        a = int(action); self.counts[a] += 1; n = self.counts[a]; q = self.values[a]
        self.values[a] = q + (reward - q) / n


class ThompsonBeta(BaseBandit):
    def __init__(self, n_actions: int, seed: Optional[int] = None) -> None:
        super().__init__(n_actions, seed)
        self.alpha = [1.0] * n_actions
        self.beta = [1.0] * n_actions
    def _sample_beta(self, a: float, b: float) -> float:
        def rgamma(k: float) -> float:
            if k <= 1.0:
                return self.rng.random() ** (1.0 / k)
            d = k - 1.0 / 3.0; c = 1.0 / math.sqrt(9.0 * d)
            while True:
                x = self.rng.normalvariate(0.0, 1.0); v = (1.0 + c * x) ** 3
                if v <= 0: continue
                u = self.rng.random()
                if u < 1 - 0.0331 * (x ** 4): return d * v
                if math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)): return d * v
        ga = rgamma(a); gb = rgamma(b); return ga / (ga + gb)
    def select(self, ctx: Optional[List[float]] = None) -> int:
        samples = [self._sample_beta(self.alpha[a], self.beta[a]) for a in range(self.n)]
        return int(max(range(self.n), key=lambda a: samples[a]))
    def update(self, action: int, reward: float, ctx: Optional[List[float]] = None) -> None:
        a = int(action); r = max(0.0, min(1.0, float(reward)))
        self.alpha[a] += r; self.beta[a] += 1.0 - r


def make_bandit(name: str, n_actions: int, **kwargs) -> BaseBandit:
    key = name.strip().lower()
    if key in {"eps", "epsilon", "epsilon-greedy", "egreedy"}: return EpsilonGreedy(n_actions, epsilon=float(kwargs.get("epsilon", 0.1)), seed=kwargs.get("seed"))
    if key in {"ucb", "ucb1"}: return UCB1(n_actions, c=float(kwargs.get("c", 2.0)), seed=kwargs.get("seed"))
    if key in {"thompson", "beta", "ts"}: return ThompsonBeta(n_actions, seed=kwargs.get("seed"))
    raise ValueError(f"unknown bandit policy: {name}")


# ---------------
# Vector runner
# ---------------

@dataclass
class EpisodeStats:
    rewards: List[float]
    infos: List[Dict[str, Any]]


class VectorRunner:
    def __init__(self, envs: List[Any], policy: Any, analytics_cb: Optional[Callable[[int, Dict[str, Any], float], Optional[Dict[str, Any]]]] = None) -> None:
        self.envs = envs
        self.policy = policy
        self.analytics_cb = analytics_cb
    def run(self, steps: int) -> EpisodeStats:
        obs: List[List[float]] = []; infos: List[Dict[str, Any]] = []
        for e in self.envs:
            o, info = e.reset(); obs.append(o); infos.append(info)
        all_rewards: List[float] = []; all_infos: List[Dict[str, Any]] = []
        for _ in range(steps):
            for i, e in enumerate(self.envs):
                a = self.policy.select(obs[i]); o2, r, done, trunc, info = e.step(a)
                self.policy.update(a, r, o2); obs[i] = o2
                all_rewards.append(float(r)); all_infos.append(info)
                # Analytics hook: if info contains verifier_scores, persist using callback-provided metadata
                try:
                    if self.analytics_cb is not None and isinstance(info, dict):
                        hook = self.analytics_cb(i, info, float(r)) or {}
                        sig = hook.get('signature_name') if isinstance(hook.get('signature_name'), str) else None
                        doc_id = hook.get('doc_id') if isinstance(hook.get('doc_id'), str) else None
                        env = hook.get('environment') if isinstance(hook.get('environment'), str) else 'development'
                        action_type = hook.get('action_type') if isinstance(hook.get('action_type'), str) else 'VERIFICATION'
                        execution_time = float(hook.get('execution_time') or 0.0)
                        query = hook.get('query') if isinstance(hook.get('query'), str) else None
                        scores = info.get('verifier_scores') if isinstance(info.get('verifier_scores'), dict) else None
                        if sig and scores:
                            record_verifier_scores(sig, scores, float(r), environment=env, doc_id=doc_id, action_type=action_type, execution_time=execution_time, query=query)
                except Exception:
                    logger.exception("analytics_cb failed")
                if done or trunc: obs[i], _ = e.reset()
        return EpisodeStats(rewards=all_rewards, infos=all_infos)


# ---------------
# Trainer (bandit + neural)
# ---------------

class RewardNormalizer:
    """Online running-mean/std normalizer for scalar rewards."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.eps = float(eps)

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / max(1, self.n)
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return max(self.eps, (self.m2 / (self.n - 1)) ** 0.5)

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std


class CheckpointManager:
    """Minimal checkpoint manager for neural policy training.

    Saves JSON metadata and optionally torch state_dict() if torch is present.
    Disabled when `interval<=0`.
    """

    def __init__(self, out_dir: Path, interval: int = 0) -> None:
        self.dir = Path(out_dir)
        self.interval = int(interval)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._best: float = float('-inf')

    def maybe_save(self, step: int, avg_reward: float, policy=None, opt=None) -> None:
        if self.interval <= 0:
            return
        if step % self.interval != 0:
            return
        meta = {"step": int(step), "avg_reward": float(avg_reward)}
        try:
            (self.dir / 'latest.json').write_text(json.dumps(meta))
        except Exception:
            pass
        if avg_reward > self._best:
            self._best = float(avg_reward)
            try:
                (self.dir / 'best.json').write_text(json.dumps(meta))
            except Exception:
                pass
        try:
            import torch  # type: ignore
            if policy is not None:
                torch.save({'policy': policy.state_dict(), 'step': step}, self.dir / 'latest.pt')  # type: ignore
            if opt is not None:
                torch.save({'optimizer': opt.state_dict(), 'step': step}, self.dir / 'latest_opt.pt')  # type: ignore
        except Exception:
            pass

@dataclass
class TrainerConfig:
    steps: int = 100
    policy: str = "epsilon-greedy"
    policy_kwargs: Mapping[str, Any] = field(default_factory=dict)
    n_envs: int = 1


def bandit_trainer(make_env: Callable[[], RLToolEnv], cfg: TrainerConfig, analytics_cb: Optional[Callable[[int, Dict[str, Any], float], Optional[Dict[str, Any]]]] = None) -> EpisodeStats:
    envs = [make_env() for _ in range(max(1, cfg.n_envs))]
    policy = make_bandit(cfg.policy, envs[0].action_dim, **dict(cfg.policy_kwargs))
    return VectorRunner(envs, policy, analytics_cb=analytics_cb).run(cfg.steps)


# ---------------
# Vector Feeder wiring
# ---------------

def vector_context_provider(path: str, *, batch_size: int = 128, poll_sec: float = 2.0) -> ContextProvider:
    """Create a context provider backed by the VectorBatchFeeder.

    Returns a callable that yields the next vector from the feeder, or [] if none.
    """
    def _make():
        try:
            from .vector_feeder import VectorBatchFeeder  # type: ignore
            feeder = VectorBatchFeeder(path, batch_size=batch_size, poll_sec=poll_sec)
            gen = feeder.iter_batches()
            buffer: List[List[float]] = []
            def _provider() -> List[float]:
                nonlocal buffer
                if not buffer:
                    try:
                        batch, _meta = next(gen)
                        buffer = list(batch)
                    except StopIteration:
                        buffer = []
                    except Exception:
                        buffer = []
                if buffer:
                    return list(buffer.pop(0))
                return []
            return _provider
        except Exception:
            return lambda: []
    # Instantiate once and reuse closure
    provider = _make()
    return provider


def attach_vector_context(make_env: Callable[[], RLToolEnv], path: str, *, batch_size: int = 128, poll_sec: float = 2.0) -> Callable[[], RLToolEnv]:
    """Wrap an env factory to attach vector feeder context provider.

    Example:
        make_env2 = attach_vector_context(make_env, '/workspace/vectorized/embeddings')
        stats = bandit_trainer(make_env2, cfg)
    """
    provider = vector_context_provider(path, batch_size=batch_size, poll_sec=poll_sec)
    def factory() -> RLToolEnv:
        env = make_env()
        try:
            env.set_context_provider(provider)
        except Exception:
            pass
        return env
    return factory


def bandit_trainer_puffer(make_env: Callable[[], RLToolEnv], cfg: TrainerConfig, backend: str = "Serial", analytics_cb: Optional[Callable[[int, Dict[str, Any], float], Optional[Dict[str, Any]]]] = None) -> EpisodeStats:
    # Allow a deterministic fallback when backend is Serial or explicitly disabled
    try:
        if str(backend).strip().lower() in {"serial", "none", "fallback"}:
            return bandit_trainer(make_env, cfg, analytics_cb=analytics_cb)
    except Exception:
        pass
    try:
        try:
            import pufferlib.emulation as emulation  # type: ignore
            import pufferlib.vector as pvector  # type: ignore
        except ImportError:
            raise ImportError("PufferLib not available. Install with: pip install pufferlib>=3.0.0")
    except Exception as e:  # pragma: no cover - optional
        raise RuntimeError("PufferLib not available. Install with 'pip install .[rl]'") from e
    tmp_env = make_env(); n_actions = tmp_env.action_dim
    try:
        # Use a top-level creator to avoid pickling closures when using multiprocessing
        def _make_wrapped_env(*_args, **_kwargs):
            return emulation.GymnasiumPufferEnv(make_env())
        venv = pvector.make(_make_wrapped_env, num_envs=max(1, cfg.n_envs), backend=backend)
        obs, infos = venv.reset(); policy = make_bandit(cfg.policy, n_actions, **dict(cfg.policy_kwargs))
        all_rewards: List[float] = []; all_infos: List[dict] = []
        for _ in range(cfg.steps):
            actions = [int(policy.select(list(obs[i]) if not isinstance(obs[i], list) else obs[i])) for i in range(len(obs))]
            obs, rewards, terms, truncs, infos = venv.step(actions)
            # infos is a list-like of dicts
            step_infos = list(infos) if isinstance(infos, (list, tuple)) else [infos]
            for i, a in enumerate(actions):
                r = float(rewards[i]); policy.update(a, r, list(obs[i]) if not isinstance(obs[i], list) else obs[i])
                all_rewards.append(r); all_infos.append(step_infos[i] if i < len(step_infos) else {})
                # analytics hook
                try:
                    if analytics_cb is not None and i < len(step_infos) and isinstance(step_infos[i], dict):
                        hook = analytics_cb(i, step_infos[i], float(r)) or {}
                        sig = hook.get('signature_name') if isinstance(hook.get('signature_name'), str) else None
                        doc_id = hook.get('doc_id') if isinstance(hook.get('doc_id'), str) else None
                        env = hook.get('environment') if isinstance(hook.get('environment'), str) else 'development'
                        action_type = hook.get('action_type') if isinstance(hook.get('action_type'), str) else 'VERIFICATION'
                        execution_time = float(hook.get('execution_time') or 0.0)
                        query = hook.get('query') if isinstance(hook.get('query'), str) else None
                        scores = step_infos[i].get('verifier_scores') if isinstance(step_infos[i].get('verifier_scores'), dict) else None
                        if sig and scores:
                            record_verifier_scores(sig, scores, float(r), environment=env, doc_id=doc_id, action_type=action_type, execution_time=execution_time, query=query)
                except Exception:
                    logger.exception('analytics_cb failed (puffer)')
            obs, infos = venv.reset()
        try: venv.close()
        except Exception: pass
        return EpisodeStats(rewards=all_rewards, infos=all_infos)
    except Exception:
        # Fallback to simple trainer when vector backend is unavailable or errors
        return bandit_trainer(make_env, cfg, analytics_cb=analytics_cb)


def train_puffer_policy(*, make_env: Callable[[], RLToolEnv], steps: int = 1000, n_envs: int = 4, lr: float = 1e-3, seed: Optional[int] = None, verbose: bool = False, log_interval: int = 10, grad_clip: float = 1.0, checkpoint_dir: Optional[str] = None, checkpoint_interval: int = 0, early_stop_patience: int = 0, entropy_coef: float = 0.01, replay_capacity: int = 2048, replay_batch: int = 128) -> EpisodeStats:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception as e:  # pragma: no cover - optional
        raise RuntimeError("Neural training requires torch. Install with 'pip install .[rl]'") from e
    # Try PufferLib vectorized env for speed
    using_vec = False
    try:
        try:
            import pufferlib.emulation as emulation  # type: ignore
            import pufferlib.vector as pvector  # type: ignore
        except ImportError:
            raise ImportError("PufferLib not available. Install with: pip install pufferlib>=3.0.0")
        def creator(): return emulation.GymnasiumPufferEnv(make_env())
        venv = pvector.make(creator, num_envs=max(1, n_envs), backend="multiprocessing")
        tmp = make_env(); act_dim = tmp.action_dim
        obs_batch, _ = venv.reset(); obs_dim = len(obs_batch[0]); using_vec = True
    except Exception:
        envs = [make_env() for _ in range(max(1, n_envs))]
        obs0, _ = envs[0].reset(); obs_dim = len(obs0); act_dim = envs[0].action_dim

    class Policy(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__(); self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, out_dim))
        def forward(self, x): return self.net(x)

    policy = Policy(obs_dim, act_dim); opt = optim.Adam(policy.parameters(), lr=lr)
    running_mean = 0.0; beta = 0.9
    rewards_all: List[float] = []; infos_all: List[dict] = []

    if using_vec:
        # On-policy experience buffer for auxiliary replay steps
        from collections import deque
        replay = deque(maxlen=int(max(1, replay_capacity)))
        for step_idx in range(1, steps + 1):
            obs, _ = venv.reset()
            import torch
            obs_t = torch.tensor(obs, dtype=torch.float32)
            logits = policy(obs_t)
            m = torch.distributions.Categorical(logits=logits)
            actions = m.sample(); logp = m.log_prob(actions); ent = m.entropy()
            obs2, rewards, terms, truncs, infos = venv.step(actions.tolist())
            r_t = torch.tensor(rewards, dtype=torch.float32)
            avg_r = float(r_t.mean().item()); running_mean = beta * running_mean + (1 - beta) * avg_r
            _rn = RewardNormalizer(); _rn.update(avg_r); adv = (r_t - running_mean) / max(1e-6, _rn.std)
            # Entropy regularization
            loss = (-(logp * adv).mean()) - float(entropy_coef) * ent.mean()
            opt.zero_grad(); loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(grad_clip))
            except Exception:
                pass
            opt.step()
            # Push to replay
            try:
                for i in range(len(obs)):
                    replay.append((obs_t[i].detach(), int(actions[i].item()), float(r_t[i].item())))
            except Exception:
                pass
            # Auxiliary replay step if buffer has enough
            try:
                if len(replay) >= max(8, replay_batch):
                    idx = torch.randperm(len(replay))[: int(replay_batch)]
                    batch = [replay[i] for i in idx.tolist()]
                    b_obs = torch.stack([b[0] for b in batch])
                    b_act = torch.tensor([b[1] for b in batch], dtype=torch.long)
                    b_rew = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                    b_logits = policy(b_obs); bm = torch.distributions.Categorical(logits=b_logits)
                    b_logp = bm.log_prob(b_act); b_ent = bm.entropy()
                    b_adv = (b_rew - running_mean) / max(1e-6, _rn.std)
                    aux_loss = (-(b_logp * b_adv).mean()) - float(entropy_coef) * b_ent.mean()
                    opt.zero_grad(); aux_loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(grad_clip))
                    except Exception:
                        pass
                    opt.step()
            except Exception:
                pass
            rewards_all.extend([float(x) for x in r_t.tolist()]); infos_all.extend(infos if isinstance(infos, list) else [infos])
            if verbose and (step_idx % max(1, int(log_interval)) == 0):
                # Summarize tool usage from infos if present
                try:
                    from collections import Counter as _C
                    tools = _C([str(it.get("tool", "")) for it in (infos if isinstance(infos, list) else [infos]) if isinstance(it, dict)])
                    print(f"[rl] step={step_idx} avg_r={avg_r:.3f} tools={dict(tools)}")
                except Exception:
                    print(f"[rl] step={step_idx} avg_r={avg_r:.3f}")
        try: venv.close()
        except Exception: pass
    else:
        import torch
        ckpt = CheckpointManager(Path(checkpoint_dir) if checkpoint_dir else Path('.dspy_checkpoints') / 'rl', interval=int(checkpoint_interval or 0))
        rn = RewardNormalizer()
        best_avg = float('-inf')
        stagnant = 0
        from collections import deque
        replay = deque(maxlen=int(max(1, replay_capacity)))
        def select_action(o: List[float]):
            # Compute logits with grad so REINFORCE can backprop through log-prob
            obs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs)
            m = torch.distributions.Categorical(logits=logits)
            a_t = m.sample()
            logp = m.log_prob(a_t)
            return int(a_t.item()), logp.squeeze()
        obses: List[List[float]] = []
        for e in envs: o, _ = e.reset(); obses.append(o)
        for step_idx in range(1, steps + 1):
            batch_lp: List[torch.Tensor] = []; batch_r: List[float] = []
            batch_ent: List[torch.Tensor] = []
            batch_obs_t: List[torch.Tensor] = []
            batch_act: List[int] = []
            for i, e in enumerate(envs):
                a, lp = select_action(obses[i]); o2, r, done, trunc, info = e.step(a)
                rewards_all.append(float(r)); infos_all.append(info)
                obses[i] = o2 if not (done or trunc) else e.reset()[0]
                batch_lp.append(lp); batch_r.append(float(r))
                # Recompute dist to get entropy and record obs/action for replay
                obs_t = torch.tensor(o2, dtype=torch.float32).unsqueeze(0)
                batch_obs_t.append(obs_t.squeeze(0)); batch_act.append(a)
                logits = policy(obs_t); m = torch.distributions.Categorical(logits=logits)
                batch_ent.append(m.entropy().squeeze())
            avg_r = sum(batch_r) / max(1, len(batch_r)); running_mean = beta * running_mean + (1 - beta) * avg_r
            rn.update(avg_r)
            norm_adv = [(r - running_mean) / max(1e-6, rn.std) for r in batch_r]
            # Add entropy regularization
            pol_loss = -sum(lp * adv for lp, adv in zip(batch_lp, norm_adv)) / max(1, len(batch_lp))
            ent_term = -float(entropy_coef) * (sum(batch_ent) / max(1, len(batch_ent)))
            loss = pol_loss + ent_term
            opt.zero_grad(); loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(grad_clip))
            except Exception:
                pass
            opt.step()
            # Push batch into replay buffer
            try:
                for obs_t, act, rew in zip(batch_obs_t, batch_act, batch_r):
                    replay.append((obs_t.detach(), int(act), float(rew)))
            except Exception:
                pass
            # Auxiliary replay pass if buffer has enough
            try:
                if len(replay) >= max(8, replay_batch):
                    idx = torch.randperm(len(replay))[: int(replay_batch)]
                    batch = [replay[i] for i in idx.tolist()]
                    b_obs = torch.stack([b[0] for b in batch])
                    b_act = torch.tensor([b[1] for b in batch], dtype=torch.long)
                    b_rew = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                    b_logits = policy(b_obs); bm = torch.distributions.Categorical(logits=b_logits)
                    b_logp = bm.log_prob(b_act); b_ent = bm.entropy().mean()
                    b_adv = (b_rew - running_mean) / max(1e-6, rn.std)
                    aux_loss = (-(b_logp * b_adv).mean()) - float(entropy_coef) * b_ent
                    opt.zero_grad(); aux_loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(grad_clip))
                    except Exception:
                        pass
                    opt.step()
            except Exception:
                pass
            if verbose and (step_idx % max(1, int(log_interval)) == 0):
                print(f"[rl] step={step_idx} avg_r={avg_r:.3f}")
            ckpt.maybe_save(step_idx, avg_r, policy=policy, opt=opt)
            if early_stop_patience and avg_r <= best_avg + 1e-6:
                stagnant += 1
                if stagnant >= early_stop_patience:
                    if verbose:
                        print(f"[rl] early stop at step={step_idx} best_avg={best_avg:.3f}")
                    break
            else:
                best_avg = max(best_avg, avg_r)
                stagnant = 0
    return EpisodeStats(rewards=rewards_all, infos=infos_all)


# ---------------
# Executor (toolchain)
# ---------------

def _run(cmd: str, cwd: Path, timeout: int) -> Tuple[bool, str, float]:
    import subprocess as sp
    t0 = time.time()
    try:
        proc = sp.run(cmd, cwd=str(cwd), shell=True, stdout=sp.PIPE, stderr=sp.STDOUT, timeout=timeout, text=True)
        ok = (proc.returncode == 0); out = proc.stdout or ""
    except Exception as e:
        ok = False; out = str(e)
    return ok, out, time.time() - t0


def _parse_pytest(out: str) -> Tuple[int, int, int]:
    import re
    p = f = s = 0
    m = re.search(r"(\d+)\s+passed", out); p = int(m.group(1)) if m else 0
    m = re.search(r"(\d+)\s+failed", out); f = int(m.group(1)) if m else 0
    m = re.search(r"(\d+)\s+skipped", out); s = int(m.group(1)) if m else 0
    return p, f, p + f + s


def _parse_ruff_json(out: str) -> int:
    try:
        data = json.loads(out)
        if isinstance(data, list):
            return sum(len(it.get("diagnostics", []) or []) for it in data)
        if isinstance(data, dict):
            return len(data.get("diagnostics", []) or [])
    except Exception:
        pass
    return len([ln for ln in out.splitlines() if ln.strip()])


@dataclass
class ToolchainConfig:
    workspace: Path
    test_cmd: Optional[str] = None
    lint_cmd: Optional[str] = None
    build_cmd: Optional[str] = None
    timeout_sec: int = 180
    shell_timeout: int = 60
    shell_defaults: Dict[str, str] = field(default_factory=dict)


class ToolchainExecutor:
    def __init__(self, cfg: ToolchainConfig) -> None:
        self.cfg = cfg
        self._workspace_root = cfg.workspace.resolve()
        self._cwd = self._workspace_root

    def _is_within_workspace(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self._workspace_root)
            return True
        except Exception:
            return False

    def _resolve_within_workspace(self, base: Path, target: str) -> Path:
        candidate = Path(target or '.')
        resolved = (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if not self._is_within_workspace(resolved):
            raise ValueError(f'Path {resolved} outside workspace')
        return resolved

    def _shell_timeout(self, override: Optional[object]) -> int:
        try:
            return int(override or self.cfg.shell_timeout or max(30, self.cfg.timeout_sec // 2))
        except Exception:
            return self.cfg.shell_timeout or max(30, self.cfg.timeout_sec // 2)
    def _log_patch_attempt(self, workspace: Path, record: Dict[str, Any]) -> None:
        try:
            path = workspace / '.dspy_patches' / 'history.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception:
            pass

    def __call__(self, tool: ToolAction, args: Dict[str, Any]) -> AgentResult:
        ws = self.cfg.workspace; metrics: Dict[str, Any] = {"tool": tool.name.lower()}; info: Dict[str, Any] = {}
        # Default policy adherence score
        metrics.setdefault("quality_policy", 1.0)
        if tool == ToolAction.RUN_TESTS:
            cmd = args.get("cmd") or self.cfg.test_cmd or ("pytest -q" if (ws / "tests").exists() else None)
            if not cmd: return AgentResult(metrics={"pass_rate": 0.0, "tests_total": 0, "tests_passed": 0, "tests_failed": 0, **metrics})
            ok, out, dt = _run(cmd, ws, self.cfg.timeout_sec); p, f, tot = _parse_pytest(out)
            pr = (p / tot) if tot > 0 else (1.0 if ok else 0.0)
            metrics.update({"pass_rate": pr, "tests_total": tot, "tests_passed": p, "tests_failed": f}); info.update({"stdout": out[-4000:], "duration_sec": dt})
        elif tool == ToolAction.LINT:
            cmd = args.get("cmd") or self.cfg.lint_cmd or "ruff check --output-format json ."
            ok, out, dt = _run(cmd, ws, self.cfg.timeout_sec); issues = _parse_ruff_json(out)
            metrics.update({"lint_ok": bool(ok and issues == 0), "lint_issues": int(issues)}); info.update({"stdout": out[-4000:], "duration_sec": dt})
        elif tool == ToolAction.BUILD:
            # Prefer a workspace-local Python (venv/.venv) when available
            def _default_python(ws_path: Path) -> str:
                candidates = [
                    ws_path / 'venv' / 'bin' / 'python',
                    ws_path / '.venv' / 'bin' / 'python',
                ]
                for c in candidates:
                    try:
                        if c.exists():
                            return str(c)
                    except Exception:
                        continue
                # Fallback to current interpreter or discovered python3/python
                return sys.executable or shutil.which('python3') or shutil.which('python') or 'python'

            py = _default_python(ws)
            cmd = args.get("cmd") or self.cfg.build_cmd or f"{py} -m compileall -q ."
            ok, out, dt = _run(cmd, ws, self.cfg.timeout_sec); metrics.update({"build_ok": bool(ok)}); info.update({"stdout": out[-4000:], "duration_sec": dt})
        elif tool == ToolAction.PATCH:
            # End-to-end patch attempt: propose -> verify -> apply -> test -> revert.
            task = str(args.get("task", "Fix errors in application"))
            context = str(args.get("context", ""))
            max_files = int(args.get("max_files", 4)); max_lines = int(args.get("max_lines", 200))
            revert_always = bool(args.get("revert_always", True))
            prompt_hint = str(args.get("prompt", "") or "").strip()
            if prompt_hint:
                context = f"{context}\n\nPrompt guidance:\n{prompt_hint}".strip()
            prompt_id = args.get("prompt_id")
            if prompt_id:
                info.update({"prompt_id": str(prompt_id)})
            # Load quality checks from args or default file
            quality_cfg = args.get("quality_checks", {})
            if not quality_cfg:
                try:
                    import json as _json
                    qc_path = ws / '.dspy_quality.json'
                    if qc_path.exists():
                        content = _json.loads(qc_path.read_text())
                        if isinstance(content, dict):
                            quality_cfg = content
                except Exception:
                    pass
            quality_checks: Dict[str, str] = {}
            if isinstance(quality_cfg, Mapping):
                for key, value in quality_cfg.items():
                    val = str(value).strip()
                    if val:
                        quality_checks[str(key)] = val
            elif isinstance(quality_cfg, (list, tuple)):
                for idx, value in enumerate(quality_cfg):
                    val = str(value).strip()
                    if val:
                        quality_checks[f"check_{idx}"] = val
            start_ts = time.time()
            patch_text = ""
            summary: Dict[str, Any] = {}
            # Ensure 'hints' exists before any early returns that use _finalize
            hints = ""

            def _quality_report() -> Dict[str, Any]:
                q = {}
                for name in quality_checks:
                    key = f"quality_{name}"
                    if key in metrics:
                        q[name] = float(metrics.get(key, 0.0))
                return q

            def _finalize(status: str) -> AgentResult:
                quality_results = _quality_report()
                record = {
                    'timestamp': start_ts,
                    'task': task,
                    'prompt_hint': prompt_hint,
                    'prompt_id': info.get('prompt_id'),
                    'workspace': str(ws),
                    'context_excerpt': context[:2000],
                    'patch_preview': patch_text[:2000],
                    'result': status,
                    'metrics': dict(metrics),
                    'info': {k: info.get(k) for k in ('apply_message', 'reverted', 'revert_message', 'test_plan', 'quality', 'error', 'warn', 'skip', 'prompt_id') if k in info},
                    'quality_checks': quality_checks,
                    'quality_results': quality_results,
                    'file_candidates': hints,
                    'high_confidence': bool(metrics.get('applied') and float(metrics.get('pass_rate', 0.0)) >= 1.0 and all(v >= 1.0 for v in quality_results.values()) if quality_results else metrics.get('applied') and float(metrics.get('pass_rate', 0.0)) >= 1.0),
                    'runtime_sec': time.time() - start_ts,
                }
                try:
                    self._log_patch_attempt(ws, record)
                except Exception:
                    pass
                if "blast_radius" not in metrics:
                    metrics["blast_radius"] = 0.0
                return AgentResult(metrics=metrics, info=info)
            # If no context, skip heavy patch attempt
            if not context.strip():
                info.update({"skip": "empty context"}); metrics.update({"pass_rate": 0.0, "blast_radius": 0.0}); return _finalize('skipped')
            try:
                from ..skills.file_locator import FileLocator  # type: ignore
                from ..skills.code_edit import CodeEdit  # type: ignore
                from ..skills.patch_verifier import PatchVerifier  # type: ignore
                from ..skills.test_planner import TestPlanner  # type: ignore
                from ..code_tools.patcher import apply_unified_patch, revert_unified_patch, summarize_patch, run_shell
            except Exception as e:
                info.update({"error": f"patch deps unavailable: {e}"}); metrics.update({"pass_rate": 0.0, "blast_radius": 0.0}); return _finalize('error')
            # Locate files -> propose patch
            hint_list: List[str] = []
            try:
                loc = FileLocator(); loc_out = loc(task=task, context=context, code_graph=""); hints = getattr(loc_out, 'file_candidates', '') or ''
                info.update({"locator": getattr(loc_out, 'notes', '') or ''})
            except Exception:
                pass
            if isinstance(hints, str):
                hint_list = [h.strip() for h in hints.split(',') if h.strip()]
            elif isinstance(hints, (list, tuple)):
                hint_list = [str(h).strip() for h in hints if str(h).strip()]
            ce = CodeEdit(use_cot=True)
            try:
                pred = ce(task=task, context=context, code_graph="", file_hints=hints)
            except Exception as e:
                info.update({"error": f"code edit failed: {e}"}); metrics.update({"pass_rate": 0.0, "blast_radius": 0.0}); return _finalize('error')
            patch_text = getattr(pred, 'patch', '') or ''
            if not patch_text.strip():
                info.update({"error": "empty patch"}); metrics.update({"pass_rate": 0.0, "blast_radius": 0.0}); return _finalize('error')
            # Verify
            try:
                ver = PatchVerifier(max_files=max_files, max_lines=max_lines)
                v = ver(task=task, context=context, patch=patch_text)
                metrics.update({"verdict": getattr(v, 'verdict', 'fail')})
                if getattr(v, 'verdict', 'fail').lower() != 'pass':
                    summary = summarize_patch(patch_text); metrics.update({"blast_radius": float(summary.get('added_lines', 0) + summary.get('removed_lines', 0))}); info.update({"verify": getattr(v, 'reasons', '')}); return _finalize('failure')
            except Exception as e:
                info.update({"warn": f"verifier error: {e}"})
            # Apply
            ok, msg = apply_unified_patch(patch_text, ws)
            info.update({"apply_message": msg}); metrics.update({"applied": bool(ok)})
            if not ok:
                summary = summarize_patch(patch_text)
                metrics.update({"pass_rate": 0.0, "blast_radius": float(summary.get('added_lines', 0) + summary.get('removed_lines', 0))})
                return _finalize('failure')
            # Test selection
            test_cmd = args.get("test_cmd") or self.cfg.test_cmd
            if not test_cmd:
                try:
                    tp = TestPlanner(); tpo = tp(task=task, context=context, repo_layout=f"repo={ws.name}")
                    test_cmd = getattr(tpo, 'commands', '') or None
                    info.update({"test_plan": {
                        "tests_to_run": getattr(tpo, 'tests_to_run', ''),
                        "commands": test_cmd or '',
                        "fast_paths": getattr(tpo, 'fast_paths', ''),
                    }})
                except Exception:
                    test_cmd = None
            # Run tests if any
            pr = 0.0
            if test_cmd:
                code, out, dt = _run(test_cmd, ws, self.cfg.timeout_sec)
                p, f, tot = _parse_pytest(out); pr = (p / tot) if tot > 0 else (1.0 if code == 0 else 0.0)
                info.update({"stdout": out[-4000:], "duration_sec": dt, "test_cmd": test_cmd, "tests_total": tot, "tests_passed": p, "tests_failed": f})
            # Optional quality checks (lint/type, etc.)
            if quality_checks:
                qinfo: Dict[str, Any] = {}
                for name, command in quality_checks.items():
                    ok, out, dt = _run(command, ws, self.cfg.timeout_sec)
                    qinfo[name] = {
                        "ok": bool(ok),
                        "stdout": out[-4000:],
                        "duration_sec": dt,
                        "command": command,
                    }
                    metrics[f"quality_{name}"] = 1.0 if ok else 0.0
                    if not ok:
                        pr = 0.0
                        info['quality'] = qinfo
                        break
                else:
                    info['quality'] = qinfo
            # Summarize patch size (blast radius)
            summary = summarize_patch(patch_text)
            metrics.update({"pass_rate": float(pr), "blast_radius": float(summary.get('added_lines', 0) + summary.get('removed_lines', 0))})
            metrics.setdefault("retrieval_precision", float(pr))
            metrics.setdefault("retrieval_coverage", float(len(hint_list)))
            # Apply user preferences as quality gate (optional)
            try:
                from ..preferences import Preferences, check_patch_against_prefs
                prefs = Preferences.load(ws)
            except Exception:
                prefs = None  # type: ignore
            if prefs is not None:
                try:
                    violations = check_patch_against_prefs(patch_text, prefs)
                    if prefs.max_blast_radius and float(metrics.get("blast_radius", 0.0)) > float(prefs.max_blast_radius):
                        violations.append(f"blast_radius>{prefs.max_blast_radius}")
                    if violations:
                        info['preferences'] = {"violations": violations}
                        metrics["quality_preferences"] = 0.0
                        metrics["quality_policy"] = 0.0
                        # Conservative: mark as failed
                        pr = 0.0
                        metrics["pass_rate"] = 0.0
                except Exception:
                    pass
            # Revert to keep training environment consistent
            if revert_always:
                try:
                    ro, rmsg = revert_unified_patch(patch_text, ws)
                    info.update({"reverted": ro, "revert_message": rmsg})
                except Exception:
                    pass
            result_status = 'success' if float(metrics.get('pass_rate', 0.0)) >= 1.0 and metrics.get('applied') else 'failure'
            return _finalize(result_status)
        elif tool == ToolAction.SHELL_LS:
            cmd = args.get('cmd') or self.cfg.shell_defaults.get('shell_ls') or 'ls -lah'
            timeout = self._shell_timeout(args.get('timeout'))
            ok, out, dt = _run(str(cmd), self._cwd, timeout)
            metrics.update({
                'shell_exit_code': 0 if ok else 1,
                'shell_stdout_lines': len(out.splitlines()),
                'shell_cmd': str(cmd),
                'cwd': str(self._cwd),
            })
            info.update({'stdout': out[-4000:], 'duration_sec': dt})
        elif tool == ToolAction.SHELL_PWD:
            cmd = args.get('cmd') or self.cfg.shell_defaults.get('shell_pwd') or 'pwd'
            timeout = self._shell_timeout(args.get('timeout'))
            ok, out, dt = _run(str(cmd), self._cwd, timeout)
            metrics.update({
                'shell_exit_code': 0 if ok else 1,
                'shell_stdout': out.strip(),
                'shell_cmd': str(cmd),
                'cwd': str(self._cwd),
            })
            info.update({'stdout': out[-4000:], 'duration_sec': dt})
        elif tool == ToolAction.SHELL_CAT:
            target = args.get('path') or self.cfg.shell_defaults.get('shell_cat') or ''
            try:
                path = self._resolve_within_workspace(self._cwd, target) if target else None
                if path is None or not path.exists():
                    fallback = self._cwd / 'README.md'
                    if fallback.exists():
                        path = fallback.resolve()
                    else:
                        for candidate in self._cwd.iterdir():
                            if candidate.is_file():
                                path = candidate.resolve()
                                break
                if path is None:
                    raise FileNotFoundError('no file available for shell_cat')
                cmd = args.get('cmd') or self.cfg.shell_defaults.get('shell_cat_cmd') or f"head -n 200 {shlex.quote(str(path))}"
                timeout = self._shell_timeout(args.get('timeout'))
                ok, out, dt = _run(str(cmd), self._cwd, timeout)
                metrics.update({
                    'shell_exit_code': 0 if ok else 1,
                    'shell_bytes': len(out.encode('utf-8', errors='ignore')),
                    'shell_cmd': str(cmd),
                    'shell_target': str(path),
                    'cwd': str(self._cwd),
                })
                info.update({'stdout': out[-4000:], 'duration_sec': dt, 'target': str(path)})
            except Exception as exc:
                metrics.update({'shell_exit_code': 1, 'shell_cmd': 'cat', 'shell_error': str(exc), 'cwd': str(self._cwd)})
                info.update({'error': str(exc)})
        elif tool == ToolAction.SHELL_CD:
            target = args.get('path') or self.cfg.shell_defaults.get('shell_cd') or '.'
            try:
                new_cwd = self._resolve_within_workspace(self._cwd, target)
                if not new_cwd.is_dir():
                    raise NotADirectoryError(str(new_cwd))
                self._cwd = new_cwd
                metrics.update({'shell_cd_ok': 1.0, 'cwd': str(self._cwd)})
                info.update({'cwd': str(self._cwd)})
            except Exception as exc:
                metrics.update({'shell_cd_ok': 0.0, 'shell_error': str(exc), 'cwd': str(self._cwd)})
                info.update({'error': str(exc)})
        elif tool == ToolAction.SHELL_RUN:
            cmd = args.get('cmd') or self.cfg.shell_defaults.get('shell_run')
            if not cmd:
                metrics.update({'shell_exit_code': 1, 'shell_error': 'no command provided', 'cwd': str(self._cwd)})
                info.update({'error': 'no command provided'})
            else:
                timeout = self._shell_timeout(args.get('timeout'))
                ok, out, dt = _run(str(cmd), self._cwd, timeout)
                metrics.update({
                    'shell_exit_code': 0 if ok else 1,
                    'shell_stdout_lines': len(out.splitlines()),
                    'shell_cmd': str(cmd),
                    'cwd': str(self._cwd),
                })
                info.update({'stdout': out[-4000:], 'duration_sec': dt})
        else:
            info.update({"warning": f"unhandled tool: {tool}"})
        if "blast_radius" not in metrics: metrics["blast_radius"] = 0.0
        return AgentResult(metrics=metrics, info=info)


def detect_toolchain(
    workspace: Path,
    *,
    test_cmd: Optional[str] = None,
    lint_cmd: Optional[str] = None,
    build_cmd: Optional[str] = None,
    timeout_sec: Optional[int] = None,
    shell_timeout: Optional[int] = None,
    shell_defaults: Optional[Mapping[str, str]] = None,
) -> ToolchainConfig:
    ws = workspace.resolve()
    test_cmd = test_cmd or None
    lint_cmd = lint_cmd or None
    build_cmd = build_cmd or None
    if not test_cmd and (ws / "tests").exists(): test_cmd = "pytest -q"
    if not lint_cmd: lint_cmd = "ruff check --output-format json ."
    if not build_cmd:
        # Prefer local venv python if present
        def _pick_python(ws_path: Path) -> str:
            for p in [ws_path / 'venv' / 'bin' / 'python', ws_path / '.venv' / 'bin' / 'python']:
                try:
                    if p.exists():
                        return str(p)
                except Exception:
                    continue
            return sys.executable or shutil.which('python3') or shutil.which('python') or 'python'
        build_cmd = f"{_pick_python(ws)} -m compileall -q ."
    env_shell_timeout = os.getenv('RL_SHELL_TIMEOUT')
    timeout_val = int(timeout_sec or 180)
    if shell_timeout is None:
        try:
            shell_timeout = int(env_shell_timeout) if env_shell_timeout else max(30, timeout_val // 2)
        except Exception:
            shell_timeout = max(30, timeout_val // 2)
    defaults: Dict[str, str] = {}
    if shell_defaults:
        defaults.update({str(k): str(v) for k, v in shell_defaults.items()})
    for key in ('shell_ls', 'shell_pwd', 'shell_cat', 'shell_cat_cmd', 'shell_cd', 'shell_run'):
        env_key = f'RL_{key.upper()}'
        val = os.getenv(env_key)
        if val:
            defaults[key] = val
    return ToolchainConfig(
        workspace=ws,
        test_cmd=test_cmd,
        lint_cmd=lint_cmd,
        build_cmd=build_cmd,
        timeout_sec=timeout_val,
        shell_timeout=int(shell_timeout),
        shell_defaults=defaults,
    )


# ---------------
# Verifiers loader + sample verifiers
# ---------------

def load_from_module(module_path: str) -> List[VerifierProtocol]:
    import importlib
    mod = importlib.import_module(module_path)
    if hasattr(mod, "get_verifiers"):  # type: ignore[attr-defined]
        v = mod.get_verifiers()  # type: ignore[attr-defined]
        return list(v)
    found: List[VerifierProtocol] = []
    for name in dir(mod):
        obj = getattr(mod, name)
        try:
            if hasattr(obj, "kind") and callable(getattr(obj, "__call__", None)) and not isinstance(obj, type):
                found.append(obj)  # type: ignore[arg-type]
        except Exception:
            pass
    if found:
        return found
    for name in dir(mod):
        obj = getattr(mod, name)
        try:
            if isinstance(obj, type) and hasattr(obj, "kind"):
                inst = obj()  # type: ignore[call-arg]
                if hasattr(inst, "kind") and callable(getattr(inst, "__call__", None)):
                    found.append(inst)  # type: ignore[arg-type]
        except Exception:
            pass
    return found


class PassRateVerifier:
    kind = "pass_rate"
    def __call__(self, result: AgentResult) -> float:
        pr = result.metrics.get("pass_rate")
        if pr is not None: return float(pr)
        tot = int(result.metrics.get("tests_total", 0) or 0); ok = int(result.metrics.get("tests_passed", 0) or 0)
        return (ok / tot) if tot > 0 else 0.0


class BlastRadiusVerifier:
    kind = "blast_radius"
    def __call__(self, result: AgentResult) -> float:
        return float(result.metrics.get("blast_radius", 0.0))


def sample_get_verifiers():
    class BuildOkVerifier:
        kind = "build_ok"
        def __call__(self, result: AgentResult) -> float:
            return 1.0 if bool(result.metrics.get("build_ok")) else 0.0
    class LintOkVerifier:
        kind = "lint_ok"
        def __call__(self, result: AgentResult) -> float:
            ok = result.metrics.get("lint_ok")
            if ok is not None: return 1.0 if bool(ok) else 0.0
            issues = int(result.metrics.get("lint_issues", 0) or 0)
            return 1.0 if issues == 0 else 0.0
    class QualityAvgVerifier:
        kind = "quality_avg"
        def __call__(self, result: AgentResult) -> float:
            keys = [k for k in result.metrics.keys() if str(k).startswith('quality_')]
            if not keys: return 1.0
            vals = [float(result.metrics.get(k, 0.0) or 0.0) for k in keys]
            return sum(vals) / max(1, len(vals))
    return [PassRateVerifier(), BlastRadiusVerifier(), BuildOkVerifier(), LintOkVerifier(), QualityAvgVerifier()]


def get_verifiers():
    # Alias expected by loader
    return sample_get_verifiers()


# ---------------
# RL Config IO (JSON)
# ---------------

@dataclass
class RLConfig:
    policy: str = "epsilon-greedy"
    epsilon: float = 0.1
    ucb_c: float = 2.0
    n_envs: int = 2
    puffer: bool = False
    temperature: float = 0.85
    target_entropy: float = 0.3
    clip_higher: float = 1.1
    verifiers_module: Optional[str] = None
    actions: Optional[List[str]] = None
    weights: Dict[str, float] = field(default_factory=dict)
    penalty_kinds: List[str] = field(default_factory=list)
    clamp01_kinds: List[str] = field(default_factory=list)
    scales: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    test_cmd: Optional[str] = None
    lint_cmd: Optional[str] = None
    build_cmd: Optional[str] = None
    timeout_sec: Optional[int] = None


def load_rl_config(path: Path) -> RLConfig:
    data = json.loads(Path(path).read_text())
    cfg = RLConfig()
    for k, v in dict(data).items():
        if hasattr(cfg, k): setattr(cfg, k, v)
    if isinstance(cfg.scales, dict):
        cfg.scales = {str(k): (float(v[0]), float(v[1])) for k, v in cfg.scales.items() if isinstance(v, (list, tuple)) and len(v) == 2}
    if cfg.actions:
        if isinstance(cfg.actions, str):
            raw = cfg.actions.split(',')
        elif isinstance(cfg.actions, (list, tuple, set)):
            raw = list(cfg.actions)
        else:
            raw = [cfg.actions]
        cleaned = [str(item).strip() for item in raw if str(item).strip()]
        cfg.actions = cleaned or None
    return cfg


# ---------------
# PuffeRL PPO shell (sketch)
# ---------------

def run_puffer_ppo(make_env: Callable[[], RLToolEnv], n_envs: int = 8, total_steps: int = 100_000):  # pragma: no cover - optional
    try:
        try:
            import pufferlib.emulation as emulation  # type: ignore
            import pufferlib.vector as pvector  # type: ignore
        except ImportError:
            raise ImportError("PufferLib not available. Install with: pip install pufferlib>=3.0.0")
        try:
            import pufferlib.pufferl as ppo  # type: ignore
        except ImportError:
            raise ImportError("PufferLib not available. Install with: pip install pufferlib>=3.0.0")
        import torch
        import torch.nn as nn
    except Exception as e:
        raise RuntimeError("PufferLib not available. Install with 'pip install pufferlib --no-build-isolation' or '.[rl]'") from e
    def creator(): return emulation.GymnasiumPufferEnv(make_env())
    vecenv = pvector.make(creator, num_envs=max(1, n_envs), backend='multiprocessing')
    def make_policy(obs_space, act_space):
        in_dim = int(obs_space.shape[0]); out_dim = int(getattr(act_space, 'n', 2))
        body = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        policy_head = nn.Linear(128, out_dim); value_head = nn.Linear(128, 1)
        class AC(nn.Module):
            def __init__(self): super().__init__(); self.body=body; self.pi=policy_head; self.v=value_head
            def forward(self, x): h=self.body(x); return self.pi(h), self.v(h)
        return AC()
    try:
        runner = ppo.PuffeRL(config={'total_steps': int(total_steps), 'batch_size': 2048, 'rollout_length': 128, 'learning_rate': 3e-4, 'entropy_coef': 0.01, 'value_coef': 0.5}, vecenv=vecenv, policy_factory=make_policy)  # type: ignore[attr-defined]
    except TypeError:
        runner = ppo.PuffeRL({'total_steps': int(total_steps)}, vecenv, make_policy)  # type: ignore
    runner.train()  # type: no cover
    try: vecenv.close()
    except Exception: pass


__all__ = [
    # env
    'ToolAction','AgentResult','VerifierProtocol','EnvConfig','RLToolEnv',
    # reward
    'RewardConfig','aggregate_reward',
    # bandits
    'BaseBandit','EpsilonGreedy','UCB1','ThompsonBeta','make_bandit',
    # runner & trainer
    'EpisodeStats','VectorRunner','TrainerConfig','bandit_trainer','bandit_trainer_puffer','train_puffer_policy',
    # executor
    'ToolchainConfig','ToolchainExecutor','detect_toolchain',
    # verifiers loader + sample
    'load_from_module','PassRateVerifier','BlastRadiusVerifier','sample_get_verifiers','get_verifiers',
    # config
    'RLConfig','load_rl_config',
    # puffer shell
    'run_puffer_ppo',
]
def record_verifier_scores(signature_name: str, scores: Mapping[str, float], reward: float, *, environment: str = "development", doc_id: Optional[str] = None, action_type: str = "VERIFICATION", execution_time: float = 0.0, query: Optional[str] = None) -> str:
    """Persist an ActionRecord with per-verifier scores for analytics.

    Convenience helper you can call inside training/verification loops.
    """
    try:
        from dspy_agent.db import get_enhanced_data_manager, create_action_record, Environment, ActionType
        dm = get_enhanced_data_manager()
        try:
            env = getattr(Environment, environment.strip().upper())
        except Exception:
            env = Environment.DEVELOPMENT
        try:
            at = ActionType[action_type.strip().upper()]
        except Exception:
            at = ActionType.VERIFICATION
        rec = create_action_record(
            action_type=at,
            state_before={'signature_name': signature_name},
            state_after={'signature_name': signature_name},
            parameters={'signature_name': signature_name, 'verifier_scores': dict(scores), **({'doc_id': doc_id} if doc_id else {}), **({'query': query} if query else {})},
            result={'signature_name': signature_name, 'verifier_scores': dict(scores)},
            reward=float(reward),
            confidence=0.95,
            execution_time=float(execution_time),
            environment=env,
        )
        dm.record_action(rec)
        return rec.action_id
    except Exception:
        logger.exception("failed to record verifier scores")
        return ""


def make_default_analytics_cb(
    signature_name: Optional[str] = None,
    *,
    env: str = "development",
    action_type: str = "VERIFICATION",
    doc_fn: Optional[Callable[[int, Dict[str, Any]], Optional[str]]] = None,
    query_fn: Optional[Callable[[int, Dict[str, Any]], Optional[str]]] = None,
) -> Callable[[int, Dict[str, Any], float], Optional[Dict[str, Any]]]:
    """Create a default analytics callback for VectorRunner/Puffer that extracts metadata
    from the info dict and persists verifier scores using record_verifier_scores.

    - signature_name: static name or None to read from info['signature_name']
    - env/action_type: defaults for ActionRecord
    - doc_fn: optional function to derive doc_id from (env_idx, info); defaults to info.get('doc_id')
    - query_fn: optional function to derive query text from (env_idx, info)
    """
    def cb(env_idx: int, info: Dict[str, Any], reward: float) -> Optional[Dict[str, Any]]:
        try:
            sig = signature_name or (isinstance(info.get('signature_name'), str) and info.get('signature_name')) or None
            if not sig:
                # also check nested dictionaries
                for k in ('parameters', 'result', 'state', 'meta'):
                    v = info.get(k)
                    if isinstance(v, dict) and isinstance(v.get('signature_name'), str):
                        sig = v.get('signature_name'); break
            if not sig:
                return None
            did = None
            if doc_fn:
                try: did = doc_fn(env_idx, info)
                except Exception: did = None
            if not did:
                did = info.get('doc_id') if isinstance(info.get('doc_id'), str) else None
            q = None
            if query_fn:
                try: q = query_fn(env_idx, info)
                except Exception: q = None
            res = {
                'signature_name': sig,
                'doc_id': did,
                'environment': env,
                'action_type': action_type,
                'execution_time': float(info.get('execution_time') or 0.0),
            }
            if q: res['query'] = q
            return res
        except Exception:
            return None
    return cb
