from __future__ import annotations

# Consolidated RL toolkit: env, verifiers adapter, bandits, runner, trainer,
# executor, verifiers loader, RL config, and PuffeRL shell.
#
# This single module replaces prior small files to keep the codebase compact
# while preserving functionality. Public symbols are re-exported via
# dspy_agent.rl.__init__ for stable imports.

import json
import math
import random
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple


# ----------------
# RL Environment
# ----------------

class ToolAction(IntEnum):
    RUN_TESTS = 0
    LINT = 1
    BUILD = 2

    @classmethod
    def names(cls) -> List[str]:
        return ["run_tests", "lint", "build"]


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


try:  # optional gymnasium for PufferLib
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
        aa: Dict[int, Dict[str, Any]] = {}
        if cfg.action_args:
            for idx, name in enumerate(ToolAction.names()):
                if name in cfg.action_args:
                    aa[idx] = dict(cfg.action_args[name])
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
            self.action_space = _spaces.Discrete(len(ToolAction))  # type: ignore[attr-defined]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[List[float], Dict[str, Any]]:
        self._t = 0
        ctx = self._cfg.context_provider() if self._cfg.context_provider else []
        self._last_obs = list(ctx)
        return self._last_obs, {"t": self._t}

    def step(self, action: int) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
        tool = ToolAction(action)
        args = self._action_args.get(int(action), {})
        result = self._exec(tool, args)
        reward, vvec, details = self._cfg.reward_fn(result, self._cfg.verifiers, self._cfg.weights)
        ctx = self._cfg.context_provider() if self._cfg.context_provider else []
        obs = vvec + ctx
        self._last_obs = obs
        self._t += 1
        terminated = self._t >= self._episode_len
        truncated = False
        info = {"tool": tool.name.lower(), "verifier_scores": details}
        return obs, float(reward), bool(terminated), bool(truncated), info

    @property
    def action_dim(self) -> int:
        return len(ToolAction)

    @property
    def obs_size_hint(self) -> int:
        base = len(list(self._cfg.verifiers))
        ctx = len(self._cfg.context_provider() or []) if self._cfg.context_provider else 0
        return base + ctx


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
    def __init__(self, envs: List[Any], policy: Any) -> None:
        self.envs = envs; self.policy = policy
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
                if done or trunc: obs[i], _ = e.reset()
        return EpisodeStats(rewards=all_rewards, infos=all_infos)


# ---------------
# Trainer (bandit + neural)
# ---------------

@dataclass
class TrainerConfig:
    steps: int = 100
    policy: str = "epsilon-greedy"
    policy_kwargs: Mapping[str, Any] = field(default_factory=dict)
    n_envs: int = 1


def bandit_trainer(make_env: Callable[[], RLToolEnv], cfg: TrainerConfig) -> EpisodeStats:
    envs = [make_env() for _ in range(max(1, cfg.n_envs))]
    policy = make_bandit(cfg.policy, envs[0].action_dim, **dict(cfg.policy_kwargs))
    return VectorRunner(envs, policy).run(cfg.steps)


def bandit_trainer_puffer(make_env: Callable[[], RLToolEnv], cfg: TrainerConfig, backend: str = "multiprocessing") -> EpisodeStats:
    try:
        import pufferlib.emulation as emulation  # type: ignore
        import pufferlib.vector as pvector  # type: ignore
    except Exception as e:  # pragma: no cover - optional
        raise RuntimeError("PufferLib not available. Install with 'pip install .[rl]'") from e
    tmp_env = make_env(); n_actions = tmp_env.action_dim
    def creator(): return emulation.GymnasiumPufferEnv(make_env())
    venv = pvector.make(creator, num_envs=max(1, cfg.n_envs), backend=backend)
    obs, infos = venv.reset(); policy = make_bandit(cfg.policy, n_actions, **dict(cfg.policy_kwargs))
    all_rewards: List[float] = []; all_infos: List[dict] = []
    for _ in range(cfg.steps):
        actions = [int(policy.select(list(obs[i]) if not isinstance(obs[i], list) else obs[i])) for i in range(len(obs))]
        obs, rewards, terms, truncs, infos = venv.step(actions)
        for i, a in enumerate(actions):
            r = float(rewards[i]); policy.update(a, r, list(obs[i]) if not isinstance(obs[i], list) else obs[i])
            all_rewards.append(r); all_infos.append(infos[i] if isinstance(infos, list) else infos)
        obs, infos = venv.reset()
    try: venv.close()
    except Exception: pass
    return EpisodeStats(rewards=all_rewards, infos=all_infos)


def train_puffer_policy(*, make_env: Callable[[], RLToolEnv], steps: int = 1000, n_envs: int = 4, lr: float = 1e-3, seed: Optional[int] = None, verbose: bool = False, log_interval: int = 10) -> EpisodeStats:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception as e:  # pragma: no cover - optional
        raise RuntimeError("Neural training requires torch. Install with 'pip install .[rl]'") from e
    # Try PufferLib vectorized env for speed
    using_vec = False
    try:
        import pufferlib.emulation as emulation  # type: ignore
        import pufferlib.vector as pvector  # type: ignore
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
        for step_idx in range(1, steps + 1):
            obs, _ = venv.reset()
            import torch
            obs_t = torch.tensor(obs, dtype=torch.float32)
            logits = policy(obs_t)
            m = torch.distributions.Categorical(logits=logits)
            actions = m.sample(); logp = m.log_prob(actions)
            obs2, rewards, terms, truncs, infos = venv.step(actions.tolist())
            r_t = torch.tensor(rewards, dtype=torch.float32)
            avg_r = float(r_t.mean().item()); running_mean = beta * running_mean + (1 - beta) * avg_r
            adv = r_t - running_mean
            loss = -(logp * adv).mean(); opt.zero_grad(); loss.backward(); opt.step()
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
            for i, e in enumerate(envs):
                a, lp = select_action(obses[i]); o2, r, done, trunc, info = e.step(a)
                rewards_all.append(float(r)); infos_all.append(info)
                obses[i] = o2 if not (done or trunc) else e.reset()[0]
                batch_lp.append(lp); batch_r.append(float(r))
            avg_r = sum(batch_r) / max(1, len(batch_r)); running_mean = beta * running_mean + (1 - beta) * avg_r
            loss = -sum(lp * (r - running_mean) for lp, r in zip(batch_lp, batch_r)) / max(1, len(batch_lp))
            opt.zero_grad(); loss.backward(); opt.step()
            if verbose and (step_idx % max(1, int(log_interval)) == 0):
                print(f"[rl] step={step_idx} avg_r={avg_r:.3f}")
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


class ToolchainExecutor:
    def __init__(self, cfg: ToolchainConfig) -> None:
        self.cfg = cfg
    def __call__(self, tool: ToolAction, args: Dict[str, Any]) -> AgentResult:
        ws = self.cfg.workspace; metrics: Dict[str, Any] = {"tool": tool.name.lower()}; info: Dict[str, Any] = {}
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
            cmd = args.get("cmd") or self.cfg.build_cmd or "python -m compileall -q ."
            ok, out, dt = _run(cmd, ws, self.cfg.timeout_sec); metrics.update({"build_ok": bool(ok)}); info.update({"stdout": out[-4000:], "duration_sec": dt})
        else:
            info.update({"warning": f"unhandled tool: {tool}"})
        if "blast_radius" not in metrics: metrics["blast_radius"] = 0.0
        return AgentResult(metrics=metrics, info=info)


def detect_toolchain(workspace: Path, *, test_cmd: Optional[str] = None, lint_cmd: Optional[str] = None, build_cmd: Optional[str] = None, timeout_sec: Optional[int] = None) -> ToolchainConfig:
    ws = workspace.resolve()
    test_cmd = test_cmd or None
    lint_cmd = lint_cmd or None
    build_cmd = build_cmd or None
    if not test_cmd and (ws / "tests").exists(): test_cmd = "pytest -q"
    if not lint_cmd: lint_cmd = "ruff check --output-format json ."
    if not build_cmd: build_cmd = "python -m compileall -q ."
    return ToolchainConfig(workspace=ws, test_cmd=test_cmd, lint_cmd=lint_cmd, build_cmd=build_cmd, timeout_sec=int(timeout_sec or 180))


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
    return [PassRateVerifier(), BlastRadiusVerifier()]


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
    verifiers_module: Optional[str] = None
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
    return cfg


# ---------------
# PuffeRL PPO shell (sketch)
# ---------------

def run_puffer_ppo(make_env: Callable[[], RLToolEnv], n_envs: int = 8, total_steps: int = 100_000):  # pragma: no cover - optional
    try:
        import pufferlib.emulation as emulation  # type: ignore
        import pufferlib.vector as pvector  # type: ignore
        import pufferlib.pufferl as ppo  # type: ignore
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
