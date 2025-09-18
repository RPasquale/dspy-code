"""Hyperparameter sweep orchestration for the DSPy RL toolchain.

This module wires the vendored PufferLib sweep strategies into a loop that
builds RL environments, evaluates candidate configurations, and persists the
best-performing setup per repository.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from ..rl.rlkit import (
    RLConfig,
    RLToolEnv,
    EnvConfig,
    RewardConfig,
    ToolAction,
    ToolchainExecutor,
    detect_toolchain,
    aggregate_reward,
    load_from_module,
    get_verifiers as _default_verifiers,
    TrainerConfig,
    bandit_trainer,
    bandit_trainer_puffer,
)
from ..rl.puffer_sweep import Hyperparameters, get_strategy
from ..rl import hparam_guide


@dataclass
class SweepSettings:
    method: str = "pareto"
    metric: str = "reward"
    goal: str = "maximize"
    iterations: int = 20
    trainer_steps: Optional[int] = None
    trainer_backend: str = "bandit"
    puffer_backend: bool = False
    persist_path: Optional[Path] = None


@dataclass
class TrialSummary:
    params: Dict[str, Any]
    metric: float
    cost: float
    avg_reward: float
    avg_pass_rate: float
    avg_blast_radius: float
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepOutcome:
    best_config: RLConfig
    best_summary: TrialSummary
    history: List[TrialSummary]


DEFAULT_CONFIG_PATH = Path(__file__).with_name('rl_default_sweep.json')
DEFAULT_PERSIST_PATH = Path('.dspy') / 'rl' / 'best.json'


def load_sweep_config(path: Optional[Path] = None) -> Dict[str, Any]:
    candidate_paths: Iterable[Path] = []
    if path:
        candidate_paths = (path,)
    else:
        candidate_paths = (DEFAULT_CONFIG_PATH,)
    for candidate in candidate_paths:
        try:
            data = json.loads(candidate.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    raise FileNotFoundError(f"Sweep configuration not found (looked at {list(candidate_paths)})")


def describe_default_hparams() -> List[Mapping[str, object]]:
    """Return the curated hyperparameter guide as JSON-compatible structures."""

    return hparam_guide.as_dict()


def _build_env_factory(
    workspace: Path,
    *,
    cfg: RLConfig,
    weights: Mapping[str, float],
    penalty_kinds: Iterable[str],
    clamp01_kinds: Iterable[str],
    scales: Mapping[str, Tuple[float, float]],
    actions: Optional[Iterable[str]],
    test_cmd: Optional[str],
    lint_cmd: Optional[str],
    build_cmd: Optional[str],
    timeout_sec: Optional[int],
    verifiers_module: Optional[str],
) -> Tuple[callable, List[str]]:
    try:
        verifiers = load_from_module(verifiers_module) if verifiers_module else None
    except Exception:
        verifiers = None
    if not verifiers:
        verifiers = _default_verifiers()
    rc = RewardConfig(
        weights=dict(weights),
        penalty_kinds=list(penalty_kinds or []),
        clamp01_kinds=list(clamp01_kinds or []),
        scales=dict(scales or {}),
    )

    allowed = None
    if actions:
        allowed = [str(x).strip() for x in actions if str(x).strip()]

    def reward_fn(result, verifiers_list, weights_map):  # type: ignore[no-redef]
        total, vec, details = aggregate_reward(result, verifiers_list, rc)
        return total, vec, details

    def make_env() -> RLToolEnv:
        tcfg = detect_toolchain(
            workspace,
            test_cmd=test_cmd or cfg.test_cmd,
            lint_cmd=lint_cmd or cfg.lint_cmd,
            build_cmd=build_cmd or cfg.build_cmd,
            timeout_sec=timeout_sec or cfg.timeout_sec or 180,
        )
        executor = ToolchainExecutor(tcfg)

        def exec_fn(action: ToolAction, args: Dict[str, Any]) -> Any:
            return executor(action, args)

        env_cfg = EnvConfig(
            verifiers=verifiers,
            reward_fn=reward_fn,
            weights=rc.weights,
            action_args=None,
            allowed_actions=allowed,
        )
        return RLToolEnv(executor=exec_fn, cfg=env_cfg, episode_len=1)

    return make_env, allowed or []


def _metric_from_stats(metric: str, goal: str, stats: TrialSummary) -> float:
    value = stats.metric
    if metric == "pass_rate":
        value = stats.avg_pass_rate
    elif metric == "blast_radius":
        value = stats.avg_blast_radius
    elif metric == "reward":
        value = stats.avg_reward
    if goal == "minimize":
        return -value
    return value


def _summary_from_episode(metric_name: str, params: Dict[str, Any], rewards: List[float], infos: List[Dict[str, Any]], cost: float) -> TrialSummary:
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    pass_rates: List[float] = []
    blast_values: List[float] = []
    for info in infos:
        if not isinstance(info, Mapping):
            continue
        scores = info.get("verifier_scores")
        if isinstance(scores, Mapping):
            pr = scores.get("pass_rate")
            if pr is not None:
                try:
                    pass_rates.append(float(pr))
                except (TypeError, ValueError):
                    pass
            br = scores.get("blast_radius")
            if br is not None:
                try:
                    blast_values.append(float(br))
                except (TypeError, ValueError):
                    pass
    avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
    avg_blast = sum(blast_values) / len(blast_values) if blast_values else 0.0
    metric_value = avg_reward if metric_name == "reward" else (avg_pass_rate if metric_name == "pass_rate" else avg_blast)
    return TrialSummary(
        params=params,
        metric=metric_value,
        cost=float(cost),
        avg_reward=avg_reward,
        avg_pass_rate=avg_pass_rate,
        avg_blast_radius=avg_blast,
        info={},
    )


def _merge_config(base: RLConfig, suggestion: Mapping[str, Any]) -> Tuple[RLConfig, Dict[str, Any], Dict[str, Any]]:
    cfg = RLConfig(**vars(base))
    params_record: Dict[str, Any] = {}
    weights = dict(cfg.weights or {"pass_rate": 1.0, "blast_radius": 1.0})
    suggestion_weights = suggestion.get("weights")
    if isinstance(suggestion_weights, Mapping):
        for key, value in suggestion_weights.items():
            try:
                weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    params_record["weights"] = dict(weights)

    epsilon = suggestion.get("epsilon")
    if epsilon is not None:
        try:
            cfg.epsilon = float(epsilon)
            params_record["epsilon"] = cfg.epsilon
        except (TypeError, ValueError):
            pass
    ucb_c = suggestion.get("ucb_c")
    if ucb_c is not None:
        try:
            cfg.ucb_c = float(ucb_c)
            params_record["ucb_c"] = cfg.ucb_c
        except (TypeError, ValueError):
            pass
    timeout = suggestion.get("timeout_sec")
    if timeout is not None:
        try:
            cfg.timeout_sec = int(timeout)
            params_record["timeout_sec"] = cfg.timeout_sec
        except (TypeError, ValueError):
            pass
    temperature = suggestion.get("temperature")
    if temperature is not None:
        try:
            cfg.temperature = float(temperature)
            params_record["temperature"] = cfg.temperature
        except (TypeError, ValueError):
            pass
    target_entropy = suggestion.get("target_entropy")
    if target_entropy is not None:
        try:
            cfg.target_entropy = float(target_entropy)
            params_record["target_entropy"] = cfg.target_entropy
        except (TypeError, ValueError):
            pass
    clip_higher = suggestion.get("clip_higher")
    if clip_higher is not None:
        try:
            cfg.clip_higher = float(clip_higher)
            params_record["clip_higher"] = cfg.clip_higher
        except (TypeError, ValueError):
            pass
    actions = suggestion.get("actions")
    if isinstance(actions, Iterable) and not isinstance(actions, (str, bytes)):
        cfg.actions = [str(a).strip() for a in actions if str(a).strip()]
        params_record["actions"] = cfg.actions

    trainer = suggestion.get("trainer") if isinstance(suggestion, Mapping) else None
    trainer_params: Dict[str, Any] = {}
    if isinstance(trainer, Mapping):
        steps = trainer.get("steps")
        if steps is not None:
            try:
                trainer_params["steps"] = int(steps)
            except (TypeError, ValueError):
                pass
        n_envs = trainer.get("n_envs")
        if n_envs is not None:
            try:
                cfg.n_envs = max(1, int(n_envs))
                trainer_params["n_envs"] = cfg.n_envs
            except (TypeError, ValueError):
                pass
    params_record["trainer"] = trainer_params

    other = {}
    for key in ("verifiers_module", "test_cmd", "lint_cmd", "build_cmd"):
        value = suggestion.get(key)
        if value is not None:
            setattr(cfg, key, value)
            other[key] = value
    if other:
        params_record.update(other)

    return cfg, params_record, weights


def run_sweep(
    workspace: Path,
    sweep_config: Mapping[str, Any],
    *,
    base_config: Optional[RLConfig] = None,
    settings: Optional[SweepSettings] = None,
) -> SweepOutcome:
    if settings is None:
        settings = SweepSettings()
    method = str(sweep_config.get("method", settings.method)).strip() or settings.method
    metric = str(sweep_config.get("metric", settings.metric)).strip() or settings.metric
    goal = str(sweep_config.get("goal", settings.goal)).strip() or settings.goal
    iterations = int(sweep_config.get("iterations", settings.iterations))
    trainer_steps_override = settings.trainer_steps or sweep_config.get("trainer_steps")
    strategy_cls = get_strategy(method)
    hyper = Hyperparameters(sweep_config)
    strategy = strategy_cls(sweep_config)

    base = base_config or RLConfig()
    persist_path = settings.persist_path or DEFAULT_PERSIST_PATH
    history: List[TrialSummary] = []
    best_summary: Optional[TrialSummary] = None
    best_config: Optional[RLConfig] = None

    fill_template: MutableMapping[str, Any] = {
        "epsilon": base.epsilon,
        "ucb_c": base.ucb_c,
        "timeout_sec": base.timeout_sec or 180,
        "weights": dict(base.weights or {"pass_rate": 1.0, "blast_radius": 1.0}),
        "temperature": getattr(base, 'temperature', 0.85),
        "target_entropy": getattr(base, 'target_entropy', 0.3),
        "clip_higher": getattr(base, 'clip_higher', 1.1),
        "trainer": {"steps": trainer_steps_override or 200, "n_envs": base.n_envs or 1},
    }
    if base.actions:
        fill_template["actions"] = list(base.actions)

    for idx in range(1, iterations + 1):
        suggestion, extra = strategy.suggest(fill=fill_template)
        cfg, params_record, weights = _merge_config(base, suggestion)
        trainer_steps = trainer_steps_override or params_record.get("trainer", {}).get("steps") or 200
        policy_kwargs: Dict[str, Any] = {}
        policy_name = (cfg.policy or "epsilon-greedy").strip().lower()
        if policy_name in {"epsilon", "epsilon-greedy", "egreedy", "eps"}:
            policy_kwargs["epsilon"] = cfg.epsilon
        elif policy_name in {"ucb", "ucb1"}:
            policy_kwargs["c"] = cfg.ucb_c
        start = time.time()
        make_env, allowed = _build_env_factory(
            workspace,
            cfg=cfg,
            weights=weights,
            penalty_kinds=cfg.penalty_kinds or [],
            clamp01_kinds=cfg.clamp01_kinds or [],
            scales=cfg.scales or {},
            actions=cfg.actions,
            test_cmd=params_record.get("test_cmd") or cfg.test_cmd,
            lint_cmd=params_record.get("lint_cmd") or cfg.lint_cmd,
            build_cmd=params_record.get("build_cmd") or cfg.build_cmd,
            timeout_sec=params_record.get("timeout_sec") or cfg.timeout_sec,
            verifiers_module=cfg.verifiers_module,
        )
        trainer_cfg = TrainerConfig(steps=int(trainer_steps), policy=cfg.policy, policy_kwargs=policy_kwargs, n_envs=cfg.n_envs or 1)
        try:
            if settings.puffer_backend or sweep_config.get("puffer", False):
                stats = bandit_trainer_puffer(make_env, trainer_cfg)
            else:
                stats = bandit_trainer(make_env, trainer_cfg)
        except Exception as exc:
            duration = time.time() - start
            summary = TrialSummary(
                params=params_record,
                metric=float('-inf'),
                cost=duration,
                avg_reward=0.0,
                avg_pass_rate=0.0,
                avg_blast_radius=0.0,
                info={"error": str(exc)},
            )
            history.append(summary)
            strategy.observe(suggestion, summary.metric, summary.cost, is_failure=True)
            continue
        duration = time.time() - start
        summary = _summary_from_episode(metric, params_record, stats.rewards or [], stats.infos or [], duration)
        summary.info.update(extra or {})
        history.append(summary)
        strategy.observe(suggestion, summary.metric, summary.cost, is_failure=False)
        score = _metric_from_stats(metric, goal, summary)
        if best_summary is None or score > _metric_from_stats(metric, goal, best_summary):
            best_summary = summary
            best_config = cfg
        fill_template.update(suggestion)

    if best_config is None or best_summary is None:
        raise RuntimeError("Sweep did not produce a valid configuration")

    try:
        persist_path = (workspace / persist_path) if not persist_path.is_absolute() else persist_path
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": vars(best_config),
            "summary": {
                "metric": best_summary.metric,
                "avg_reward": best_summary.avg_reward,
                "avg_pass_rate": best_summary.avg_pass_rate,
                "avg_blast_radius": best_summary.avg_blast_radius,
                "cost": best_summary.cost,
            },
            "history": [
                {
                    "metric": item.metric,
                    "avg_reward": item.avg_reward,
                    "avg_pass_rate": item.avg_pass_rate,
                    "avg_blast_radius": item.avg_blast_radius,
                    "cost": item.cost,
                    "params": item.params,
                    "info": item.info,
                }
                for item in history
            ],
        }
        persist_path.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass

    return SweepOutcome(best_config=best_config, best_summary=best_summary, history=history)
