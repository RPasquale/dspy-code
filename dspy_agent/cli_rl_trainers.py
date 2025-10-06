from __future__ import annotations

import json
import os
import time
import typing as _typing
from collections import Counter
from pathlib import Path
from typing import List, Optional

import typer

from .cli import (
    _rl_build_make_env,
    _RLTrainerConfig,
    _rl_bandit_trainer,
    _rl_bandit_trainer_puffer,
    _rl_train_puffer_policy,
    _rl_run_puffer_ppo,
    _RLConfig,
    _load_rl_config,
    _stream_action_event,
    _stream_metric_event,
    console,
    Panel,
    escape,
    _maybe_configure_lm,
    _get_code_summary,
)
from .rl.rl_helpers import (
    load_effective_rl_config_dict as _load_effective_rl_config_dict,
    rl_config_from_dict as _rl_config_from_dict,
)
from .training.autogen_dataset import bootstrap_datasets
from .training.train_gepa import run_gepa
from .training.train_orchestrator import run_gepa_orchestrator
from .training.train_codegen import run_gepa_codegen

rl_trainers_app = typer.Typer(no_args_is_help=True, help="RL trainer commands")


def _run_gepa_cycle(workspace: Path, modules: List[str]) -> None:
    modules_norm = []
    for mod in modules:
        name = str(mod).strip().lower()
        if name and name not in modules_norm:
            modules_norm.append(name)
    if not modules_norm:
        modules_norm = ["context", "task", "code"]

    logs_dir = workspace / 'logs'
    data_dir = workspace / '.dspy_data'
    try:
        paths = bootstrap_datasets(workspace, logs_dir if logs_dir.exists() else None, data_dir)
    except Exception as exc:
        console.print(Panel(f"Failed to bootstrap GEPA datasets: {exc}", title="gepa", border_style="red"))
        return

    use_ollama = os.getenv('USE_OLLAMA', 'true').lower() not in {'0', 'false', 'no'}
    model_name = os.getenv('MODEL_NAME') or os.getenv('OLLAMA_MODEL')
    lm = _maybe_configure_lm(True, use_ollama, model_name, os.getenv('OPENAI_BASE_URL'), os.getenv('OPENAI_API_KEY'), workspace=workspace)

    code_summary: Optional[str] = None
    if any(m in {'context', 'task', 'code'} for m in modules_norm):
        try:
            code_summary = _get_code_summary(workspace)
        except Exception:
            code_summary = None

    for module in modules_norm:
        dataset_path = paths.get(module)
        if not dataset_path or not dataset_path.exists():
            console.print(Panel(f"Skipping GEPA for '{module}' (no dataset)", title="gepa", border_style="yellow"))
            continue
        log_dir = workspace / f'.gepa_{module}'
        progress_path = log_dir / 'progress.jsonl'
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            console.print(Panel.fit(f"Running GEPA for {module}", title="gepa", border_style="cyan"))
            if module == 'orchestrator':
                run_gepa_orchestrator(
                    train_jsonl=dataset_path,
                    auto='light',
                    reflection_lm=lm,
                    log_dir=str(log_dir),
                    track_stats=True,
                    progress_path=str(progress_path),
                )
            elif module == 'codegen':
                run_gepa_codegen(
                    train_jsonl=dataset_path,
                    workspace=workspace,
                    test_cmd=None,
                    type_cmd="python -m compileall -q .",
                    lint_cmd=None,
                    auto='light',
                    reflection_lm=lm,
                    log_dir=str(log_dir),
                    track_stats=True,
                )
            else:
                run_gepa(
                    module=module,
                    train_jsonl=dataset_path,
                    auto='light',
                    reflection_lm=lm,
                    log_dir=str(log_dir),
                    track_stats=True,
                    progress_path=str(progress_path),
                    code_summary=code_summary,
                )
            console.print(Panel.fit(f"GEPA {module} completed", title="gepa", border_style="green"))
        except Exception as exc:
            console.print(Panel(f"GEPA {module} failed: {exc}", title="gepa", border_style="red"))


def impl_rl_train(
    workspace: Path,
    steps: int,
    n_envs: int,
    policy: str,
    epsilon: float,
    ucb_c: float,
    neural: bool,
    vector_path: Path | None,
    puffer: bool,
    rl_config: Path | None,
    verifiers_module: str | None,
    test_cmd: str | None,
    lint_cmd: str | None,
    build_cmd: str | None,
    timeout_sec: int | None,
) -> None:
    try:
        _stream_action_event(workspace, 'rl.train.start', {'policy': policy, 'steps': int(steps), 'n_envs': int(n_envs)})
    except Exception:
        pass
    cfg: _RLConfig | None = None
    if rl_config:
        try:
            cfg = _load_rl_config(rl_config)
        except Exception as e:
            console.print(Panel(escape(str(e)), title="rl config load failed", border_style="red"))
            raise typer.Exit(1)
    else:
        cfg_data = _load_effective_rl_config_dict(workspace)
        if cfg_data:
            try:
                cfg = _rl_config_from_dict(cfg_data)
            except Exception:
                cfg = None
    eff_policy = policy or (cfg.policy if cfg else "epsilon-greedy")
    eff_epsilon = epsilon if epsilon is not None else ((cfg.epsilon if cfg else 0.1))
    eff_ucb_c = ucb_c if ucb_c is not None else ((cfg.ucb_c if cfg else 2.0))
    eff_n_envs = n_envs or (cfg.n_envs if cfg else 2)
    eff_puffer = (puffer if puffer else (cfg.puffer if cfg else False))
    eff_verifiers_mod = verifiers_module or ((cfg.verifiers_module if cfg else None))
    eff_weights = (cfg.weights if cfg and cfg.weights else {"pass_rate": 1.0, "blast_radius": 1.0})
    eff_pen = cfg.penalty_kinds if cfg else []
    eff_clamp = cfg.clamp01_kinds if cfg else []
    eff_scales = cfg.scales if cfg else {}
    eff_test = test_cmd if test_cmd is not None else ((cfg.test_cmd if cfg else None))
    eff_lint = lint_cmd if lint_cmd is not None else ((cfg.lint_cmd if cfg else None))
    eff_build = build_cmd if build_cmd is not None else ((cfg.build_cmd if cfg else None))
    eff_to = timeout_sec if timeout_sec is not None else ((cfg.timeout_sec if cfg else None))
    eff_actions = cfg.actions if cfg and getattr(cfg, 'actions', None) else None
    make_env = _rl_build_make_env(
        workspace,
        verifiers_module=eff_verifiers_mod,
        weights=eff_weights,
        penalty_kinds=eff_pen,
        clamp01_kinds=eff_clamp,
        scales=eff_scales,
        test_cmd=eff_test,
        lint_cmd=eff_lint,
        build_cmd=eff_build,
        timeout_sec=eff_to,
        actions=eff_actions,
    )
    try:
        _stream_metric_event(workspace, 'rl.train.env_ready', {'actions': (eff_actions or []), 'puffer': bool(eff_puffer)})
    except Exception:
        pass
    kwargs: dict[str, _typing.Any] = {}
    pol = (eff_policy or "epsilon-greedy").lower().strip()
    if pol in {"epsilon", "epsilon-greedy", "egreedy", "eps"}:
        kwargs["epsilon"] = float(eff_epsilon)
    elif pol in {"ucb", "ucb1"}:
        kwargs["c"] = float(eff_ucb_c)
    try:
        if neural:
            stats = _rl_train_puffer_policy(make_env=make_env, steps=int(steps), n_envs=int(eff_n_envs), verbose=True, log_interval=max(1, int(steps)//10 or 1))
        elif eff_puffer:
            tcfg = _RLTrainerConfig(steps=int(steps), policy=eff_policy, policy_kwargs=kwargs, n_envs=int(eff_n_envs))
            try:
                stats = _rl_bandit_trainer_puffer(make_env, tcfg)
            except Exception:
                tcfg_fallback = _RLTrainerConfig(steps=int(steps), policy=eff_policy, policy_kwargs=kwargs, n_envs=1)
                stats = _rl_bandit_trainer(make_env, tcfg_fallback)
        else:
            tcfg = _RLTrainerConfig(steps=int(steps), policy=eff_policy, policy_kwargs=kwargs, n_envs=1)
            stats = _rl_bandit_trainer(make_env, tcfg)
    except Exception as e:
        console.print(Panel(escape(str(e)), title="rl train failed", border_style="red"))
        raise typer.Exit(1)
    r = stats.rewards or []
    avg = (sum(r) / len(r)) if r else 0.0
    tools = Counter([str(it.get("tool", "")) for it in (stats.infos or []) if isinstance(it, dict)])
    console.print(Panel.fit(f"steps={len(r)} avg_reward={avg:.3f}\ntools={dict(tools)}", title="rl train result", border_style="cyan"))


@rl_trainers_app.command("train")
def train(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True),
    steps: int = typer.Option(200, '--steps'),
    n_envs: int = typer.Option(2, '--n-envs'),
    policy: str = typer.Option("epsilon-greedy", '--policy'),
    epsilon: float = typer.Option(0.1, '--epsilon'),
    ucb_c: float = typer.Option(2.0, '--ucb-c'),
    neural: bool = typer.Option(False, '--neural/--no-neural'),
    vector_path: Path | None = typer.Option(None, '--vector-path'),
    puffer: bool = typer.Option(False, '--puffer/--no-puffer'),
    rl_config: Path | None = typer.Option(None, '--rl-config', exists=True),
    verifiers_module: str | None = typer.Option(None, '--verifiers-module'),
    test_cmd: str | None = typer.Option(None, '--test-cmd'),
    lint_cmd: str | None = typer.Option(None, '--lint-cmd'),
    build_cmd: str | None = typer.Option(None, '--build-cmd'),
    timeout_sec: int | None = typer.Option(None, '--timeout-sec'),
):
    impl_rl_train(
        workspace=workspace, steps=steps, n_envs=n_envs, policy=policy, epsilon=epsilon, ucb_c=ucb_c,
        neural=neural, vector_path=vector_path, puffer=puffer, rl_config=rl_config, verifiers_module=verifiers_module,
        test_cmd=test_cmd, lint_cmd=lint_cmd, build_cmd=build_cmd, timeout_sec=timeout_sec,
    )


def impl_rl_async_train(
    workspace: Path,
    steps: int,
    rollout_workers: int,
    judge_workers: int,
    policy: str,
    epsilon: float,
    ucb_c: float,
    rl_config: Path | None,
    wall_clock: float,
) -> None:
    try:
        _stream_action_event(workspace, 'rl.async.start', {'policy': policy, 'steps': int(steps), 'rollouts': int(rollout_workers), 'judges': int(judge_workers)})
    except Exception:
        pass
    cfg: _RLConfig | None = None
    if rl_config:
        try:
            cfg = _load_rl_config(rl_config)
        except Exception as e:
            console.print(Panel(escape(str(e)), title="rl config load failed", border_style="red"))
            raise typer.Exit(1)
    else:
        cfg_data = _load_effective_rl_config_dict(workspace)
        if cfg_data:
            try:
                cfg = _rl_config_from_dict(cfg_data)
            except Exception:
                cfg = None
    make_env = _rl_build_make_env(
        workspace,
        verifiers_module=(cfg.verifiers_module if cfg else None) if cfg else None,
        weights=(cfg.weights if cfg and cfg.weights else {"pass_rate": 1.0, "blast_radius": 1.0}),
        penalty_kinds=(cfg.penalty_kinds if cfg else []),
        clamp01_kinds=(cfg.clamp01_kinds if cfg else []),
        scales=(cfg.scales if cfg else {}),
        test_cmd=(cfg.test_cmd if cfg else None),
        lint_cmd=(cfg.lint_cmd if cfg else None),
        build_cmd=(cfg.build_cmd if cfg else None),
        timeout_sec=(cfg.timeout_sec if cfg else None),
        actions=(cfg.actions if cfg else None),
    )
    kwargs: dict[str, float] = {}
    pol = policy.lower().strip()
    if pol in {"epsilon", "epsilon-greedy", "egreedy", "eps"}:
        kwargs["epsilon"] = float(epsilon)
    elif pol in {"ucb", "ucb1"}:
        kwargs["c"] = float(ucb_c)
    from .rl.async_loop import AsyncRLTrainer
    trainer = AsyncRLTrainer(
        make_env,
        policy=policy,
        policy_kwargs=kwargs,
        rollout_workers=rollout_workers,
        judge_workers=judge_workers,
    )
    trainer.start()
    console.print(Panel.fit(
        f"Async trainer started with {rollout_workers} rollout worker(s) and {judge_workers} judge(s). Running for {wall_clock}s...",
        title="async rl",
        border_style="cyan",
    ))
    t0 = time.time()
    try:
        while time.time() - t0 < wall_clock and trainer.snapshot_stats().get("count", 0.0) < steps:
            time.sleep(1.0)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted, stopping trainer...[/yellow]")
    finally:
        trainer.stop()
        trainer.join()
    summary = trainer.snapshot_stats()
    console.print(Panel.fit(json.dumps(summary, indent=2), title="async stats", border_style="green"))
    try:
        _stream_metric_event(workspace, 'rl.async.summary', {'summary': summary})
        _stream_action_event(workspace, 'rl.async.finished', {'count': summary.get('count', 0), 'avg_reward': summary.get('avg_reward', 0.0)})
    except Exception:
        pass


@rl_trainers_app.command("async-train")
def async_train(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True),
    steps: int = typer.Option(200, '--steps'),
    rollout_workers: int = typer.Option(2, '--rollout-workers'),
    judge_workers: int = typer.Option(2, '--judge-workers'),
    policy: str = typer.Option("epsilon-greedy", '--policy'),
    epsilon: float = typer.Option(0.1, '--epsilon'),
    ucb_c: float = typer.Option(2.0, '--ucb-c'),
    rl_config: Path | None = typer.Option(None, '--rl-config', exists=True),
    wall_clock: float = typer.Option(120.0, '--wall-clock'),
):
    impl_rl_async_train(
        workspace=workspace, steps=steps, rollout_workers=rollout_workers, judge_workers=judge_workers,
        policy=policy, epsilon=epsilon, ucb_c=ucb_c, rl_config=rl_config, wall_clock=wall_clock,
    )


def impl_rl_ppo(
    workspace: Path,
    rl_config: Path | None,
    n_envs: int,
    total_steps: int,
) -> None:
    try:
        _stream_action_event(workspace, 'rl.ppo.start', {'n_envs': int(n_envs), 'total_steps': int(total_steps)})
    except Exception:
        pass
    cfg: _RLConfig | None = None
    if rl_config:
        try:
            cfg = _load_rl_config(rl_config)
        except Exception as e:
            console.print(Panel(escape(str(e)), title="rl config load failed", border_style="red"))
            raise typer.Exit(1)
    eff_actions = cfg.actions if cfg and getattr(cfg, 'actions', None) else None
    make_env = _rl_build_make_env(
        workspace,
        verifiers_module=(cfg.verifiers_module if cfg else None),
        weights=(cfg.weights if cfg else {"pass_rate": 1.0, "blast_radius": 1.0}),
        penalty_kinds=(cfg.penalty_kinds if cfg else []),
        clamp01_kinds=(cfg.clamp01_kinds if cfg else []),
        scales=(cfg.scales if cfg else {}),
        test_cmd=(cfg.test_cmd if cfg else None),
        lint_cmd=(cfg.lint_cmd if cfg else None),
        build_cmd=(cfg.build_cmd if cfg else None),
        timeout_sec=(cfg.timeout_sec if cfg else None),
        actions=eff_actions,
    )
    try:
        _rl_run_puffer_ppo(make_env=make_env, n_envs=int(n_envs), total_steps=int(total_steps))
    except Exception as e:
        console.print(Panel(escape(str(e)), title="rl ppo failed", border_style="red"))
        raise typer.Exit(1)
    try:
        _stream_action_event(workspace, 'rl.ppo.finished', {'n_envs': int(n_envs), 'total_steps': int(total_steps)})
    except Exception:
        pass


@rl_trainers_app.command("ppo")
def ppo(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True),
    rl_config: Path | None = typer.Option(None, '--rl-config', exists=True),
    n_envs: int = typer.Option(4, '--n-envs'),
    total_steps: int = typer.Option(100_000, '--total-steps'),
):
    impl_rl_ppo(workspace=workspace, rl_config=rl_config, n_envs=n_envs, total_steps=total_steps)


def impl_rl_neural(
    workspace: Path,
    steps: int,
    n_envs: int,
    lr: float,
    entropy_coef: float,
    replay_capacity: int,
    replay_batch: int,
    grad_clip: float,
    checkpoint_dir: Path | None,
    checkpoint_interval: int,
    early_stop_patience: int,
    log_interval: int,
    echo_actions: bool,
    log_jsonl: Path | None,
    skip_gepa: bool,
    gepa_modules: List[str],
) -> None:
    try:
        _stream_action_event(workspace, 'rl.neural.start', {'steps': int(steps), 'n_envs': int(n_envs)})
    except Exception:
        pass
    cfg_data = _load_effective_rl_config_dict(workspace)
    cfg: _RLConfig | None = None
    if cfg_data:
        try:
            cfg = _rl_config_from_dict(cfg_data)
        except Exception:
            cfg = None
    modules = [m for m in gepa_modules if m] or ["context", "task", "code"]
    if not skip_gepa:
        _run_gepa_cycle(workspace, modules)
    make_env = _rl_build_make_env(
        workspace,
        verifiers_module=(cfg.verifiers_module if cfg else None) if cfg else None,
        weights=(cfg.weights if cfg and cfg.weights else {"pass_rate": 1.0, "blast_radius": 1.0}),
        penalty_kinds=(cfg.penalty_kinds if cfg else []),
        clamp01_kinds=(cfg.clamp01_kinds if cfg else []),
        scales=(cfg.scales if cfg else {}),
        test_cmd=(cfg.test_cmd if cfg else None),
        lint_cmd=(cfg.lint_cmd if cfg else None),
        build_cmd=(cfg.build_cmd if cfg else None),
        timeout_sec=(cfg.timeout_sec if cfg else None),
        actions=(cfg.actions if cfg else None),
    )
    try:
        stats = _rl_train_puffer_policy(
            make_env=make_env,
            steps=int(steps),
            n_envs=int(n_envs),
            lr=float(lr),
            verbose=True,
            log_interval=max(1, int(log_interval)),
            grad_clip=float(grad_clip),
            checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
            checkpoint_interval=int(checkpoint_interval),
            early_stop_patience=int(early_stop_patience),
            entropy_coef=float(entropy_coef),
            replay_capacity=int(replay_capacity),
            replay_batch=int(replay_batch),
            echo_actions=echo_actions,
            log_jsonl=str(log_jsonl) if log_jsonl else None,
        )
    except Exception as e:
        console.print(Panel(escape(str(e)), title="rl neural failed", border_style="red"))
        raise typer.Exit(1)
    r = stats.rewards or []
    avg = (sum(r) / len(r)) if r else 0.0
    console.print(Panel.fit(f"steps={len(r)} avg_reward={avg:.3f}", title="rl neural", border_style="cyan"))
    try:
        _stream_metric_event(workspace, 'rl.neural.summary', {'steps': len(r), 'avg_reward': float(avg)})
        _stream_action_event(workspace, 'rl.neural.finished', {'steps': len(r), 'avg_reward': float(avg)})
    except Exception:
        pass


@rl_trainers_app.command("neural")
def neural(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True),
    steps: int = typer.Option(500, '--steps'),
    n_envs: int = typer.Option(2, '--n-envs'),
    lr: float = typer.Option(1e-3, '--lr'),
    entropy_coef: float = typer.Option(0.01, '--entropy'),
    replay_capacity: int = typer.Option(2048, '--replay-capacity'),
    replay_batch: int = typer.Option(128, '--replay-batch'),
    grad_clip: float = typer.Option(1.0, '--grad-clip'),
    checkpoint_dir: Path | None = typer.Option(None, '--checkpoint-dir'),
    checkpoint_interval: int = typer.Option(0, '--checkpoint-interval'),
    early_stop_patience: int = typer.Option(0, '--early-stop'),
    log_interval: int = typer.Option(10, '--log-interval', min=1, help="Steps between console log lines"),
    echo_actions: bool = typer.Option(False, '--echo-actions/--no-echo-actions', help="Print each environment action detail"),
    log_jsonl: Path | None = typer.Option(None, '--log-jsonl', help="Write per-step JSON logs to this file"),
    skip_gepa: bool = typer.Option(False, '--skip-gepa', help="Skip GEPA signature training before RL"),
    gepa_module: List[str] = typer.Option([], '--gepa-module', help="GEPA module to train before RL (repeatable, default: context, task, code)"),
):
    impl_rl_neural(
        workspace=workspace, steps=steps, n_envs=n_envs, lr=lr, entropy_coef=entropy_coef,
        replay_capacity=replay_capacity, replay_batch=replay_batch, grad_clip=grad_clip,
        checkpoint_dir=checkpoint_dir, checkpoint_interval=checkpoint_interval, early_stop_patience=early_stop_patience,
        log_interval=log_interval, echo_actions=echo_actions, log_jsonl=log_jsonl,
        skip_gepa=skip_gepa, gepa_modules=gepa_module,
    )
