from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import typer

from .cli import (
    _rl_build_make_env,
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
from .rl.rlkit import train_puffer_policy as _rl_train_puffer_policy
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
    modules_norm: List[str] = []
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
    reflection_model = (
        os.getenv('GEPA_MODEL')
        or os.getenv('MODEL_NAME')
        or os.getenv('OLLAMA_MODEL')
    )
    lm = _maybe_configure_lm(
        True,
        use_ollama,
        reflection_model,
        os.getenv('OPENAI_BASE_URL'),
        os.getenv('OPENAI_API_KEY'),
        workspace=workspace,
    )

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


def impl_rl_puffer(
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
        _stream_action_event(workspace, 'rl.train.start', {'steps': int(steps), 'n_envs': int(n_envs)})
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
        console.print(Panel(escape(str(e)), title="rl train failed", border_style="red"))
        raise typer.Exit(1)

    rewards = stats.rewards or []
    avg_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
    console.print(Panel.fit(f"steps={len(rewards)} avg_reward={avg_reward:.3f}", title="rl train", border_style="cyan"))
    try:
        _stream_metric_event(workspace, 'rl.train.summary', {'steps': len(rewards), 'avg_reward': float(avg_reward)})
        _stream_action_event(workspace, 'rl.train.finished', {'steps': len(rewards), 'avg_reward': float(avg_reward)})
    except Exception:
        pass


@rl_trainers_app.command("train")
def train(
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
    impl_rl_puffer(
        workspace=workspace,
        steps=steps,
        n_envs=n_envs,
        lr=lr,
        entropy_coef=entropy_coef,
        replay_capacity=replay_capacity,
        replay_batch=replay_batch,
        grad_clip=grad_clip,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        early_stop_patience=early_stop_patience,
        log_interval=log_interval,
        echo_actions=echo_actions,
        log_jsonl=log_jsonl,
        skip_gepa=skip_gepa,
        gepa_modules=gepa_module,
    )
