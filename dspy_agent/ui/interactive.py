"""High-level interactive shell for `dspy-code`.

The goal is to provide an opinionated, menu-driven experience so users never
have to remember subcommands or flags. All existing workflows (task execution,
RL sweeps, async training, memory inspection) are wired behind numbered menu
options with sensible defaults.
"""

from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.prompt import Prompt, Confirm
from rich.text import Text

from ..config import get_settings
from ..context.context_manager import ContextManager
from ..rl.rlkit import RLConfig
from ..rl.async_loop import AsyncRLTrainer
from ..training.rl_sweep import run_sweep, describe_default_hparams, load_sweep_config, DEFAULT_CONFIG_PATH
from ..db import get_enhanced_data_manager, Environment
from ..llm import get_sampling_hints

STATE_PATH = ".dspy_session_state.json"


@dataclass
class SessionState:
    recent_tasks: List[str]
    last_option: Optional[str]
    last_updated: float

    @classmethod
    def load(cls, workspace: Path) -> "SessionState":
        path = workspace / STATE_PATH
        if not path.exists():
            return cls(recent_tasks=[], last_option=None, last_updated=time.time())
        try:
            data = json.loads(path.read_text())
            return cls(
                recent_tasks=data.get("recent_tasks", []),
                last_option=data.get("last_option"),
                last_updated=data.get("last_updated", time.time()),
            )
        except Exception:
            return cls(recent_tasks=[], last_option=None, last_updated=time.time())

    def persist(self, workspace: Path) -> None:
        path = workspace / STATE_PATH
        payload = {
            "recent_tasks": self.recent_tasks[-10:],
            "last_option": self.last_option,
            "last_updated": time.time(),
        }
        try:
            path.write_text(json.dumps(payload, indent=2))
        except Exception:
            pass


class InteractiveShell:
    def __init__(self, workspace: Path, logs: Optional[Path] = None) -> None:
        self.console = Console()
        self.workspace = workspace.resolve()
        self.logs = logs.resolve() if logs else (self.workspace / "logs")
        self.settings = get_settings()
        self.state = SessionState.load(self.workspace)
        self._running = True

    # ------------------------------------------------------------------
    def run(self) -> None:
        self._greet()
        while self._running:
            self._render_dashboard()
            choice = self._prompt_main_menu()
            self.state.last_option = choice
            self._dispatch(choice)
            self.state.persist(self.workspace)

    # ------------------------------------------------------------------
    def _greet(self) -> None:
        banner = Text.from_markup("[bold cyan]DSPy Code Companion[/bold cyan]")
        self.console.print(Panel(banner, title="Welcome", border_style="cyan"))
        if self.state.recent_tasks:
            last = self.state.recent_tasks[-1]
            self.console.print(f"[dim]Last session task: [italic]{last}[/italic][/dim]")
        self.console.print("[dim]Use numbers to choose an action. Press Ctrl+C to exit at any time.[/dim]\n")

    # ------------------------------------------------------------------
    def _render_dashboard(self) -> None:
        harvest = self._gather_status()
        panels: List[Panel] = []
        panels.append(self._render_workspace_panel(harvest["workspace"]))
        panels.append(self._render_rl_panel(harvest["rl_config"], harvest["sampling"]))
        panels.append(self._render_memory_panel(harvest["memory"]))
        self.console.print(Columns(panels, expand=True))

    def _gather_status(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        info["workspace"] = {
            "path": str(self.workspace),
            "logs": str(self.logs),
            "local_mode": self.settings.local_mode,
        }
        info["rl_config"] = self._load_rl_config()
        info["sampling"] = get_sampling_hints()
        ctx = ContextManager(self.workspace, self.logs)
        bundle = ctx.build_patch_context("status")
        info["memory"] = {
            "patch_stats": bundle.get("stats", {}),
            "retrieval_events": bundle.get("retrieval_events", []),
            "kg_features": bundle.get("kg_features", []),
        }
        return info

    def _render_workspace_panel(self, data: Dict[str, Any]) -> Panel:
        lines = [
            f"Workspace: [bold]{data['path']}[/bold]",
            f"Logs: {data['logs']}",
            f"Local mode: {data['local_mode']}",
        ]
        if self.state.recent_tasks:
            lines.append("Recent tasks:")
            for task in self.state.recent_tasks[-5:]:
                lines.append(f"- {task}")
        return Panel("\n".join(lines), title="Workspace", border_style="blue")

    def _render_rl_panel(self, cfg: Optional[RLConfig], sampling: Dict[str, float]) -> Panel:
        if cfg is None:
            body = "No RL config found. Run a sweep to generate one."
        else:
            weights = cfg.weights or {}
            lines = [
                f"Policy: {cfg.policy} | n_envs: {cfg.n_envs}",
                f"Temp: {getattr(cfg, 'temperature', 0.85):.2f} | Target H: {getattr(cfg, 'target_entropy', 0.3):.2f} | Clip+: {getattr(cfg, 'clip_higher', 1.1):.2f}",
                f"Weights: pass={weights.get('pass_rate',1.0):.2f} blast={weights.get('blast_radius',1.0):.2f} retr_prec={weights.get('retrieval_precision',0.3):.2f}",
            ]
            body = "\n".join(lines)
        if sampling:
            hints = " | ".join(f"{k}={v:.2f}" for k, v in sampling.items())
            body += f"\nActive sampling hints: {hints}"
        return Panel(body, title="RL Config", border_style="magenta")

    def _render_memory_panel(self, memory: Dict[str, Any]) -> Panel:
        stats = memory.get("patch_stats", {})
        retrievals = memory.get("retrieval_events", [])
        lines = [
            f"Patches: {int(stats.get('total',0))} | success={stats.get('recent_success_rate',0.0):.2f}",
            f"Avg pass={stats.get('avg_pass_rate',0.0):.2f} | Avg blast={stats.get('avg_blast_radius',0.0):.2f}",
        ]
        if retrievals:
            last = retrievals[-1]
            lines.append(f"Last retrieval: '{last.get('query','')}' ({len(last.get('hits',[]))} hits)")
        else:
            lines.append("No retrieval events logged yet.")
        return Panel("\n".join(lines), title="Memory", border_style="green")

    # ------------------------------------------------------------------
    def _prompt_main_menu(self) -> str:
        menu = [
            "[1] Start guided task",
            "[2] Review memory & retrievals",
            "[3] Run RL sweep",
            "[4] Run async trainer",
            "[5] View configuration",
            "[6] Command palette",
            "[0] Exit",
        ]
        self.console.print(Panel("\n".join(menu), title="Main Menu", border_style="cyan"))
        return Prompt.ask("Select an option", choices=["1","2","3","4","5","6","0"], default="1")

    def _dispatch(self, choice: str) -> None:
        try:
            if choice == "1":
                self._start_task_flow()
            elif choice == "2":
                self._show_memory_detail()
            elif choice == "3":
                self._run_rl_sweep_flow()
            elif choice == "4":
                self._run_async_trainer_flow()
            elif choice == "5":
                self._show_configuration_detail()
            elif choice == "6":
                self._command_palette()
            else:
                self._running = False
                self.console.print("[green]Goodbye![/green]")
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Action cancelled.[/yellow]")
        except Exception as exc:
            self.console.print(Panel(f"{exc}", title="Error", border_style="red"))

    # ------------------------------------------------------------------
    def _start_task_flow(self) -> None:
        task = Prompt.ask("Describe what you want to accomplish")
        if not task.strip():
            self.console.print("[yellow]Task cancelled.[/yellow]")
            return
        auto_mode = Confirm.ask("Auto-apply patches when safe?", default=True)
        approval = "auto" if auto_mode else "manual"
        self.console.print(Panel(f"Running guided agent on '{task}'", border_style="cyan"))
        try:
            from ..cli import start_command
            start_command(
                workspace=self.workspace,
                logs=self.logs if self.logs.exists() else None,
                ollama=self.settings.use_ollama,
                model=self.settings.model_name,
                base_url=self.settings.openai_base_url,
                api_key=self.settings.openai_api_key,
                approval=approval,
            )
        except SystemExit:
            pass
        self.state.recent_tasks.append(task)
        self.console.print("[green]Task session complete.[/green]")

    def _show_memory_detail(self) -> None:
        ctx = ContextManager(self.workspace, self.logs)
        bundle = ctx.build_patch_context("memory", max_patches=10)
        retrievals = bundle.get('retrieval_events', [])
        kg = bundle.get('kg_features', [])
        table = Table(title="Recent Patches", show_lines=True)
        table.add_column("When")
        table.add_column("Result")
        table.add_column("Pass")
        table.add_column("Files")
        for rec in bundle.get('patches', [])[:10]:
            table.add_row(
                str(rec.get('human_ts','')),
                str(rec.get('result','')),
                f"{rec.get('metrics',{}).get('pass_rate',0):.2f}",
                str(rec.get('file_candidates','')),
            )
        self.console.print(table)
        if retrievals:
            retr_table = Table(title="Retrieval Events", show_lines=True)
            retr_table.add_column("Timestamp")
            retr_table.add_column("Query")
            retr_table.add_column("Hits")
            retr_table.add_column("Sources")
            for ev in retrievals[-10:]:
                ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ev.get('timestamp', time.time())))
                hits = ev.get('hits') or []
                sources = {h.get('source','?') for h in hits}
                retr_table.add_row(ts, ev.get('query',''), str(len(hits)), ','.join(sources))
            self.console.print(retr_table)
        if kg:
            self.console.print(Panel(f"KG features: {kg}", title="Knowledge Graph", border_style="green"))
        self.console.print(Panel("Press Enter to return to the menu.", border_style="dim"))
        input()

    def _run_rl_sweep_flow(self) -> None:
        iterations = Prompt.ask("Iterations", default="12")
        try:
            iterations_int = int(iterations)
        except ValueError:
            iterations_int = 12
        try:
            sweep_cfg = load_sweep_config(DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None)
        except Exception:
            sweep_cfg = {}
        sweep_cfg = dict(sweep_cfg or {})
        sweep_cfg['iterations'] = iterations_int
        sweep_cfg.setdefault('metric', 'reward')
        sweep_cfg.setdefault('goal', 'maximize')
        method_default = str(sweep_cfg.get('method', 'pareto')).lower() or 'pareto'
        method_choice = Prompt.ask(
            "Sweep method",
            choices=["pareto", "random", "protein", "carbs"],
            default=method_default,
        )
        if method_choice == "protein" and importlib.util.find_spec("pyro") is None:
            self.console.print(Panel("Protein sweep requires pyro-ppl (install extras via `pip install .[rl]`).", title="Dependency missing", border_style="yellow"))
            return
        if method_choice == "carbs" and importlib.util.find_spec("carbs") is None:
            self.console.print(Panel("Carbs sweep requires the `carbs` package (install extras via `pip install .[rl]`).", title="Dependency missing", border_style="yellow"))
            return
        sweep_cfg['method'] = method_choice
        self.console.print(Panel(f"Running sweep ({iterations_int} iterations)...", border_style="magenta"))
        base_cfg = self._load_rl_config()
        outcome = run_sweep(self.workspace, sweep_cfg, base_config=base_cfg)
        best = outcome.best_summary
        body = (
            f"metric={best.metric:.4f}\n"
            f"avg_reward={best.avg_reward:.4f}\n"
            f"avg_pass_rate={best.avg_pass_rate:.4f}\n"
            f"avg_blast_radius={best.avg_blast_radius:.4f}\n"
            f"iterations={len(outcome.history)}"
        )
        self.console.print(Panel(body, title="Sweep Result", border_style="cyan"))
        self.console.print("[green]Best config persisted to .dspy/rl/best.json.[/green]")
        input("Press Enter to continue...")

    def _run_async_trainer_flow(self) -> None:
        rollout = int(Prompt.ask("Rollout workers", default="2"))
        judges = int(Prompt.ask("Judge workers", default="2"))
        steps = int(Prompt.ask("Target steps", default="200"))
        wall_clock = float(Prompt.ask("Wall-clock seconds", default="120"))
        self.console.print(Panel(f"Starting async trainer ({steps} steps | {wall_clock}s max)...", border_style="magenta"))
        from ..cli import _rl_build_make_env
        cfg = self._load_rl_config()
        make_env = _rl_build_make_env(
            self.workspace,
            verifiers_module=cfg.verifiers_module if cfg else None,
            weights=cfg.weights if cfg and cfg.weights else {"pass_rate":1.0, "blast_radius":1.0},
            penalty_kinds=cfg.penalty_kinds if cfg else [],
            clamp01_kinds=cfg.clamp01_kinds if cfg else [],
            scales=cfg.scales if cfg else {},
            test_cmd=cfg.test_cmd if cfg else None,
            lint_cmd=cfg.lint_cmd if cfg else None,
            build_cmd=cfg.build_cmd if cfg else None,
            timeout_sec=cfg.timeout_sec if cfg else None,
            actions=cfg.actions if cfg else None,
        )
        trainer = AsyncRLTrainer(make_env, rollout_workers=rollout, judge_workers=judges)
        trainer.start()
        t0 = time.time()
        try:
            while time.time() - t0 < wall_clock and trainer.snapshot_stats().get('count', 0.0) < steps:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.console.print("[yellow]Stopping trainer...[/yellow]")
        trainer.stop(); trainer.join()
        stats = trainer.snapshot_stats()
        self.console.print(Panel(json.dumps(stats, indent=2), title="Async Stats", border_style="green"))
        input("Press Enter to continue...")

    def _show_configuration_detail(self) -> None:
        cfg = self._load_rl_config()
        sampling = get_sampling_hints()
        table = Table(title="RL Configuration", show_lines=True)
        table.add_column("Field")
        table.add_column("Value")
        if cfg:
            for field in (
                "policy", "epsilon", "ucb_c", "n_envs", "temperature", "target_entropy", "clip_higher",
            ):
                table.add_row(field, str(getattr(cfg, field, None)))
            weights = cfg.weights or {}
            for name, value in weights.items():
                table.add_row(f"weight:{name}", f"{value:.3f}")
        if sampling:
            for k, v in sampling.items():
                table.add_row(f"sampling:{k}", f"{v:.3f}")
        guide = describe_default_hparams()
        self.console.print(table)
        guide_panels = []
        for group in guide:
            rows = []
            for item in group['items']:
                tgt = item.get('target')
                tgt_str = f"→{tgt:.2f}" if isinstance(tgt, (int, float)) else ""
                rows.append(f"[bold]{item['name']}[/bold]: {item['low']:.2f}-{item['high']:.2f}{tgt_str}\n{item['rationale']}")
            guide_panels.append(Panel("\n".join(rows), title=group['title'], border_style="cyan"))
        self.console.print(Columns(guide_panels, expand=True))
        input("Press Enter to continue...")

    def _command_palette(self) -> None:
        raw = Prompt.ask("Enter a dspy-agent command (or leave blank to cancel)")
        if not raw.strip():
            return
        from subprocess import CalledProcessError, run
        try:
            result = run(["dspy-agent", *raw.split()], capture_output=True, text=True, check=True)
            stdout = result.stdout or "(no output)"
            self.console.print(Panel(stdout, title=f"dspy-agent {raw}", border_style="cyan"))
            if result.stderr:
                self.console.print(Panel(result.stderr, title="stderr", border_style="yellow"))
        except CalledProcessError as e:
            self.console.print(Panel(e.stderr or str(e), title="Command failed", border_style="red"))

    # ------------------------------------------------------------------
    def _load_rl_config(self) -> Optional[RLConfig]:
        try:
            from ..rl.rl_helpers import load_effective_rl_config_dict, rl_config_from_dict
            data = load_effective_rl_config_dict(self.workspace)
            if not data:
                return None
            return rl_config_from_dict(data)
        except Exception:
            return None


def run_interactive_shell(workspace: Path, logs: Optional[Path] = None) -> None:
    shell = InteractiveShell(workspace, logs)
    shell.run()
