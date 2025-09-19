"""Convenience launcher so the agent runs with one command anywhere."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_MODEL = "qwen3:1.7b"


def _print_success(message: str) -> None:
    print(f"[+] {message}")


def _print_warning(message: str) -> None:
    print(f"[!] {message}")


def _print_info(message: str) -> None:
    print(f"[>] {message}")


def _ensure_workspace(path: Path) -> Path:
    """Create the workspace and logs folder if they do not exist."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        _print_success(f"Workspace created at {path}")
    logs = path / "logs"
    if not logs.exists():
        logs.mkdir(parents=True, exist_ok=True)
        _print_success("Logs directory ready")
    return path


def _check_ollama(model: str, *, pull: bool, skip: bool) -> None:
    if skip:
        return
    executable = shutil.which("ollama")
    if not executable:
        _print_warning("Ollama not found on PATH; set OpenAI-compatible vars or install from https://ollama.com/download")
        return
    _print_success(f"Ollama detected at {executable}")
    try:
        result = subprocess.run(
            [executable, "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        _print_warning("Unable to query Ollama models (is the daemon running?)")
        return
    if model in result.stdout:
        _print_success(f"Model '{model}' is available")
        return
    if pull:
        _print_info(f"Pulling Ollama model '{model}'...")
        try:
            subprocess.run([executable, "pull", model], check=True)
            _print_success(f"Model '{model}' downloaded")
        except subprocess.CalledProcessError:
            _print_warning(f"Failed to pull model '{model}'. Try manually: ollama pull {model}")
    else:
        _print_warning(f"Model '{model}' not found. Pull it with: ollama pull {model}")


def _export_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if not args.allow_auto_train:
        env.setdefault("DSPY_AUTO_TRAIN", "false")
    if args.log_level:
        env.setdefault("DSPY_LOG_LEVEL", args.log_level)
    if args.force_openai:
        env.pop("USE_OLLAMA", None)
    else:
        if shutil.which("ollama"):
            env.setdefault("USE_OLLAMA", "true")
        if args.model:
            env.setdefault("OLLAMA_MODEL", args.model)
    return env


def _build_agent_command(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = [sys.executable, "-m", "dspy_agent.cli"]
    effective_mode = args.mode
    if args.open_dashboard and effective_mode == "cli":
        effective_mode = "dashboard"

    if effective_mode in {"code", "dashboard"}:
        cmd.append("code")

    if effective_mode == "dashboard" or (args.open_dashboard and effective_mode == "code"):
        cmd.append("--open-dashboard")

    extras = list(args.agent_args or [])
    if extras and extras[0] == "--":
        extras = extras[1:]
    cmd.extend(extras)
    return cmd


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a workspace and launch the DSPy agent",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace to operate on (default: current directory)",
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "code", "dashboard"],
        default="cli",
        help="Launch mode",
    )
    parser.add_argument(
        "--open-dashboard",
        action="store_true",
        help="Also launch the dashboard (implies --mode dashboard if not set)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate prerequisites and exit",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Do not probe for Ollama availability",
    )
    parser.add_argument(
        "--pull-model",
        action="store_true",
        help="Attempt to pull the Ollama model if missing",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Preferred Ollama model to check or pull",
    )
    parser.add_argument(
        "--force-openai",
        action="store_true",
        help="Skip Ollama setup even when available",
    )
    parser.add_argument(
        "--allow-auto-train",
        action="store_true",
        help="Leave DSPY_AUTO_TRAIN unchanged",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Default DSPY_LOG_LEVEL if not already set",
    )
    parser.add_argument(
        "agent_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed through to the agent",
    )
    return parser.parse_args(list(argv))


def run(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()
    _print_info(f"Workspace resolved to {workspace}")
    _ensure_workspace(workspace)

    _check_ollama(args.model, pull=args.pull_model, skip=args.skip_ollama or args.force_openai)

    if args.check_only:
        _print_success("Environment checks passed")
        return 0

    env = _export_environment(args)
    cmd = _build_agent_command(args)
    _print_info("Launching agent...")
    _print_info(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=str(workspace), env=env)
    except subprocess.CalledProcessError as exc:
        _print_warning("Agent exited with errors")
        return exc.returncode
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv or sys.argv[1:])


def console_entry() -> None:
    sys.exit(main())


if __name__ == "__main__":
    console_entry()
