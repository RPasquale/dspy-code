#!/usr/bin/env python3
"""Cross-platform helper that bootstraps and runs the DSPy agent."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dspy_agent.launcher import DEFAULT_MODEL


def print_success(message: str) -> None:
    print(f"[+] {message}")


def print_warning(message: str) -> None:
    print(f"[!] {message}")


def print_info(message: str) -> None:
    print(f"[>] {message}")


def run_command(cmd: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a command and bubble up failures with readable output."""
    cwd = cwd or PROJECT_ROOT
    display = " ".join(cmd)
    print_info(f"$ (cwd={cwd}) {display}")
    return subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def ensure_project_root() -> None:
    marker = PROJECT_ROOT / "pyproject.toml"
    if not marker.exists():
        print_warning("pyproject.toml not found next to this script; run from the repository checkout.")
        sys.exit(1)


def ensure_uv_available() -> str:
    uv_path = shutil.which("uv")
    if uv_path:
        print_success(f"uv found at {uv_path}")
        return uv_path
    print_warning("uv is required but was not found on PATH.")
    print("Install it via `pip install uv` or follow https://docs.astral.sh/uv/getting-started/install/")
    sys.exit(1)


def ensure_virtualenv(uv_path: str) -> None:
    venv_dir = PROJECT_ROOT / ".venv"
    if venv_dir.exists():
        print_success("Virtual environment detected (./.venv)")
        return
    print_info("Creating virtual environment with uv venv...")
    run_command([uv_path, "venv"])
    print_success("Virtual environment created")


def sync_dependencies(uv_path: str, quiet: bool) -> None:
    args = [uv_path, "sync"]
    if quiet:
        args.append("--quiet")
    print_info("Installing/updating dependencies...")
    run_command(args)
    print_success("Dependencies ready")


def check_ollama(model: str, *, pull: bool, skip: bool) -> None:
    if skip:
        return
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        print_warning("Ollama not found; set OPENAI-compatible environment variables or install Ollama from https://ollama.com/download")
        return
    print_success(f"Ollama detected at {ollama_path}")
    try:
        result = subprocess.run([ollama_path, "list"], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        print_warning("Unable to query Ollama models (is the daemon running?)")
        return
    if model in result.stdout:
        print_success(f"Model '{model}' is available")
        return
    if pull:
        print_info(f"Pulling Ollama model '{model}'...")
        try:
            subprocess.run([ollama_path, "pull", model], check=True)
            print_success(f"Model '{model}' downloaded")
        except subprocess.CalledProcessError:
            print_warning(f"Failed to pull model '{model}'. You can try manually: ollama pull {model}")
    else:
        print_warning(f"Model '{model}' not found. Pull it with: ollama pull {model}")


def build_agent_command(args: argparse.Namespace, *, uv_path: str) -> list[str]:
    base = [uv_path, "run", "dspy-agent"]
    if args.mode == "code":
        base.append("code")
    if args.open_dashboard and args.mode != "dashboard":
        base.append("--open-dashboard")
    if args.mode == "dashboard":
        base.extend(["code", "--open-dashboard"])
    workspace = Path(args.workspace).expanduser().resolve()
    base.extend(["--workspace", str(workspace)])
    if args.agent_args:
        extras = list(args.agent_args)
        if extras and extras[0] == "--":
            extras = extras[1:]
        base.extend(extras)
    return base


def export_environment(args: argparse.Namespace) -> dict[str, str]:
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
    if args.model and not args.force_openai:
        env.setdefault("OLLAMA_MODEL", args.model)
    return env


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap dependencies and launch the DSPy agent")
    parser.add_argument("--workspace", default=str(PROJECT_ROOT), help="Workspace for the agent (defaults to the repository root)")
    parser.add_argument("--mode", choices=["cli", "code", "dashboard"], default="cli", help="Agent launch mode")
    parser.add_argument("--skip-sync", action="store_true", help="Skip dependency installation")
    parser.add_argument("--quiet-sync", action="store_true", help="Use `uv sync --quiet`")
    parser.add_argument("--check-only", action="store_true", help="Check prerequisites and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Preferred Ollama model to check or pull")
    parser.add_argument("--pull-model", action="store_true", help="Attempt to pull the Ollama model if missing")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama availability checks")
    parser.add_argument("--force-openai", action="store_true", help="Do not configure Ollama even if available")
    parser.add_argument("--allow-auto-train", action="store_true", help="Leave DSPY_AUTO_TRAIN unchanged")
    parser.add_argument("--log-level", default="INFO", help="Default DSPY_LOG_LEVEL if not already set")
    parser.add_argument("--open-dashboard", action="store_true", help="Launch the dashboard alongside the CLI")
    parser.add_argument("agent_args", nargs=argparse.REMAINDER, help="Additional arguments passed to dspy-agent")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    ensure_project_root()
    args = parse_args(argv)

    uv_path = ensure_uv_available()
    ensure_virtualenv(uv_path)

    if not args.skip_sync:
        sync_dependencies(uv_path, args.quiet_sync)
    else:
        print_warning("Skipping dependency sync; ensure the environment is up to date")

    check_ollama(args.model, pull=args.pull_model, skip=args.skip_ollama or args.force_openai)

    if args.check_only:
        print_success("Environment checks passed")
        return 0

    env = export_environment(args)
    cmd = build_agent_command(args, uv_path=uv_path)
    try:
        run_command(cmd, env=env)
    except subprocess.CalledProcessError as exc:
        print_warning("Agent exited with a non-zero status")
        return exc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
