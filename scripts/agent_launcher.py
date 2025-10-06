#!/usr/bin/env python3
"""One-command launcher for the full DSPy lightweight agent stack.

This utility mirrors the simplicity of the ``codex`` command: run ``dspy-code`` to
boot all microservices (Kafka, Redis, Ollama, RL workers, dashboards, etc.),
wait for them to become healthy, preload the configured Ollama models, and then
drop into an interactive ``dspy-agent`` session inside the container. When you
exit the agent, the stack stays up by default so dashboards and background
workers keep running; use ``dspy-code stop`` to shut everything down.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable, List, Sequence

import scripts.manage_stack as manage_stack


def _detect_local_models() -> List[str]:
    env_models = _split_models(os.getenv("OLLAMA_MODELS"))
    if env_models:
        return env_models
    single = os.getenv("OLLAMA_MODEL") or os.getenv("MODEL_NAME")
    if single:
        models = _split_models(single)
        if models:
            return models
    exe = shutil.which("ollama")
    if exe:
        try:
            result = subprocess.run(
                [exe, "list"],
                capture_output=True,
                text=True,
                check=True,
            )
            models: List[str] = []
            for line in result.stdout.splitlines()[1:]:  # skip header
                parts = line.split()
                if parts:
                    tag = parts[0].strip()
                    if tag:
                        models.append(tag)
            if models:
                # Prefer deepseek-coder if present
                preferred = "deepseek-coder:1.3b"
                if preferred in models:
                    models = [preferred] + [m for m in models if m != preferred]
                return models
        except Exception:
            pass
    # Fallback preference
    return ["deepseek-coder:1.3b", "qwen3:1.7b"]


def _compose_env(use_gpu: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("WORKSPACE_DIR", str(manage_stack.ROOT))
    env.setdefault("DSPY_WORKSPACE", str(manage_stack.ROOT))
    models = _detect_local_models()
    primary = models[0] if models else "deepseek-coder:1.3b"
    env.setdefault("OLLAMA_MODEL", primary)
    env.setdefault("OLLAMA_MODELS", ",".join(models))
    env.setdefault("MODEL_NAME", primary)
    env.setdefault("USE_OLLAMA", "true")
    env.setdefault("DOCKER_ENV", "true")
    env.setdefault("DSPY_AUTO_TRAIN", "0")
    # Match container user to the invoking host user so workspace mounts stay writable.
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        env.setdefault("DSPY_UID", str(os.getuid()))
        env.setdefault("DSPY_GID", str(os.getgid()))
    if use_gpu:
        env.setdefault("DSPY_GPU_COUNT", "1")
        env.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
        env.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")
    return env


def _extra_files(use_gpu: bool) -> List[Path]:
    files: List[Path] = []
    if use_gpu:
        gpu_compose = manage_stack.STACK_GPU_COMPOSE
        if not gpu_compose.exists():
            raise SystemExit(
                f"GPU override file not found: {gpu_compose}. Ensure the repository is up to date."
            )
        files.append(gpu_compose)
    return files


def _split_models(raw: str | None) -> List[str]:
    if not raw:
        return []
    models: List[str] = []
    for chunk in raw.replace(";", ",").split(","):
        token = chunk.strip()
        if token:
            models.append(token)
    return models


def _expected_models() -> List[str]:
    env_models = _split_models(os.getenv("OLLAMA_MODELS"))
    if env_models:
        return env_models
    legacy = os.getenv("OLLAMA_MODEL")
    if legacy:
        return _split_models(legacy)
    return _detect_local_models()


def _base_model_name(tag: str) -> str:
    return tag.split(":", 1)[0]


def _wait_for_models(timeout: int = 900) -> None:
    expected = _expected_models()
    if not expected:
        return
    deadline = time.monotonic() + timeout
    url = "http://127.0.0.1:11435/api/tags"
    needed = {m for m in expected}
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            models = payload.get("models") or []
            available = set()
            for entry in models:
                tag = entry.get("model") or entry.get("name")
                if isinstance(tag, str):
                    available.add(tag)
                    available.add(_base_model_name(tag))
            if all((model in available) for model in needed):
                return
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit(
        "Timed out waiting for Ollama to report models: " + ", ".join(sorted(needed))
    )


def _wait_for_service(
    runner: manage_stack.ComposeRunner,
    service: str,
    *,
    timeout: int = 600,
    env: dict[str, str],
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        proc = subprocess.run(
            runner.command(["ps", "--format", "json", service]),
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                info = json.loads(proc.stdout.strip())
            except json.JSONDecodeError:
                # Compose v2 may output multiple JSON objects (one per line)
                try:
                    info = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()][0]
                except Exception:
                    info = {}
            state = info.get("State")
            health = info.get("Health")
            if state == "running" and (not health or health == "healthy"):
                return
        time.sleep(2)
    raise SystemExit(f"Timed out waiting for service '{service}' to become ready")


def compose_up(
    runner: manage_stack.ComposeRunner,
    *,
    env: dict[str, str],
    force_recreate: bool = False,
) -> None:
    args: List[str] = ["up", "-d", "--remove-orphans"]
    if force_recreate:
        args.append("--force-recreate")
    runner.run(args, env=env)


def compose_down(
    runner: manage_stack.ComposeRunner,
    *,
    env: dict[str, str],
    volumes: bool = False,
) -> None:
    args: List[str] = ["down"]
    if volumes:
        args.append("--volumes")
    runner.run(args, env=env)


def compose_logs(
    runner: manage_stack.ComposeRunner,
    services: Sequence[str],
    *,
    follow: bool,
    env: dict[str, str],
) -> int:
    args: List[str] = ["logs"]
    if follow:
        args.append("-f")
    args.extend(services)
    proc = subprocess.run(runner.command(args), env=env)
    return proc.returncode


def _interactive_exec(
    runner: manage_stack.ComposeRunner,
    agent_args: Iterable[str],
    *,
    env: dict[str, str],
) -> int:
    cmd = ["exec"]
    if sys.stdin.isatty() and sys.stdout.isatty():
        cmd.append("-it")
    else:
        cmd.append("-T")
    cmd.extend(
        [
            "dspy-agent",
            "dspy-agent",
            "--workspace",
            "/workspace",
        ]
    )
    filtered_args = list(agent_args)
    if filtered_args and filtered_args[0] == "--":
        filtered_args = filtered_args[1:]
    cmd.extend(filtered_args)
    full_cmd = runner.command(cmd)
    os.execvpe(full_cmd[0], full_cmd, env)
    return 0


def handle_start(args: argparse.Namespace) -> None:
    manage_stack.docker_available()
    manage_stack.ensure_stack_env()
    env = _compose_env(args.gpu)
    runner = manage_stack.ComposeRunner(extra_files=_extra_files(args.gpu))
    compose_up(runner, env=env, force_recreate=args.force_recreate)
    print("[dspy-code] waiting for Ollama health...")
    _wait_for_service(runner, "ollama", timeout=args.timeout, env=env)
    print("[dspy-code] waiting for Ollama models...")
    _wait_for_models(timeout=args.timeout)
    print("[dspy-code] waiting for agent container...")
    _wait_for_service(runner, "dspy-agent", timeout=args.timeout, env=env)
    if args.attach:
        print("[dspy-code] launching interactive agent session (Ctrl+C to exit)")
        _interactive_exec(runner, args.agent_args, env=env)
    else:
        print("[dspy-code] stack is running. Use 'dspy-code attach' to open an agent session or 'dspy-code logs' to tail logs.")


def handle_attach(args: argparse.Namespace) -> None:
    manage_stack.docker_available()
    manage_stack.ensure_stack_env(verbose=False)
    env = _compose_env(args.gpu)
    runner = manage_stack.ComposeRunner(extra_files=_extra_files(args.gpu))
    _wait_for_service(runner, "dspy-agent", timeout=args.timeout, env=env)
    _wait_for_models(timeout=args.timeout)
    _interactive_exec(runner, args.agent_args, env=env)


def handle_stop(args: argparse.Namespace) -> None:
    manage_stack.docker_available()
    manage_stack.ensure_stack_env(verbose=False)
    env = _compose_env()
    runner = manage_stack.ComposeRunner()
    compose_down(runner, env=env, volumes=args.volumes)
    print("[dspy-code] stack stopped")


def handle_status(_: argparse.Namespace) -> None:
    manage_stack.docker_available()
    manage_stack.ensure_stack_env(verbose=False)
    runner = manage_stack.ComposeRunner()
    subprocess.run(runner.command(["ps"]), env=_compose_env(), check=False)


def handle_logs(args: argparse.Namespace) -> None:
    manage_stack.docker_available()
    manage_stack.ensure_stack_env(verbose=False)
    runner = manage_stack.ComposeRunner()
    compose_logs(runner, args.service, follow=args.follow, env=_compose_env())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dspy-code",
        description="Convenience launcher for the DSPy lightweight agent stack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    start = sub.add_parser("start", help="start stack, wait for health, and open agent CLI")
    start.add_argument("agent_args", nargs=argparse.REMAINDER, help="arguments forwarded to dspy-agent")
    start.add_argument("--timeout", type=int, default=900, help="seconds to wait for services")
    start.add_argument("--no-attach", dest="attach", action="store_false", help="do not exec into agent after startup")
    start.add_argument("--force-recreate", action="store_true", help="force recreate containers")
    start.add_argument("--gpu", action="store_true", help="enable NVIDIA GPU support (requires container toolkit)")
    start.set_defaults(func=handle_start, attach=True, gpu=False)

    attach = sub.add_parser("attach", help="attach to a running agent container")
    attach.add_argument("agent_args", nargs=argparse.REMAINDER, help="arguments forwarded to dspy-agent")
    attach.add_argument("--timeout", type=int, default=300, help="seconds to wait for readiness before attaching")
    attach.add_argument("--gpu", action="store_true", help="use the GPU override compose file when exec-ing")
    attach.set_defaults(func=handle_attach, gpu=False)

    stop = sub.add_parser("stop", help="stop the entire stack")
    stop.add_argument("--volumes", action="store_true", help="remove volumes as well")
    stop.set_defaults(func=handle_stop)

    status = sub.add_parser("status", help="show docker compose status")
    status.set_defaults(func=handle_status)

    logs = sub.add_parser("logs", help="tail stack logs")
    logs.add_argument("service", nargs="*", help="specific services to stream")
    logs.add_argument("-f", "--follow", action="store_true", help="follow log output")
    logs.set_defaults(func=handle_logs)

    parser.set_defaults(command="start")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    if argv is None and len(sys.argv) == 1:
        args = parser.parse_args(["start"])
    else:
        args = parser.parse_args(argv)
        if args.command is None:
            args = parser.parse_args(["start"] + (argv or []))

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n[dspy-code] interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
