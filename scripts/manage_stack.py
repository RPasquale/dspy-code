#!/usr/bin/env python3
"""Cross-platform helper for building and running the DSPy lightweight stack."""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parent.parent
STACK_COMPOSE = ROOT / "docker" / "lightweight" / "docker-compose.yml"
STACK_GPU_COMPOSE = ROOT / "docker" / "lightweight" / "docker-compose.gpu.yml"
STACK_ENV = ROOT / "docker" / "lightweight" / ".env"


def _host_ids() -> tuple[str, str]:
    """Return the host UID/GID as strings (fallback to 1000 on non-POSIX)."""

    try:
        uid = str(os.getuid())  # type: ignore[attr-defined]
        gid = str(os.getgid())  # type: ignore[attr-defined]
    except AttributeError:
        uid = gid = "1000"
    return uid, gid


class ComposeRunner:
    """Wrapper that finds an available docker compose command."""

    def __init__(self, *, extra_files: Iterable[Path] | None = None) -> None:
        self._base_cmd = self._detect_compose()
        self._extra_files = [Path(f).resolve() for f in extra_files or []]

    @staticmethod
    def _detect_compose() -> List[str]:
        candidates = (["docker", "compose"], ["docker-compose"])
        for base in candidates:
            try:
                subprocess.run(
                    base + ["version"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return list(base)
            except (OSError, subprocess.CalledProcessError):
                continue
        raise SystemExit(
            "Unable to find 'docker compose' or 'docker-compose'. "
            "Please install Docker Desktop or the Docker CLI before continuing."
        )

    def command(self, compose_args: Iterable[str]) -> List[str]:
        cmd: List[str] = [*self._base_cmd, "-f", str(STACK_COMPOSE)]
        for extra in self._extra_files:
            cmd.extend(["-f", str(extra)])
        cmd.extend(["--env-file", str(STACK_ENV)])
        cmd.extend(list(compose_args))
        return cmd

    def run(
        self,
        compose_args: Iterable[str],
        *,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> int:
        cmd = self.command(compose_args)
        return subprocess.run(cmd, check=check, env=env).returncode


def ensure_stack_env(verbose: bool = True) -> None:
    """Ensure docker/lightweight/.env exists (mirrors `make stack-env`)."""
    uid, gid = _host_ids()

    if STACK_ENV.exists():
        content = STACK_ENV.read_text(encoding="utf-8")
        changed = False
        if "DSPY_UID=" not in content:
            content += f"DSPY_UID={uid}\n"
            changed = True
        if "DSPY_GID=" not in content:
            content += f"DSPY_GID={gid}\n"
            changed = True
        if changed:
            STACK_ENV.write_text(content, encoding="utf-8")
            if verbose:
                print(f"[manage-stack] updated {STACK_ENV.relative_to(ROOT)} with host UID/GID")
        return

    WORKSPACE_DIR = str(ROOT)
    token = secrets.token_hex(32)
    content = (
        f"WORKSPACE_DIR={WORKSPACE_DIR}\n"
        f"DSPY_UID={uid}\n"
        f"DSPY_GID={gid}\n"
        "# RedDB Configuration\n"
        f"REDDB_ADMIN_TOKEN={token}\n"
        "REDDB_URL=http://reddb:8080\n"
        "REDDB_NAMESPACE=dspy\n"
        f"REDDB_TOKEN={token}\n"
        "DB_BACKEND=reddb\n"
    )
    STACK_ENV.write_text(content, encoding="utf-8")
    if verbose:
        print(f"[manage-stack] wrote {STACK_ENV.relative_to(ROOT)}")


def docker_available() -> None:
    if shutil.which("docker") is None:
        raise SystemExit(
            "Docker CLI not found in PATH. Install Docker Desktop / Engine before running the stack."
        )


def cmd_init(_: argparse.Namespace) -> None:
    docker_available()
    ensure_stack_env()
    print("[manage-stack] stack environment ready")


def cmd_build(args: argparse.Namespace) -> None:
    docker_available()
    ensure_stack_env()
    runner = ComposeRunner()
    compose_args = ["build"]
    if args.service:
        compose_args.extend(args.service)
    runner.run(compose_args)


def cmd_up(args: argparse.Namespace) -> None:
    docker_available()
    ensure_stack_env()
    runner = ComposeRunner()
    compose_args = ["up", "-d", "--remove-orphans"]
    if args.force_recreate:
        compose_args.append("--force-recreate")
    if args.service:
        compose_args.extend(args.service)
    runner.run(compose_args)


def cmd_down(args: argparse.Namespace) -> None:
    docker_available()
    ensure_stack_env(verbose=False)
    runner = ComposeRunner()
    compose_args = ["down"]
    if args.volumes:
        compose_args.append("--volumes")
    runner.run(compose_args)


def cmd_restart(args: argparse.Namespace) -> None:
    docker_available()
    ensure_stack_env()
    runner = ComposeRunner()
    compose_args = ["up", "-d", "--remove-orphans", "--force-recreate"]
    if args.service:
        compose_args.extend(args.service)
    runner.run(compose_args)


def cmd_ps(_: argparse.Namespace) -> None:
    docker_available()
    ensure_stack_env(verbose=False)
    runner = ComposeRunner()
    runner.run(["ps"])


def cmd_logs(args: argparse.Namespace) -> None:
    docker_available()
    ensure_stack_env(verbose=False)
    runner = ComposeRunner()
    compose_args = ["logs"]
    if args.follow:
        compose_args.append("-f")
    if args.service:
        compose_args.extend(args.service)
    runner.run(compose_args)


def cmd_pull(args: argparse.Namespace) -> None:
    docker_available()
    ensure_stack_env()
    runner = ComposeRunner()
    compose_args = ["pull"]
    if args.service:
        compose_args.extend(args.service)
    runner.run(compose_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the DSPy lightweight docker stack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub_init = sub.add_parser("init", help="create the docker .env file if missing")
    sub_init.set_defaults(func=cmd_init)

    sub_build = sub.add_parser("build", help="build stack images")
    sub_build.add_argument("service", nargs="*", help="optional list of services to build")
    sub_build.set_defaults(func=cmd_build)

    sub_up = sub.add_parser("up", help="start the stack (docker compose up -d)")
    sub_up.add_argument("service", nargs="*", help="optional services to start")
    sub_up.add_argument(
        "--force-recreate", action="store_true", help="force recreate containers"
    )
    sub_up.set_defaults(func=cmd_up)

    sub_down = sub.add_parser("down", help="stop the stack")
    sub_down.add_argument("--volumes", action="store_true", help="also remove volumes")
    sub_down.set_defaults(func=cmd_down)

    sub_restart = sub.add_parser("restart", help="recreate containers in place")
    sub_restart.add_argument("service", nargs="*", help="optional services to restart")
    sub_restart.set_defaults(func=cmd_restart)

    sub_ps = sub.add_parser("ps", help="show container status")
    sub_ps.set_defaults(func=cmd_ps)

    sub_logs = sub.add_parser("logs", help="stream logs")
    sub_logs.add_argument("service", nargs="*", help="optional service names")
    sub_logs.add_argument("-f", "--follow", action="store_true", help="follow output")
    sub_logs.set_defaults(func=cmd_logs)

    sub_pull = sub.add_parser("pull", help="pull images used by the stack")
    sub_pull.add_argument("service", nargs="*", help="optional service names")
    sub_pull.set_defaults(func=cmd_pull)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
