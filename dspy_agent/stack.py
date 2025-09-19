from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .templates.lightweight import (
    render_compose as _render_lightweight_compose,
    render_dockerfile as _render_lightweight_dockerfile,
    extra_lightweight_assets,
)


DEFAULT_STACK_DIR = Path(os.getenv("DSPY_STACK_HOME", Path.home() / ".dspy" / "stack"))


@dataclass
class StackBundle:
    root: Path
    dockerfile: Path
    compose: Path
    workspace: Path
    logs: Optional[Path]
    warnings: list[str]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def docker_available() -> bool:
    return shutil.which("docker") is not None or shutil.which("docker-compose") is not None


def prepare_stack(
    *,
    workspace: Path,
    logs: Optional[Path],
    out_dir: Path,
    db: str = "auto",
    install_source: str = "pip",
    pip_spec: Optional[str] = None,
) -> StackBundle:
    warnings: list[str] = []
    ws = workspace.expanduser().resolve()
    logs_path = logs.expanduser().resolve() if logs else None

    for target in (ws, logs_path):
        if target is None:
            continue
        if not target.exists():
            try:
                target.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # pragma: no cover - filesystem dependent
                warnings.append(f"failed to create {target}: {exc}")
        elif not target.is_dir():
            warnings.append(f"{target} is not a directory")

    ensure_dir(out_dir)

    for asset in extra_lightweight_assets():
        destination = out_dir / asset.relative_path
        ensure_dir(destination.parent)
        destination.write_text(asset.content)
        if asset.executable:
            try:
                destination.chmod(0o755)
            except Exception:
                warnings.append(f"could not mark {destination} as executable")

    install_mode = install_source.lower().strip()
    pip_spec_value = pip_spec.strip() if pip_spec else None
    if install_mode != "pip" and pip_spec_value:
        warnings.append("--pip-spec ignored when install_source is not 'pip'")
        pip_spec_value = None

    if install_mode == "local":
        src_root = Path(__file__).resolve().parents[1]
        pkg_src = src_root / "dspy_agent"
        dest_pkg = out_dir / "dspy_agent"
        if pkg_src.exists():
            if dest_pkg.exists():
                shutil.rmtree(dest_pkg)
            shutil.copytree(pkg_src, dest_pkg)
        else:
            warnings.append(f"missing package source at {pkg_src}")
        for name in ("pyproject.toml", "README.md"):
            src = src_root / name
            if src.exists():
                shutil.copy2(src, out_dir / name)
            else:
                warnings.append(f"missing {name} for local build")

    dockerfile = out_dir / "Dockerfile"
    compose = out_dir / "docker-compose.yml"
    dockerfile.write_text(_render_lightweight_dockerfile(install_source=install_mode, pip_spec=pip_spec_value))
    compose.write_text(
        _render_lightweight_compose(
            image="dspy-lightweight:latest",
            host_ws=ws,
            host_logs=logs_path,
            db_backend=db,
        )
    )

    return StackBundle(
        root=out_dir,
        dockerfile=dockerfile,
        compose=compose,
        workspace=ws,
        logs=logs_path,
        warnings=warnings,
    )


def compose_command(
    compose_file: Path,
    args: Sequence[str],
    *,
    check: bool = True,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    commands: Iterable[Sequence[str]] = (
        ("docker", "compose"),
        ("docker-compose",),
    )
    last_error: Optional[Exception] = None
    for base in commands:
        cmd = list(base) + ["-f", str(compose_file)] + list(args)
        try:
            return subprocess.run(cmd, cwd=str(cwd or compose_file.parent), check=check)
        except FileNotFoundError as exc:  # pragma: no cover - depends on environment
            last_error = exc
            continue
    raise RuntimeError("docker compose not available") from last_error
