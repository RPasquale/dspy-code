from __future__ import annotations

import typer

# Thin wrapper to expose RL sub-app from the main CLI without duplicating code.
try:
    from .cli import rl_app as rl_group  # type: ignore
except Exception:
    # Fallback to an empty group if main CLI import fails
    rl_group = typer.Typer(no_args_is_help=True, help="RL commands")

