"""
Shared runtime helpers for accessing the Rust/Go infrastructure from Python.

The helper functions in this module centralise the lifecycle handling of
``AgentInfra`` so that different subsystems (streaming, RL, Spark jobs, …)
can reuse a single gRPC connection instead of standing up their own copies.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import AsyncIterator, Optional, Union

from .agent_infra import AgentInfra

_infra: Optional[AgentInfra] = None
_infra_lock = asyncio.Lock()


async def ensure_infra(
    auto_start_services: bool = True,
    *,
    orchestrator_addr: Optional[str] = None,
    workspace: Optional[Union[Path, str]] = None,
    cli_path: Optional[Union[Path, str]] = None,
) -> AgentInfra:
    """
    Ensure the shared ``AgentInfra`` instance is running and return it.

    Parameters
    ----------
    auto_start_services:
        When True (default) the helper will attempt to start the unified dspy
        stack if it cannot reach the orchestrator on the first try.
    """
    global _infra
    if _infra and getattr(_infra, "_started", False):
        return _infra

    async with _infra_lock:
        if _infra and getattr(_infra, "_started", False):
            return _infra

        resolved_workspace = Path(workspace) if isinstance(workspace, (str, Path)) else None
        resolved_cli_path = Path(cli_path) if isinstance(cli_path, (str, Path)) else None
        infra = AgentInfra(
            orchestrator_addr=orchestrator_addr,
            workspace=resolved_workspace,
            auto_start_services=auto_start_services,
            cli_path=resolved_cli_path,
        )
        await infra.start_async()
        _infra = infra
        return infra


def ensure_infra_sync(
    auto_start_services: bool = True,
    *,
    orchestrator_addr: Optional[str] = None,
    workspace: Optional[Union[Path, str]] = None,
    cli_path: Optional[Union[Path, str]] = None,
) -> AgentInfra:
    """
    Synchronous helper that wraps :func:`ensure_infra`.

    This is useful for legacy code paths that are still synchronous (for
    example CLI entrypoints).  If the caller is already inside an event loop,
    they must switch to ``await ensure_infra(...)`` instead to avoid deadlocks.
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            ensure_infra(
                auto_start_services=auto_start_services,
                orchestrator_addr=orchestrator_addr,
                workspace=workspace,
                cli_path=cli_path,
            )
        )
    raise RuntimeError(
        "ensure_infra_sync() cannot be invoked from within a running event loop; "
        "use `await ensure_infra(...)` instead."
    )


async def shutdown_infra() -> None:
    """Stop the shared ``AgentInfra`` instance if it is running."""
    global _infra
    if not _infra:
        return
    try:
        await _infra.stop()
    finally:
        _infra = None


@contextlib.asynccontextmanager
async def use_infra(
    auto_start_services: bool = True,
    *,
    orchestrator_addr: Optional[str] = None,
    workspace: Optional[Union[Path, str]] = None,
    cli_path: Optional[Union[Path, str]] = None,
) -> AsyncIterator[AgentInfra]:
    """
    Async context manager that yields the shared ``AgentInfra``.

    Unlike ``AgentInfra.start()``, this keeps the global instance alive when the
    context exits so other subsystems can reuse the already-initialised gRPC
    connections.
    """
    infra = await ensure_infra(
        auto_start_services=auto_start_services,
        orchestrator_addr=orchestrator_addr,
        workspace=workspace,
        cli_path=cli_path,
    )
    try:
        yield infra
    finally:
        # keep infra warm – callers that truly want to shut it down can call
        # ``shutdown_infra`` explicitly.
        pass
