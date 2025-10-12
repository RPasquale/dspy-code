"""Unified infrastructure management for the DSPy agent."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from .grpc_client import OrchestratorClient, DEFAULT_ORCHESTRATOR_ADDR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

CLI_ENV_SKIP_START = "DSPY_AGENT_SKIP_START"
CLI_ENV_PATH = "DSPY_AGENT_CLI_PATH"
ORCHESTRATOR_ADDR_ENV = "ORCHESTRATOR_GRPC_ADDR"
ENV_MANAGER_ADDR_ENV = "ENV_MANAGER_GRPC_ADDR"

DEFAULT_CLI_CANDIDATES = (
    "dspy-agent",
    "./dspy-agent",
    "./cmd/dspy-agent/dspy-agent",
    "../cmd/dspy-agent/dspy-agent",
)

SERVICE_WAIT_TIMEOUT = float(os.getenv("DSPY_AGENT_SERVICE_TIMEOUT", "60"))
RECONNECT_BACKOFF = float(os.getenv("DSPY_AGENT_RECONNECT_BACKOFF", "2.0"))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Agent infrastructure
# ---------------------------------------------------------------------------


class AgentInfra:
    """Lifecycle manager for the Rust/Go powered infrastructure stack."""

    def __init__(
        self,
        orchestrator_addr: Optional[str] = None,
        workspace: Optional[Path] = None,
        auto_start_services: bool = True,
        cli_path: Optional[Path] = None,
    ) -> None:
        self.orchestrator_addr = orchestrator_addr or DEFAULT_ORCHESTRATOR_ADDR
        self.workspace = workspace or Path.cwd()
        self.auto_start_services = auto_start_services
        self.cli_path = cli_path or self._discover_cli_binary()

        self.orchestrator: Optional[OrchestratorClient] = None
        self._started = False
        self._bootstrapped_process = False

    # ------------------------------------------------------------------
    async def start(self) -> "AgentInfra":
        if self._started:
            logger.warning("Infrastructure already started")
            return self

        logger.info("Initialising DSPy infrastructure via orchestrator at %s", self.orchestrator_addr)
        self.orchestrator = OrchestratorClient(self.orchestrator_addr)

        try:
            await self.orchestrator.connect()
        except Exception as exc:
            logger.warning("Initial orchestrator connection failed: %s", exc)
            if _env_bool(CLI_ENV_SKIP_START):
                raise RuntimeError(
                    "Unable to connect to orchestrator and auto-start is disabled. "
                    f"Unset {CLI_ENV_SKIP_START} or set it to 0 to allow automatic startup."
                ) from exc
            await self._start_cli_daemon()
            await asyncio.sleep(RECONNECT_BACKOFF)
            await self.orchestrator.connect()

        await self._wait_for_health()

        if self.auto_start_services:
            await self._verify_services()

        self._started = True
        logger.info("Infrastructure ready")
        return self

    async def stop(self) -> None:
        if not self._started:
            return

        logger.info("Shutting down infrastructure")
        if self.orchestrator:
            try:
                await self.orchestrator.close()
            finally:
                self.orchestrator = None

        if self._bootstrapped_process and self.cli_path:
            try:
                subprocess.run([str(self.cli_path), "stop"], check=False)
            except Exception as exc:
                logger.warning("Failed to stop infrastructure via CLI: %s", exc)

        self._started = False

    # ------------------------------------------------------------------
    async def submit_task(
        self,
        task_id: str,
        payload: Dict[str, Any],
        task_class: str = "cpu_short",
        priority: int = 0,
    ) -> Dict[str, Any]:
        if not self._started or not self.orchestrator:
            raise RuntimeError("Infrastructure not started")

        str_payload = {k: str(v) for k, v in payload.items()}
        return await self.orchestrator.submit_task(
            task_id=task_id,
            task_class=task_class,
            payload=str_payload,
            priority=priority,
        )

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        if not self._started or not self.orchestrator:
            raise RuntimeError("Infrastructure not started")
        if not hasattr(self.orchestrator, "get_task_status"):
            raise NotImplementedError("Orchestrator client lacks get_task_status RPC")
        return await self.orchestrator.get_task_status(task_id)  # type: ignore[attr-defined]

    async def get_metrics(self) -> Dict[str, Any]:
        if not self._started or not self.orchestrator:
            raise RuntimeError("Infrastructure not started")
        return await self.orchestrator.get_metrics()

    async def health_check(self) -> Dict[str, Any]:
        if not self.orchestrator:
            return {"healthy": False, "error": "Not connected"}
        return await self.orchestrator.health_check()

    # ------------------------------------------------------------------
    @classmethod
    @asynccontextmanager
    async def start_context(
        cls,
        orchestrator_addr: Optional[str] = None,
        workspace: Optional[Path] = None,
        auto_start_services: bool = True,
        cli_path: Optional[Path] = None,
    ):
        infra = cls(
            orchestrator_addr=orchestrator_addr,
            workspace=workspace,
            auto_start_services=auto_start_services,
            cli_path=cli_path,
        )
        await infra.start()
        try:
            yield infra
        finally:
            await infra.stop()

    @classmethod
    def start(
        cls,
        orchestrator_addr: Optional[str] = None,
        workspace: Optional[Path] = None,
        auto_start_services: bool = True,
        cli_path: Optional[Path] = None,
    ):
        return cls.start_context(
            orchestrator_addr=orchestrator_addr,
            workspace=workspace,
            auto_start_services=auto_start_services,
            cli_path=cli_path,
        )

    # ------------------------------------------------------------------
    def _discover_cli_binary(self) -> Optional[Path]:
        explicit = os.getenv(CLI_ENV_PATH)
        candidates = []
        if explicit:
            candidates.append(Path(explicit))
        candidates.extend((self.workspace / name) for name in DEFAULT_CLI_CANDIDATES)
        candidates.extend(Path(name) for name in DEFAULT_CLI_CANDIDATES)
        for candidate in candidates:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate
        resolved = shutil.which("dspy-agent")
        return Path(resolved) if resolved else None

    async def _start_cli_daemon(self) -> None:
        if not self.cli_path:
            raise RuntimeError(
                "dspy-agent CLI not found. Set DSPY_AGENT_CLI_PATH or install the CLI binary."
            )
        logger.info("Launching unified CLI via %s", self.cli_path)
        env = os.environ.copy()
        env.setdefault(ORCHESTRATOR_ADDR_ENV, self.orchestrator_addr)
        env.setdefault(ENV_MANAGER_ADDR_ENV, env.get(ENV_MANAGER_ADDR_ENV, "127.0.0.1:50051"))
        process = subprocess.run(
            [str(self.cli_path), "start"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.returncode != 0:
            logger.error("dspy-agent start failed: %s", process.stderr.strip())
            raise RuntimeError(f"Failed to start infrastructure: {process.stderr.strip()}")
        logger.info("dspy-agent start completed")
        self._bootstrapped_process = True

    async def _wait_for_health(self) -> None:
        assert self.orchestrator is not None
        deadline = asyncio.get_event_loop().time() + SERVICE_WAIT_TIMEOUT
        while True:
            try:
                health = await self.orchestrator.health_check()
                if health.get("healthy"):
                    logger.info("Orchestrator healthy (version=%s)", health.get("version"))
                    return
            except Exception as exc:
                logger.debug("Health probe failed: %s", exc)
            if asyncio.get_event_loop().time() >= deadline:
                raise RuntimeError("Timed out waiting for orchestrator health check")
            await asyncio.sleep(RECONNECT_BACKOFF)

    async def _verify_services(self) -> None:
        logger.info("Verifying dependent services via orchestrator metrics")
        try:
            metrics = await asyncio.wait_for(self.get_metrics(), timeout=SERVICE_WAIT_TIMEOUT)
            if not metrics:
                logger.warning("Orchestrator returned no metrics; services may still be starting")
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for orchestrator metrics")
        except Exception as exc:
            logger.warning("Could not verify services: %s", exc)
