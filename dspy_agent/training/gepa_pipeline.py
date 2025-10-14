from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from dspy_agent.infra.grpc_client import orchestrator_client


@dataclass
class GEPAModuleConfig:
    name: str
    train_jsonl: Path
    val_jsonl: Optional[Path] = None
    test_jsonl: Optional[Path] = None


class GEPATrainingPipeline:
    def __init__(
        self,
        *,
        orchestrator_addr: Optional[str] = None,
        task_class: str = "cpu_long",
        tenant: str = "gepa",
        priority: int = 0,
        timeout_seconds: int = 1800,
    ) -> None:
        self.orchestrator_addr = orchestrator_addr or os.getenv("ORCHESTRATOR_GRPC_ADDR", "127.0.0.1:50052")
        self.task_class = task_class
        self.tenant = tenant
        self.priority = priority
        self.timeout_seconds = timeout_seconds

    def _default_working_dir(self, task_id: str) -> str:
        base = os.getenv("DSPY_TASK_BASE", "/tmp/dspy-tasks")
        return str(Path(base) / self.tenant / task_id)

    def _build_command(
        self,
        module: GEPAModuleConfig,
        working_dir: Path,
        auto: Optional[str],
        workspace: Optional[Path],
        extra_args: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        outputs_dir = working_dir / "outputs"
        logs_dir = working_dir / "logs"
        result_path = outputs_dir / "gepa_result.json"
        progress_path = logs_dir / "progress.jsonl"

        cmd: List[str] = [
            os.getenv("PYTHON_BIN", "python3"),
            "-m",
            "dspy_agent.training.gepa_task_cli",
            "--module",
            module.name,
            "--train-jsonl",
            str(module.train_jsonl),
            "--output-json",
            str(result_path),
            "--progress-path",
            str(progress_path),
            "--log-dir",
            str(logs_dir),
            "--timeout-seconds",
            str(self.timeout_seconds),
        ]
        if module.val_jsonl:
            cmd.extend(["--val-jsonl", str(module.val_jsonl)])
        if module.test_jsonl:
            cmd.extend(["--test-jsonl", str(module.test_jsonl)])
        if workspace:
            cmd.extend(["--workspace", str(workspace)])
        if auto:
            cmd.extend(["--auto", str(auto)])
        if extra_args:
            for key, value in extra_args.items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    if value:
                        cmd.append(str(key))
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        cmd.append(str(key))
                        cmd.append(str(item))
                else:
                    cmd.append(str(key))
                    cmd.append(str(value))

        env = {}
        if "PYTHONPATH" in os.environ:
            env["PYTHONPATH"] = os.environ["PYTHONPATH"]
        return {
            "execution": {
                "command": cmd,
                "env": env,
                "timeout_seconds": self.timeout_seconds,
                "working_dir": str(working_dir),
            },
            "artifacts": {
                "outputs": [
                    {"path": str(result_path), "type": "json", "name": f"{module.name}_result"},
                    {"path": str(progress_path), "type": "jsonl", "name": f"{module.name}_progress"},
                ]
            },
            "tenant": self.tenant,
        }

    async def _wait_for_completion(self, client, task_id: str, interval: float = 5.0) -> Dict[str, Any]:
        while True:
            status = await client.get_task_status(task_id)
            state = status.get("status")
            if state in {"completed", "failed"}:
                return status
            await asyncio.sleep(interval)

    async def train_modules(
        self,
        modules: Iterable[GEPAModuleConfig],
        *,
        auto: Optional[str] = "light",
        workspace: Optional[Path] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        async with orchestrator_client(self.orchestrator_addr) as client:
            for module in modules:
                task_id = f"gepa-{module.name}-{uuid4().hex[:8]}"
                working_dir = Path(self._default_working_dir(task_id))
                payload = self._build_command(module, working_dir, auto, workspace, extra_args)
                response = await client.submit_task(
                    task_id=task_id,
                    task_class=self.task_class,
                    payload=payload,
                    priority=self.priority,
                )
                if not response.get("success", False):
                    raise RuntimeError(f"Failed to submit GEPA task {task_id}: {response.get('error')}")

                status = await self._wait_for_completion(client, task_id)
                results.append(self._decode_result(module.name, status))
        return results

    def _decode_result(self, module_name: str, status: Dict[str, Any]) -> Dict[str, Any]:
        metadata_raw = status.get("result", {})
        metadata_blob: Dict[str, Any] = {}
        if isinstance(metadata_raw, dict):
            candidate = metadata_raw.get("metadata")
            if isinstance(candidate, str):
                try:
                    metadata_blob = json.loads(candidate)
                except Exception:
                    metadata_blob = {"metadata_json": candidate}
            elif isinstance(candidate, dict):
                metadata_blob = candidate

        outputs_summary = {}
        if isinstance(metadata_blob, dict):
            outputs_summary = metadata_blob.get("outputs_summary", {}) or {}
        stdout_tail = ""
        logs = metadata_blob.get("logs") if isinstance(metadata_blob, dict) else {}
        if isinstance(logs, dict):
            stdout_tail = logs.get("stdout_tail", "") or ""
        parsed_marker = {}
        if stdout_tail:
            parsed_marker = _extract_result_marker(stdout_tail)

        decoded = {
            "module": module_name,
            "status": status.get("status"),
            "result": metadata_raw,
            "metadata": metadata_blob,
            "stdout_marker": parsed_marker,
            "completed_at": status.get("completed_at"),
            "error": status.get("error"),
            "outputs_summary": outputs_summary,
        }
        return decoded


def _extract_result_marker(stdout_tail: str) -> Dict[str, Any]:
    marker_prefix = "__DSPY_RESULT__:"
    for line in stdout_tail.splitlines()[::-1]:
        line = line.strip()
        if line.startswith(marker_prefix):
            payload = line[len(marker_prefix) :].strip()
            try:
                return json.loads(payload)
            except Exception:
                return {"raw": payload}
    return {}


async def run_gepa_pipeline(
    modules: Iterable[GEPAModuleConfig],
    *,
    orchestrator_addr: Optional[str] = None,
    task_class: str = "cpu_long",
    tenant: str = "gepa",
    priority: int = 0,
    auto: Optional[str] = "light",
    workspace: Optional[Path] = None,
    extra_args: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    pipeline = GEPATrainingPipeline(
        orchestrator_addr=orchestrator_addr,
        task_class=task_class,
        tenant=tenant,
        priority=priority,
    )
    return await pipeline.train_modules(
        modules,
        auto=auto,
        workspace=workspace,
        extra_args=extra_args,
    )
