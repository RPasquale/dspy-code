"""
Go Orchestrator Integration for RL Training

This module integrates the Go orchestrator with the RL training system
for optimal coordination, resource management, and performance monitoring.
"""

import asyncio
import json
import time
import logging
import os
import subprocess
import threading
from pathlib import Path
from uuid import uuid4
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass
import aiohttp
import numpy as np

# Local imports
from .universal_pufferlib import UniversalPufferConfig
from .rust_rl_runner import RustRLConfig, OptimizedRLTrainer
from .rl_tracking import get_rl_tracker
from ..streaming.streamkit import LocalBus
from ..infra import OrchestratorClient, ensure_infra
from ..infra.agent_infra import DEFAULT_ORCHESTRATOR_ADDR

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from ..infra import AgentInfra


@dataclass
class GoOrchestratorConfig:
    """Configuration for Go orchestrator integration"""
    
    # Go orchestrator settings
    orchestrator_path: str = "orchestrator"
    orchestrator_port: int = 8080
    orchestrator_host: str = "localhost"
    orchestrator_addr: str = DEFAULT_ORCHESTRATOR_ADDR
    auto_start_services: bool = True
    workspace: Optional[Path] = None
    cli_path: Optional[Path] = None
    task_class: str = "cpu_short"
    poll_interval: float = 1.0
    grpc_timeout: float = 600.0
    
    # Resource management
    base_limit: int = 4
    min_limit: int = 1
    max_limit: int = 32
    increase_step: int = 2
    decrease_step: int = 1
    
    # Performance thresholds
    queue_high_watermark: float = 0.8
    gpu_wait_high: float = 5.0
    error_rate_high: float = 0.1
    adaptation_interval: float = 30.0  # seconds
    
    # RL training coordination
    rl_task_priority: int = 10  # High priority for RL tasks
    batch_size: int = 64
    max_concurrent_tasks: int = 16
    
    # Monitoring
    metrics_enabled: bool = True
    health_check_interval: float = 10.0
    performance_log_interval: float = 60.0


class GoOrchestratorClient:
    """Client for Go orchestrator coordination"""
    
    def __init__(self, config: GoOrchestratorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Preferred mode: try gRPC first, fall back to legacy HTTP on demand.
        self.mode = "grpc"
        if os.getenv("DSPY_ORCH_FORCE_HTTP"):
            self.mode = "http"

        # gRPC infrastructure state
        self._infra: Optional["AgentInfra"] = None
        self._orchestrator: Optional[OrchestratorClient] = None
        self._start_lock: Optional[asyncio.Lock] = None
        self._result_cache: Dict[str, Dict[str, Any]] = {}

        # Legacy HTTP state
        self.orchestrator_process: Optional[subprocess.Popen] = None
        self.orchestrator_thread: Optional[threading.Thread] = None

        # Performance metrics
        self.metrics = {
            'queue_depth': 0,
            'gpu_wait_time': 0.0,
            'error_rate': 0.0,
            'concurrency_limit': self.config.base_limit,
            'active_tasks': 0,
            'raw': {},
        }

        if self.mode == "http":
            self._start_http_orchestrator()
    
    def _start_http_orchestrator(self):
        """Start the legacy HTTP orchestrator process."""
        try:
            # Check if Go orchestrator exists
            if not os.path.exists(self.config.orchestrator_path):
                self.logger.warning(
                    "Go orchestrator not found at %s, falling back to Python implementation",
                    self.config.orchestrator_path,
                )
                self.orchestrator_process = None
                return
            
            # Start orchestrator process
            self.orchestrator_process = subprocess.Popen(
                [self.config.orchestrator_path],
                env={
                    **os.environ,
                    "ORCHESTRATOR_PORT": str(self.config.orchestrator_port),
                    "BASE_LIMIT": str(self.config.base_limit),
                    "MIN_LIMIT": str(self.config.min_limit),
                    "MAX_LIMIT": str(self.config.max_limit)
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.logger.info("Started Go orchestrator (PID: %s)", self.orchestrator_process.pid)
            
            # Start metrics collection thread
            self.orchestrator_thread = threading.Thread(target=self._collect_metrics)
            self.orchestrator_thread.daemon = True
            self.orchestrator_thread.start()
            
        except Exception as e:
            self.logger.warning(
                "Failed to start Go orchestrator: %s, falling back to Python implementation",
                e,
            )
            self.orchestrator_process = None
    
    def _collect_metrics(self):
        """Collect metrics from Go orchestrator"""
        while self.orchestrator_process and self.orchestrator_process.poll() is None:
            try:
                # Query orchestrator metrics
                import requests
                response = requests.get(
                    f"http://{self.config.orchestrator_host}:{self.config.orchestrator_port}/metrics",
                    timeout=5,
                )
                if response.status_code == 200:
                    metrics_data = response.json()
                    if isinstance(metrics_data, dict):
                        self.metrics['raw'] = metrics_data
                        if 'queue_depth' in metrics_data:
                            self.metrics['queue_depth'] = metrics_data.get('queue_depth', self.metrics['queue_depth'])
                        if 'gpu_wait_time' in metrics_data:
                            self.metrics['gpu_wait_time'] = metrics_data.get('gpu_wait_time', self.metrics['gpu_wait_time'])
                        if 'error_rate' in metrics_data:
                            self.metrics['error_rate'] = metrics_data.get('error_rate', self.metrics['error_rate'])
                        if 'concurrency_limit' in metrics_data:
                            self.metrics['concurrency_limit'] = metrics_data.get('concurrency_limit', self.metrics['concurrency_limit'])
                        if 'active_tasks' in metrics_data:
                            self.metrics['active_tasks'] = metrics_data.get('active_tasks', self.metrics['active_tasks'])
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.debug("Failed to collect legacy orchestrator metrics: %s", e)
                time.sleep(self.config.health_check_interval)

    async def _ensure_infra(self) -> None:
        """Ensure the gRPC infrastructure is started and ready."""
        if self.mode != "grpc":
            return
        if self._orchestrator is not None:
            return
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            if self._orchestrator is not None:
                return
            try:
                workspace: Optional[Path]
                if self.config.workspace is None:
                    workspace = None
                elif isinstance(self.config.workspace, Path):
                    workspace = self.config.workspace
                else:
                    workspace = Path(self.config.workspace)
                cli_path = self.config.cli_path
                infra = await ensure_infra(
                    auto_start_services=self.config.auto_start_services,
                    orchestrator_addr=self.config.orchestrator_addr,
                    workspace=workspace,
                    cli_path=cli_path,
                )
                self._infra = infra
                self._orchestrator = infra.orchestrator
                if not self._orchestrator:
                    # As a safety net create a direct client so subsequent calls still work.
                    self._orchestrator = OrchestratorClient(self.config.orchestrator_addr)
                    await self._orchestrator.connect()
                try:
                    health = await self._orchestrator.health_check()
                    if not health.get("healthy", True):
                        self.logger.warning("Orchestrator health probe reported unhealthy status: %s", health)
                except Exception as exc:
                    self.logger.debug("Orchestrator health probe failed: %s", exc)
            except Exception as exc:
                self.logger.warning("Falling back to legacy HTTP orchestrator: %s", exc)
                self.mode = "http"
                self._infra = None
                self._orchestrator = None
                self._start_http_orchestrator()

    def _build_payload(self, task: Dict[str, Any]) -> Dict[str, str]:
        payload: Dict[str, str] = {
            "task_type": str(task.get("task_type", "rl_episode")),
            "session_id": str(task.get("session_id", "")),
            "episode_id": str(task.get("episode_id", "")),
            "timestamp": str(task.get("timestamp", time.time())),
        }
        config_data = task.get("config") or {}
        try:
            payload["config_json"] = json.dumps(config_data)
        except Exception:
            payload["config_json"] = json.dumps({})
        extras = task.get("extras") or {}
        if isinstance(extras, dict) and extras:
            payload["extras_json"] = json.dumps(extras)
        for key, value in task.items():
            if key in {"config", "extras", "task_type", "session_id", "episode_id", "timestamp"}:
                continue
            payload[key] = str(value)
        return payload

    def _decode_result_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        result_payload = dict(update.get("result") or {})
        data = result_payload.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                data = {}
        elif data is None and "data_json" in result_payload:
            try:
                data = json.loads(result_payload["data_json"])
            except Exception:
                data = {}
        elif isinstance(data, dict):
            data = dict(data)
        else:
            data = {}
        result_payload["data"] = data
        decoded = {
            "task_id": update.get("task_id"),
            "status": update.get("status"),
            "result": result_payload,
            "error": update.get("error"),
            "duration_ms": update.get("duration_ms"),
            "completed_at": update.get("completed_at"),
        }
        return decoded

    def _update_metrics_from_grpc(self, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        try:
            if "tasks_pending" in metrics:
                self.metrics["queue_depth"] = int(metrics["tasks_pending"])
            if "tasks_running" in metrics:
                self.metrics["active_tasks"] = int(metrics["tasks_running"])
            if "task_error_rate" in metrics:
                self.metrics["error_rate"] = float(metrics["task_error_rate"])
            if "gpu_wait_seconds" in metrics:
                self.metrics["gpu_wait_time"] = float(metrics["gpu_wait_seconds"])
            if "concurrency_limit" in metrics:
                self.metrics["concurrency_limit"] = int(metrics["concurrency_limit"])
            self.metrics["raw"] = metrics
        except Exception as exc:
            self.logger.debug("Failed to normalise orchestrator metrics: %s", exc)

    def _format_http_result(self, raw: Dict[str, Any], default_task_id: str) -> Dict[str, Any]:
        update = {
            "task_id": raw.get("task_id", default_task_id),
            "status": raw.get("status"),
            "result": raw.get("result") or raw.get("data") or {},
            "error": raw.get("error"),
            "duration_ms": raw.get("duration_ms"),
            "completed_at": raw.get("completed_at"),
        }
        decoded = self._decode_result_update(update)
        decoded["legacy_raw"] = raw
        return decoded

    async def _submit_rl_task_grpc(self, task: Dict[str, Any]) -> str:
        await self._ensure_infra()
        if not self._orchestrator:
            raise RuntimeError("Orchestrator unavailable")
        task_id = str(task.get("episode_id") or task.get("task_id") or f"rl-task-{uuid4().hex}")
        priority = int(task.get("priority", self.config.rl_task_priority))
        payload = self._build_payload(task)
        response = await self._orchestrator.submit_task(
            task_id=task_id,
            task_class=self.config.task_class,
            payload=payload,
            priority=priority,
        )
        if not response.get("success", False):
            raise RuntimeError(f"Failed to submit task {task_id}: {response.get('error')}")
        submitted_id = response.get("task_id") or task_id
        self.logger.info("Submitted RL task via gRPC: %s", submitted_id)
        return submitted_id

    async def _submit_rl_task_http(self, task: Dict[str, Any]) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{self.config.orchestrator_host}:{self.config.orchestrator_port}/tasks",
                    json=task,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        task_id = result.get("task_id")
                        self.logger.info("Submitted RL task: %s", task_id)
                        return task_id  # type: ignore[return-value]
                    raise RuntimeError(f"Failed to submit task: {response.status}")
        except Exception as exc:
            self.logger.error(f"Failed to submit RL task: {exc}")
            raise

    async def _wait_for_task_completion_http(self, task_id: str, timeout: int) -> Dict[str, Any]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{self.config.orchestrator_host}:{self.config.orchestrator_port}/tasks/{task_id}"
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            status = result.get("status")
                            if status == "completed":
                                return result
                            if status == "failed":
                                raise RuntimeError(f"Task failed: {result.get('error')}")
                await asyncio.sleep(max(self.config.poll_interval, 0.5))
            except Exception as exc:
                self.logger.error(f"Error waiting for task completion: {exc}")
                await asyncio.sleep(max(self.config.poll_interval, 0.5))
        raise TimeoutError(f"Timeout waiting for task {task_id}")

    async def _wait_for_batch_completion_http(self, task_ids: List[str], timeout: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for task_id in task_ids:
            try:
                raw = await self._wait_for_task_completion_http(task_id, timeout)
                results.append(self._format_http_result(raw, task_id))
            except Exception as exc:
                self.logger.error(f"Task {task_id} failed: {exc}")
                results.append({
                    "task_id": task_id,
                    "status": "failed",
                    "result": {},
                    "error": str(exc),
                    "duration_ms": None,
                    "completed_at": None,
                })
        return results

    async def _wait_for_batch_completion_grpc(self, task_ids: List[str], timeout: int) -> List[Dict[str, Any]]:
        await self._ensure_infra()
        if not self._orchestrator:
            return [{
                "task_id": tid,
                "status": "failed",
                "result": {},
                "error": "orchestrator unavailable",
                "duration_ms": None,
                "completed_at": None,
            } for tid in task_ids]

        pending = set(task_ids)
        results: Dict[str, Dict[str, Any]] = {}
        stream = self._orchestrator.stream_task_results(task_ids=list(task_ids))
        start = time.time()
        try:
            while pending:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    break
                timeout_step = max(0.1, min(self.config.poll_interval, remaining))
                try:
                    update = await asyncio.wait_for(stream.__anext__(), timeout=timeout_step)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    continue
                except Exception as exc:
                    self.logger.error("Task results stream error: %s", exc)
                    break
                tid = update.get("task_id")
                if not tid:
                    continue
                decoded = self._decode_result_update(update)
                self._result_cache[tid] = decoded
                status = str(decoded.get("status") or "").lower()
                if tid in pending and status in {"completed", "failed"}:
                    results[tid] = decoded
                    pending.remove(tid)
            ordered: List[Dict[str, Any]] = []
            for tid in task_ids:
                ordered.append(
                    results.get(tid)
                    or self._result_cache.get(tid)
                    or {
                        "task_id": tid,
                        "status": "timeout",
                        "result": {},
                        "error": f"Timed out waiting for task {tid}",
                        "duration_ms": None,
                        "completed_at": None,
                    }
                )
            return ordered
        finally:
            try:
                await stream.aclose()
            except Exception:
                pass

    async def _get_system_metrics_http(self) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{self.config.orchestrator_host}:{self.config.orchestrator_port}/metrics"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict):
                            self.metrics['raw'] = data
                            self.metrics.update({k: v for k, v in data.items() if k in self.metrics})
                        return dict(self.metrics)
        except Exception as exc:
            self.logger.error(f"Failed to get system metrics: {exc}")
        return dict(self.metrics)

    async def _adjust_concurrency_http(self, new_limit: int) -> None:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{self.config.orchestrator_host}:{self.config.orchestrator_port}/concurrency",
                    json={'limit': new_limit}
                ) as response:
                    if response.status == 200:
                        self.logger.info("Adjusted concurrency limit to %s", new_limit)
                        self.metrics["concurrency_limit"] = new_limit
                    else:
                        self.logger.error("Failed to adjust concurrency: %s", response.status)
        except Exception as exc:
            self.logger.error(f"Failed to adjust concurrency: {exc}")
    async def submit_rl_task(self, task: Dict[str, Any]) -> str:
        """Submit an RL task to the orchestrator"""
        if self.mode == "grpc":
            return await self._submit_rl_task_grpc(task)
        return await self._submit_rl_task_http(task)
    
    async def wait_for_task_completion(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for task completion"""
        if self.mode == "grpc":
            results = await self._wait_for_batch_completion_grpc([task_id], timeout)
            return results[0]
        return await self._wait_for_task_completion_http(task_id, timeout)
    
    async def submit_batch_tasks(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Submit a batch of RL tasks"""
        task_ids = []
        
        if self.mode == "grpc":
            await self._ensure_infra()
        
        for task in tasks:
            try:
                task_id = await self.submit_rl_task(task)
                task_ids.append(task_id)
            except Exception as e:
                self.logger.error(f"Failed to submit batch task: {e}")
                continue
        
        self.logger.info(f"Submitted {len(task_ids)} batch tasks")
        return task_ids
    
    async def wait_for_batch_completion(self, task_ids: List[str], timeout: int = 600) -> List[Dict[str, Any]]:
        """Wait for batch task completion"""
        if self.mode == "grpc":
            return await self._wait_for_batch_completion_grpc(task_ids, timeout)
        return await self._wait_for_batch_completion_http(task_ids, timeout)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics from orchestrator"""
        if self.mode == "grpc":
            await self._ensure_infra()
            if self._orchestrator:
                try:
                    metrics = await self._orchestrator.get_metrics()
                    self._update_metrics_from_grpc(metrics)
                except Exception as exc:
                    self.logger.error("Failed to get system metrics via gRPC: %s", exc)
            return dict(self.metrics)
        return await self._get_system_metrics_http()
    
    async def adjust_concurrency(self, new_limit: int):
        """Adjust orchestrator concurrency limit"""
        if self.mode == "grpc":
            self.logger.info("Concurrency adjustment via gRPC orchestrator not yet supported (requested %s)", new_limit)
            self.metrics["concurrency_limit"] = new_limit
            return
        await self._adjust_concurrency_http(new_limit)
    
    async def close(self):
        """Close the orchestrator client"""
        if self.mode == "grpc":
            if self._infra:
                try:
                    await self._infra.stop()
                finally:
                    self._infra = None
                    self._orchestrator = None
            return

        if self.orchestrator_process:
            try:
                self.orchestrator_process.terminate()
                self.orchestrator_process.wait()
            except Exception:
                pass
            self.orchestrator_process = None
        
        if self.orchestrator_thread:
            try:
                self.orchestrator_thread.join(timeout=5.0)
            except Exception:
                pass
            self.orchestrator_thread = None
        
        self.logger.info("Go orchestrator client closed")


class CoordinatedRLTrainer:
    """RL trainer coordinated by Go orchestrator"""
    
    def __init__(
        self, 
        config: UniversalPufferConfig,
        rust_config: RustRLConfig,
        orchestrator_config: GoOrchestratorConfig,
        bus: LocalBus
    ):
        self.config = config
        self.rust_config = rust_config
        self.orchestrator_config = orchestrator_config
        self.bus = bus
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.rl_tracker = get_rl_tracker()
        self.orchestrator_client = GoOrchestratorClient(orchestrator_config)
        self.session_id = f"coordinated_session_{int(time.time())}"
        
        # Training state
        self.training_active = False
        self.episode_count = 0
        self.total_timesteps = 0
        self.performance_history = []
        
        # Start session tracking
        self._start_session_tracking()
    
    def _start_session_tracking(self):
        """Start tracking this coordinated training session"""
        session_config = {
            'framework': self.config.framework,
            'num_workers': self.config.num_workers,
            'num_gpus': self.config.num_gpus,
            'num_envs': self.config.num_envs,
            'rust_optimized': True,
            'go_orchestrated': True,
            'coordinated': True
        }
        
        self.rl_tracker.start_training_session(self.session_id, session_config)
        self.logger.info(f"Started coordinated RL training session: {self.session_id}")
    
    async def train(self):
        """Main training loop with Go orchestrator coordination"""
        self.logger.info("Starting coordinated RL training with Go orchestrator")
        self.training_active = True
        
        try:
            # Training loop
            while self.training_active and self.total_timesteps < self.config.total_timesteps:
                # Create RL training tasks
                tasks = self._create_rl_tasks()
                
                # Submit tasks to orchestrator
                task_ids = await self.orchestrator_client.submit_batch_tasks(tasks)
                
                # Wait for task completion
                results = await self.orchestrator_client.wait_for_batch_completion(task_ids)
                
                # Process results
                self._process_training_results(results)
                
                # Log metrics
                await self._log_training_metrics()
                
                # Check for convergence
                if self._check_convergence():
                    self.logger.info("Training converged")
                    break
                
                # Adaptive concurrency adjustment
                await self._adjust_concurrency()
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.training_active = False
        finally:
            await self.orchestrator_client.close()
        
        self.logger.info("Coordinated RL training completed")
    
    def _create_rl_tasks(self) -> List[Dict[str, Any]]:
        """Create RL training tasks for orchestrator"""
        tasks = []
        
        # Create episode tasks
        for episode in range(self.config.num_envs):
            task = {
                'task_type': 'rl_episode',
                'episode_id': f"episode_{self.episode_count}_{episode}",
                'session_id': self.session_id,
                'config': {
                    'framework': self.config.framework,
                    'episode_length': self.config.episode_length,
                    'batch_size': self.config.batch_size,
                    'memory_limit': self.rust_config.env_memory_limit,
                    'cpu_limit': self.rust_config.env_cpu_limit
                },
                'priority': self.orchestrator_config.rl_task_priority,
                'timestamp': time.time()
            }
            tasks.append(task)
        
        return tasks
    
    def _process_training_results(self, results: List[Dict[str, Any]]):
        """Process training results from orchestrator"""
        for result in results:
            status = str(result.get('status') or '').lower()
            if status != 'completed':
                continue

            payload = result.get('result') or {}
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {}
            if not isinstance(payload, dict):
                payload = {}

            episode_data = payload.get('data', {})
            if isinstance(episode_data, str):
                try:
                    episode_data = json.loads(episode_data)
                except Exception:
                    episode_data = {}
            elif not isinstance(episode_data, dict):
                episode_data = {}

            reward = float(episode_data.get('total_reward', 0.0) or 0.0)
            steps = int(episode_data.get('steps', 0) or 0)

            self.performance_history.append(reward)
            self.total_timesteps += steps
            self.episode_count += 1

            if self.config.reddb_tracking:
                episode_metrics = {
                    'reward': reward,
                    'episode_length': steps,
                    'fps': episode_data.get('fps', 0.0),
                    'memory_usage': episode_data.get('memory_usage', 0.0),
                    'cpu_usage': episode_data.get('cpu_usage', 0.0),
                    'gpu_usage': episode_data.get('gpu_usage', 0.0),
                    'convergence_score': min(reward / 100.0, 1.0),
                    'success_rate': 1.0 if reward > 0 else 0.0,
                    'orchestrator_metrics': payload.get('orchestrator_metrics', {}),
                }
                self.rl_tracker.log_episode_metrics(self.session_id, self.episode_count, episode_metrics)
    
    async def _log_training_metrics(self):
        """Log training metrics to RedDB"""
        if self.config.reddb_tracking:
            # Get orchestrator metrics
            orchestrator_metrics = await self.orchestrator_client.get_system_metrics()
            
            # Update session with current metrics
            self.rl_tracker.update_training_session(self.session_id, {
                'num_episodes': self.episode_count,
                'total_timesteps': self.total_timesteps,
                'best_performance': max(self.performance_history) if self.performance_history else 0.0,
                'final_performance': self.performance_history[-1] if self.performance_history else 0.0,
                'orchestrator_metrics': orchestrator_metrics
            })
    
    async def _adjust_concurrency(self):
        """Adjust orchestrator concurrency based on performance"""
        try:
            metrics = await self.orchestrator_client.get_system_metrics()
            
            # Get current metrics
            queue_depth = metrics.get('queue_depth', 0)
            gpu_wait_time = metrics.get('gpu_wait_time', 0.0)
            error_rate = metrics.get('error_rate', 0.0)
            current_limit = metrics.get('concurrency_limit', self.orchestrator_config.base_limit)
            
            # Adaptive concurrency adjustment
            new_limit = current_limit
            
            # Increase concurrency if queue is high and GPU wait is low
            if (queue_depth > self.orchestrator_config.queue_high_watermark * 100 and 
                gpu_wait_time < self.orchestrator_config.gpu_wait_high and
                error_rate < self.orchestrator_config.error_rate_high):
                new_limit = min(current_limit + self.orchestrator_config.increase_step, 
                               self.orchestrator_config.max_limit)
            
            # Decrease concurrency if error rate is high or GPU wait is high
            elif (error_rate > self.orchestrator_config.error_rate_high or 
                  gpu_wait_time > self.orchestrator_config.gpu_wait_high):
                new_limit = max(current_limit - self.orchestrator_config.decrease_step, 
                               self.orchestrator_config.min_limit)
            
            # Apply new limit if changed
            if new_limit != current_limit:
                await self.orchestrator_client.adjust_concurrency(new_limit)
                self.logger.info(f"Adjusted concurrency from {current_limit} to {new_limit}")
                
        except Exception as e:
            self.logger.error(f"Failed to adjust concurrency: {e}")
    
    def _check_convergence(self) -> bool:
        """Check if training has converged"""
        if len(self.performance_history) < 10:
            return False
        
        # Check if recent performance is stable
        recent_performance = self.performance_history[-10:]
        performance_std = np.std(recent_performance)
        
        return performance_std < 0.01  # Very stable performance
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'training_active': self.training_active,
            'episode_count': self.episode_count,
            'total_timesteps': self.total_timesteps,
            'performance_history': self.performance_history[-10:] if self.performance_history else [],
            'framework': self.config.framework,
            'rust_optimized': True,
            'go_orchestrated': True,
            'coordinated': True,
            'session_id': self.session_id
        }


def create_coordinated_trainer(
    config: UniversalPufferConfig,
    rust_config: Optional[RustRLConfig] = None,
    orchestrator_config: Optional[GoOrchestratorConfig] = None,
    bus: Optional[LocalBus] = None
) -> CoordinatedRLTrainer:
    """Create a coordinated RL trainer using Go orchestrator"""
    
    if rust_config is None:
        rust_config = RustRLConfig()
    
    if orchestrator_config is None:
        orchestrator_config = GoOrchestratorConfig()
    
    if bus is None:
        bus = LocalBus()
    
    return CoordinatedRLTrainer(config, rust_config, orchestrator_config, bus)


async def run_coordinated_training(
    config: UniversalPufferConfig,
    rust_config: Optional[RustRLConfig] = None,
    orchestrator_config: Optional[GoOrchestratorConfig] = None,
    bus: Optional[LocalBus] = None
):
    """Run coordinated RL training with Go orchestrator"""
    
    trainer = create_coordinated_trainer(config, rust_config, orchestrator_config, bus)
    
    try:
        await trainer.train()
    except KeyboardInterrupt:
        trainer.training_active = False
        print("Training interrupted by user")
    except Exception as e:
        trainer.training_active = False
        print(f"Training failed: {e}")


if __name__ == "__main__":
    # Example usage
    config = UniversalPufferConfig(
        framework="auto",
        num_envs=32,
        num_workers=8,
        num_gpus=1,
        total_timesteps=1_000_000,
        reddb_tracking=True,
        react_dashboard=True
    )
    
    rust_config = RustRLConfig(
        num_envs=32,
        max_parallel_envs=64,
        prefetch_enabled=True,
        batch_processing=True
    )
    
    orchestrator_config = GoOrchestratorConfig(
        base_limit=8,
        max_limit=32,
        rl_task_priority=10,
        batch_size=64
    )
    
    asyncio.run(run_coordinated_training(config, rust_config, orchestrator_config))
