"""gRPC client for communicating with Go orchestrator."""

import importlib
import os
import sys
import grpc
import logging
from typing import Any, Dict, List, Optional, AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

_ORCHESTRATOR_PB = None
_ORCHESTRATOR_GRPC = None


def _ensure_proto_loaded() -> bool:
    global _ORCHESTRATOR_PB, _ORCHESTRATOR_GRPC
    if _ORCHESTRATOR_PB is not None and _ORCHESTRATOR_GRPC is not None:
        return True
    try:
        module_pb = importlib.import_module("dspy_agent.infra.pb.orchestrator.v1_pb2")
        orchestrator_pkg = sys.modules.setdefault("orchestrator", type(sys)("orchestrator"))
        orchestrator_pkg.v1_pb2 = module_pb
        sys.modules.setdefault("orchestrator.v1_pb2", module_pb)

        module_grpc = importlib.import_module("dspy_agent.infra.pb.orchestrator.v1_pb2_grpc")
        orchestrator_pkg.v1_pb2_grpc = module_grpc
        sys.modules.setdefault("orchestrator.v1_pb2_grpc", module_grpc)
        _ORCHESTRATOR_PB = module_pb
        _ORCHESTRATOR_GRPC = module_grpc
        return True
    except ImportError as exc:
        logger.warning("Protobuf modules not generated yet. Run: make proto-python (%s)", exc)
        _ORCHESTRATOR_PB = None
        _ORCHESTRATOR_GRPC = None
        return False


if _ensure_proto_loaded():
    orchestrator_pb = _ORCHESTRATOR_PB
    orchestrator_grpc = _ORCHESTRATOR_GRPC
else:
    orchestrator_pb = None
    orchestrator_grpc = None


DEFAULT_ORCHESTRATOR_ADDR = os.getenv("ORCHESTRATOR_GRPC_ADDR", "127.0.0.1:50052")


class OrchestratorClient:
    """Client for communicating with Go orchestrator via gRPC."""

    def __init__(self, address: str = DEFAULT_ORCHESTRATOR_ADDR):
        """Initialize the orchestrator client.

        Args:
            address: Orchestrator gRPC address (host:port)
        """
        self.address = address
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub = None

    async def connect(self) -> None:
        """Establish connection to orchestrator."""
        logger.info(f"Connecting to orchestrator at {self.address}")
        
        self.channel = grpc.aio.insecure_channel(self.address)
        
        if orchestrator_grpc:
            self.stub = orchestrator_grpc.OrchestratorServiceStub(self.channel)
        
        # Wait for channel to be ready
        try:
            await self.channel.channel_ready()
            logger.info("Connected to orchestrator successfully")
        except Exception as e:
            logger.error(f"Failed to connect to orchestrator: {e}")
            raise

    async def close(self) -> None:
        """Close the connection."""
        if self.channel:
            await self.channel.close()
            logger.info("Disconnected from orchestrator")

    async def health_check(self) -> Dict[str, any]:
        """Check orchestrator health.

        Returns:
            Health status dictionary
        """
        if not self.stub:
            return {"healthy": False, "error": "Not connected"}

        try:
            request = orchestrator_pb.HealthRequest()
            response = await self.stub.Health(request)
            
            return {
                "healthy": response.healthy,
                "version": response.version,
                "services": dict(response.services)
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    async def submit_task(
        self,
        task_id: str,
        task_class: str = "cpu_short",
        payload: Optional[Dict[str, str]] = None,
        priority: int = 0,
        workflow_id: Optional[str] = None
    ) -> Dict[str, any]:
        """Submit a task for execution.

        Args:
            task_id: Unique task identifier
            task_class: Task class (cpu_short, cpu_long, gpu, gpu_slurm)
            payload: Task payload data
            priority: Task priority
            workflow_id: Optional workflow ID

        Returns:
            Submission result
        """
        if not self.stub:
            raise RuntimeError("Not connected to orchestrator")

        try:
            str_payload = {str(k): str(v) for k, v in (payload or {}).items()}
            request = orchestrator_pb.SubmitTaskRequest(
                id=task_id,
                payload=str_payload,
                priority=priority,
                workflow_id=workflow_id or ""
            )
            try:
                setattr(request, "class", task_class)
            except AttributeError:
                request.class_ = task_class
            
            response = await self.stub.SubmitTask(request)
            
            return {
                "success": response.success,
                "task_id": response.task_id,
                "error": response.error
            }
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            return {"success": False, "error": str(e)}

    async def stream_task_results(
        self,
        task_ids: Optional[List[str]] = None
    ) -> AsyncIterator[Dict[str, any]]:
        """Stream task results as they complete.

        Args:
            task_ids: Optional list of specific task IDs to monitor

        Yields:
            Task result dictionaries
        """
        if not self.stub:
            raise RuntimeError("Not connected to orchestrator")

        try:
            request = orchestrator_pb.StreamTaskResultsRequest(
                task_ids=task_ids or [],
                include_completed=True
            )
            
            async for result in self.stub.StreamTaskResults(request):
                yield {
                    "task_id": result.task_id,
                    "status": result.status,
                    "result": dict(result.result),
                    "error": result.error,
                    "duration_ms": result.duration_ms,
                    "completed_at": result.completed_at
                }
        except Exception as e:
            logger.error(f"Task results stream error: {e}")
            raise

    async def get_metrics(
        self,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get current system metrics.

        Args:
            metric_names: Optional list of specific metrics to retrieve

        Returns:
            Metrics dictionary
        """
        if not self.stub:
            raise RuntimeError("Not connected to orchestrator")

        try:
            request = orchestrator_pb.GetMetricsRequest(
                metric_names=metric_names or []
            )
            
            response = await self.stub.GetMetrics(request)
            
            return dict(response.metrics)
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        if not self.stub:
            raise RuntimeError("Not connected to orchestrator")
        if not orchestrator_pb:
            raise RuntimeError("Protobuf stubs not generated")
        try:
            request = orchestrator_pb.GetTaskStatusRequest(task_id=task_id)
            response = await self.stub.GetTaskStatus(request)
            result = dict(response.result or {})
            return {
                "task_id": response.task_id,
                "status": response.status,
                "result": result,
                "result_payload": result,
                "error": response.error,
            }
        except Exception as exc:
            logger.error(f"Failed to get task status: {exc}")
            raise

    async def stream_events(
        self,
        event_types: Optional[List[str]] = None,
        since_timestamp: int = 0
    ) -> AsyncIterator[Dict[str, any]]:
        """Stream system events.

        Args:
            event_types: Optional list of event types to filter
            since_timestamp: Only receive events after this timestamp

        Yields:
            Event dictionaries
        """
        if not self.stub:
            raise RuntimeError("Not connected to orchestrator")

        try:
            request = orchestrator_pb.StreamEventsRequest(
                event_types=event_types or [],
                since_timestamp=since_timestamp
            )
            
            async for event in self.stub.StreamEvents(request):
                yield {
                    "event_type": event.event_type,
                    "resource_id": event.resource_id,
                    "data": dict(event.data),
                    "timestamp": event.timestamp
                }
        except Exception as e:
            logger.error(f"Events stream error: {e}")
            raise


@asynccontextmanager
async def orchestrator_client(address: str = DEFAULT_ORCHESTRATOR_ADDR):
    """Context manager for orchestrator client.

    Example:
        async with orchestrator_client() as client:
            await client.submit_task("task-123", "cpu_short")
    """
    client = OrchestratorClient(address)
    await client.connect()
    try:
        yield client
    finally:
        await client.close()

