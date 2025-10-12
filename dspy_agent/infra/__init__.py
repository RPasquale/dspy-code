"""Infrastructure module for agent communication with Go/Rust services via gRPC.

This module provides a simple interface for starting and managing all DSPy agent
infrastructure services (RedDB, InferMesh, Orchestrator, etc.) through Go/Rust backends.

Example:
    ```python
    from dspy_agent.infra import AgentInfra
    
    async def main():
        async with AgentInfra.start() as infra:
            # All services ready
            result = await infra.submit_task("task-1", {"data": "value"})
            print(result)
    
    import asyncio
    asyncio.run(main())
    ```
"""

from .agent_infra import AgentInfra
from .grpc_client import OrchestratorClient, orchestrator_client

__all__ = ["AgentInfra", "OrchestratorClient", "orchestrator_client"]

