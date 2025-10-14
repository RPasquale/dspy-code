#!/usr/bin/env python3
"""
Debug gRPC connection to understand the UNIMPLEMENTED error.
"""

import asyncio
import grpc
from dspy_agent.infra.grpc_client import OrchestratorClient

async def debug_connection():
    print("=== gRPC Connection Debug ===")
    
    # Test 1: Basic connection
    print("1. Testing basic connection...")
    try:
        client = OrchestratorClient('localhost:50053')
        await client.connect()
        print("   ✓ Connected successfully")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return
    
    # Test 2: Check what methods are available
    print("2. Testing Health method...")
    try:
        health = await client.health_check()
        print(f"   ✓ Health check result: {health}")
    except grpc.aio.AioRpcError as e:
        print(f"   ✗ Health check failed: {e.code()} - {e.details()}")
        print(f"   Debug string: {e.debug_error_string()}")
    except Exception as e:
        print(f"   ✗ Health check error: {e}")
    
    # Test 3: Test GetMetrics method
    print("3. Testing GetMetrics method...")
    try:
        metrics = await client.get_metrics()
        print(f"   ✓ Metrics result: {metrics}")
    except grpc.aio.AioRpcError as e:
        print(f"   ✗ GetMetrics failed: {e.code()} - {e.details()}")
        print(f"   Debug string: {e.debug_error_string()}")
    except Exception as e:
        print(f"   ✗ GetMetrics error: {e}")
    
    # Test 4: Test SubmitTask method
    print("4. Testing SubmitTask method...")
    try:
        result = await client.submit_task(
            task_id="test-task-001",
            task_class="cpu_short",
            payload={"test": "data"}
        )
        print(f"   ✓ SubmitTask result: {result}")
    except grpc.aio.AioRpcError as e:
        print(f"   ✗ SubmitTask failed: {e.code()} - {e.details()}")
        print(f"   Debug string: {e.debug_error_string()}")
    except Exception as e:
        print(f"   ✗ SubmitTask error: {e}")
    
    await client.close()
    print("   ✓ Connection closed")

if __name__ == "__main__":
    asyncio.run(debug_connection())
