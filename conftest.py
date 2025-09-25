from __future__ import annotations

import asyncio
import inspect


def pytest_pyfunc_call(pyfuncitem):  # type: ignore[func-returns-value]
    # Run async def tests even when pytest-asyncio is not installed.
    obj = pyfuncitem.obj
    if asyncio.iscoroutinefunction(obj):
        loop = asyncio.new_event_loop()
        try:
            # Filter kwargs to match function signature (handles bound methods)
            sig = inspect.signature(obj)
            call_kwargs = {k: v for k, v in pyfuncitem.funcargs.items() if k in sig.parameters}
            loop.run_until_complete(obj(**call_kwargs))
        finally:
            loop.close()
        return True
    return None
