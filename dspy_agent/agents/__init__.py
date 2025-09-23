"""
Agent components and orchestration for DSPy agent.

Keep optional adapters lazy to avoid hard dependency on DSPy at import time.
"""

from .orchestrator_runtime import *
from .knowledge import *

# Optional worker (Kafka) and adapter (DSPy). Import defensively.
try:  # pragma: no cover - optional
    from .router_worker import *
except Exception:
    pass
try:  # pragma: no cover - optional
    from .adapter import *
except Exception:
    pass

__all__ = [
    'EvalOutcome',
    'evaluate_tool_choice',
    'build_code_graph',
    'summarize_code_graph',
]
