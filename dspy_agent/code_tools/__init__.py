"""
Code analysis and manipulation tools for DSPy agent.

This package keeps optional, heavy integrations (Kafka watchers, background
indexers) as optional imports so that core utilities can be imported without
pulling extra dependencies.
"""

from .code_search import (
    LineHit,
    search_text,
    search_file,
    extract_context,
    python_extract_symbol,
    ast_grep_available,
    run_ast_grep,
)
from .code_eval import *  # lightweight
from .code_snapshot import *  # lightweight
from .diffutil import *
from .patcher import *

# Optional modules: only import if dependencies are present
try:  # pragma: no cover - optional
    from .code_watch import *  # requires kafka optional
except Exception:  # keep base tools available even if optional deps missing
    pass
try:  # pragma: no cover - optional
    from .code_indexer_worker import *  # requires confluent-kafka, dspy or HF
except Exception:
    pass

__all__ = [
    'LineHit',
    'search_text',
    'search_file',
    'extract_context',
    'python_extract_symbol',
    'ast_grep_available',
    'run_ast_grep',
]
