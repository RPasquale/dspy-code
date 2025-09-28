"""
Embedding and indexing functionality for DSPy agent.
"""

from importlib import import_module
from typing import Any

from .embeddings_index import (
    EmbIndexItem,
    build_emb_index,
    save_emb_index,
    load_emb_index,
    embed_query,
    emb_search,
)
from .indexer import (
    Chunk,
    iter_chunks,
    build_index,
    save_index,
    load_index,
    semantic_search,
    tokenize,
)

_LAZY_MODULES = {
    'dspy_embedder_service',
    'embed_worker',
    'kafka_indexer',
    'infermesh_server',
    'infermesh_mock',
    'smoke_embed_pipeline',
}

__all__ = [
    'EmbIndexItem',
    'build_emb_index',
    'save_emb_index',
    'load_emb_index',
    'embed_query',
    'emb_search',
    'Chunk',
    'iter_chunks',
    'build_index',
    'save_index',
    'load_index',
    'semantic_search',
    'tokenize',
    'dspy_embedder_service',
    'embed_worker',
    'kafka_indexer',
    'infermesh_server',
    'infermesh_mock',
    'smoke_embed_pipeline',
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
