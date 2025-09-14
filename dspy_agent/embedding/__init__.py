"""
Embedding and indexing functionality for DSPy agent.
"""

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
]
