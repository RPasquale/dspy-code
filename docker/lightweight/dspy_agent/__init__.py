"""
DSPy Agent - A trainable coding agent with streaming and RL capabilities.

This package exposes multiple subpackages (embedding, code_tools, agents, streaming,
training, rl, skills, db). To avoid circular-import issues at import time, we do not
eagerly import subpackages here. Import the submodules you need directly, e.g.:

    from dspy_agent.embedding import build_emb_index
    from dspy_agent.rl import RLToolEnv

"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
