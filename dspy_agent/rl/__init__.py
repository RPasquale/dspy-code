"""RL toolkit facade.

To avoid import-time crashes from optional native deps (gym/numpy), we intentionally
do NOT import :mod:`dspy_agent.rl.rlkit` at package import. Import submodules
directly, e.g. ``from dspy_agent.rl import puffer_sweep`` or
``import dspy_agent.rl.rlkit as rl``.
"""

__all__ = []

# Optional submodules (safe to import without native deps)
try:  # optional PufferLib sweep wrappers (numpy-light path used in tests)
    from . import puffer_sweep  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    puffer_sweep = None  # type: ignore
else:
    __all__.append("puffer_sweep")

try:
    from . import hparam_guide  # noqa: F401
except Exception:  # pragma: no cover - defensive
    hparam_guide = None  # type: ignore
else:
    __all__.append("hparam_guide")
