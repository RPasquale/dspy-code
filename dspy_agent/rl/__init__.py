"""RL toolkit consolidated under a single module for compactness.

Public symbols are re-exported from :mod:`dspy_agent.rlkit`. Additional
utilities (e.g. hyperparameter sweeps) live alongside the toolkit and are
exposed as modules.
"""

from .rlkit import *  # type: ignore[F401]  # re-export primary API
from .rlkit import __all__ as _rl_all

__all__ = list(_rl_all)

try:  # optional PufferLib integration
    from . import puffer_sweep  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    puffer_sweep = None  # type: ignore
else:
    __all__.append("puffer_sweep")

try:
    from . import hparam_guide  # noqa: F401
except ImportError:  # pragma: no cover - defensive
    hparam_guide = None  # type: ignore
else:
    __all__.append("hparam_guide")
