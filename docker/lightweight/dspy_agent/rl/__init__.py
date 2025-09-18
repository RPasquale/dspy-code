"""RL toolkit consolidated under a single module for compactness.

Public symbols are re-exported from :mod:`dspy_agent.rlkit`.
"""

from .rlkit import *  # re-export
from .rlkit import __all__ as _rl_all

__all__ = list(_rl_all)

try:
    from . import puffer_sweep  # noqa: F401
except ImportError:  # pragma: no cover - optional
    puffer_sweep = None  # type: ignore
else:
    __all__.append('puffer_sweep')

try:
    from . import hparam_guide  # noqa: F401
except ImportError:  # pragma: no cover - optional
    hparam_guide = None  # type: ignore
else:
    __all__.append('hparam_guide')
