from __future__ import annotations

from typing import Optional

from .base import Storage
from ..dbkit import get_storage as get_storage  # re-export consolidated factory

__all__ = ["get_storage", "Storage"]
