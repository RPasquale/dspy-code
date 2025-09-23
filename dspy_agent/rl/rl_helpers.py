from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from .rlkit import RLConfig as _RLConfig


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_effective_rl_config_dict(workspace: Path) -> Dict[str, Any]:
    """Return the effective RL config dict.

    Prefers <ws>/.dspy/rl/best.json{"config": {...}}; falls back to <ws>/.dspy_rl.json
    """
    best_path = workspace / '.dspy' / 'rl' / 'best.json'
    try:
        raw = json.loads(best_path.read_text())
        if isinstance(raw, dict) and isinstance(raw.get('config'), dict):
            return dict(raw['config'])
    except Exception:
        pass
    return _safe_read_json(workspace / '.dspy_rl.json')


def rl_config_from_dict(data: Mapping[str, Any]) -> _RLConfig:
    """Convert a plain dict into an rlkit.RLConfig instance with normalized types."""
    cfg = _RLConfig()
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if isinstance(cfg.weights, dict):
        cfg.weights = {str(k): float(v) for k, v in cfg.weights.items()}  # type: ignore[arg-type]
    if isinstance(cfg.penalty_kinds, (list, tuple)):
        cfg.penalty_kinds = [str(x) for x in cfg.penalty_kinds]
    if isinstance(cfg.clamp01_kinds, (list, tuple)):
        cfg.clamp01_kinds = [str(x) for x in cfg.clamp01_kinds]
    if isinstance(cfg.scales, dict):
        cfg.scales = {
            str(k): (
                (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) and len(v) == 2 else tuple(v) if isinstance(v, tuple) else v
            )
            for k, v in cfg.scales.items()
        }
    if isinstance(cfg.actions, (list, tuple)):
        cfg.actions = [str(a) for a in cfg.actions]
    return cfg


__all__ = [
    'load_effective_rl_config_dict',
    'rl_config_from_dict',
]

