"""Hyperparameter guidance distilled from recent reasoning-RL literature.

This module centralises recommended defaults so both the CLI and automated
sweep pipelines can surface consistent advice. Values are intentionally
lightweight (no heavy deps) so they can be imported in low-resource contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class HyperParamBand:
    """Simple container describing a recommended range and explanation."""

    name: str
    low: float
    high: float
    target: Optional[float]
    unit: str
    rationale: str


@dataclass(frozen=True)
class HyperParamGroup:
    """Logically grouped hyperparameters (e.g., sampling, advantages)."""

    title: str
    items: List[HyperParamBand]


def _band(name: str, low: float, high: float, target: Optional[float], unit: str, rationale: str) -> HyperParamBand:
    return HyperParamBand(name=name, low=low, high=high, target=target, unit=unit, rationale=rationale)


_GUIDE: List[HyperParamGroup] = [
    HyperParamGroup(
        title="Sampling & Entropy",
        items=[
            _band(
                "temperature",
                0.6,
                1.2,
                0.85,
                "",
                "Target entropy ≈0.3 with adaptive scheduling (Skywork-OR1, AceReason)",
            ),
            _band(
                "target_entropy",
                0.25,
                0.35,
                0.30,
                "nats",
                "Keeps exploration healthy without collapsing diversity",
            ),
        ],
    ),
    HyperParamGroup(
        title="Loss & Advantage Norm",
        items=[
            _band(
                "adv_norm/group_size",
                8,
                32,
                16,
                "samples",
                "Group-mean, batch-std normalisation (Lite-PPO / Tricks-or-Traps)",
            ),
            _band(
                "clip_higher",
                1.0,
                1.3,
                1.2,
                "ratio",
                "Encourage exploration on aligned policies when scores plateau",
            ),
        ],
    ),
    HyperParamGroup(
        title="Curriculum & Context",
        items=[
            _band(
                "max_tokens_stage1",
                512,
                1024,
                768,
                "tokens",
                "Warm-up on short contexts before expanding to 2k+/4k windows",
            ),
            _band(
                "max_tokens_stage2",
                2048,
                4096,
                3072,
                "tokens",
                "Intermediate stage for stabilising long-CoT reasoning",
            ),
        ],
    ),
    HyperParamGroup(
        title="Reward Signals",
        items=[
            _band(
                "verifier_weight/tests",
                0.8,
                1.2,
                1.0,
                "weight",
                "Primary correctness source (unit tests / math sandbox)",
            ),
            _band(
                "verifier_weight/blast_radius_penalty",
                -0.8,
                -0.2,
                -0.5,
                "weight",
                "Discourage edits that touch excessive lines",
            ),
        ],
    ),
]


def get_hparam_groups() -> List[HyperParamGroup]:
    """Return the curated hyperparameter guidance list."""

    return list(_GUIDE)


def as_dict() -> List[Mapping[str, object]]:
    """Return guide as JSON-compatible structures for CLI output."""

    result: List[Mapping[str, object]] = []
    for group in _GUIDE:
        result.append(
            {
                "title": group.title,
                "items": [
                    {
                        "name": item.name,
                        "low": item.low,
                        "high": item.high,
                        "target": item.target,
                        "unit": item.unit,
                        "rationale": item.rationale,
                    }
                    for item in group.items
                ],
            }
        )
    return result
