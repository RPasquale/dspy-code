from __future__ import annotations

"""
Torch-based Group Relative Preference Optimization (GRPO) trainer and service.

This package provides a production-oriented GRPO implementation with:
- Dataset loader for grouped preferences (JSONL)
- Flexible policy interfaces (HF Transformers or lightweight fallback)
- Reference model KL regularization
- Stable group-wise advantage normalization and clipping
- Checkpointing and metrics streaming
- Background service integration for the dashboard HTTP server

All components are designed to degrade gracefully if optional deps are missing
(e.g., Transformers). The trainer will report clear errors via status.
"""

from .dataset import GroupPreferenceDataset, GroupSample
from .model import PolicyModel, HFPolicy, BowPolicy, PolicyConfig
from .trainer import GRPOConfig, GRPOTrainer, GRPOMetrics
from .service import GRPOService, GlobalGrpoService

__all__ = [
    "GroupPreferenceDataset",
    "GroupSample",
    "PolicyModel",
    "HFPolicy",
    "BowPolicy",
    "PolicyConfig",
    "GRPOConfig",
    "GRPOTrainer",
    "GRPOMetrics",
    "GRPOService",
    "GlobalGrpoService",
]

