from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Mapping, Optional


@dataclass
class FeatureSnapshot:
    timestamp: float
    count: int
    means: List[float]
    variances: List[float]
    min_values: List[float]
    max_values: List[float]
    feature_names: List[str]


class FeatureStore:
    """Sliding-window feature statistics for streaming RL signals."""

    def __init__(self, window: int = 512, feature_names: Optional[Iterable[str]] = None) -> None:
        self._window = max(1, int(window))
        self._lock = threading.Lock()
        self._records: Deque[List[float]] = deque(maxlen=self._window)
        self._names: List[str] = list(feature_names) if feature_names else []
        self._last_snapshot: Optional[FeatureSnapshot] = None

    def _ensure_names(self, payload: Mapping[str, object]) -> None:
        if self._names:
            return
        names = payload.get('feature_names') or payload.get('names')
        if isinstance(names, Iterable):
            try:
                self._names = [str(n) for n in names]
            except Exception:
                self._names = []

    def update(self, payload: Mapping[str, object]) -> None:
        features = payload.get('features') or payload.get('vector')
        if not isinstance(features, Iterable):
            return
        try:
            values = [float(v) for v in features]
        except Exception:
            return
        if not values:
            return
        self._ensure_names(payload)
        with self._lock:
            self._records.append(values)
            self._last_snapshot = None

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._last_snapshot = None

    def snapshot(self) -> Optional[FeatureSnapshot]:
        with self._lock:
            if self._last_snapshot is not None:
                return self._last_snapshot
            if not self._records:
                return None
            dim = len(self._records[0])
            sums = [0.0] * dim
            sums_sq = [0.0] * dim
            min_vals = [math.inf] * dim
            max_vals = [-math.inf] * dim
            count = len(self._records)
            for vec in self._records:
                if len(vec) != dim:
                    continue
                for idx, value in enumerate(vec):
                    sums[idx] += value
                    sums_sq[idx] += value * value
                    if value < min_vals[idx]:
                        min_vals[idx] = value
                    if value > max_vals[idx]:
                        max_vals[idx] = value
            means = [s / count for s in sums]
            variances = []
            for idx in range(dim):
                mean = means[idx]
                var = max(sums_sq[idx] / count - mean * mean, 0.0)
                variances.append(var)
            names = self._names if len(self._names) == dim else [f'f_{i}' for i in range(dim)]
            snapshot = FeatureSnapshot(
                timestamp=time.time(),
                count=count,
                means=means,
                variances=variances,
                min_values=[-math.inf if math.isinf(v) and v < 0 else v for v in min_vals],
                max_values=[math.inf if math.isinf(v) and v > 0 else v for v in max_vals],
                feature_names=names,
            )
            self._last_snapshot = snapshot
            return snapshot


__all__ = ['FeatureStore', 'FeatureSnapshot']

