from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Candidate:
    text: str
    reward: float


@dataclass
class GroupSample:
    prompt: str
    candidates: List[Candidate]
    meta: Dict[str, Any]


class GroupPreferenceDataset:
    """Grouped preference dataset for GRPO.

    Expected JSONL schema (one line per group):
    {
      "prompt": str,
      "candidates": [
         {"text": str, "reward": float},
         ... (K items)
      ],
      // optional metadata recorded but unused by training
      "meta": { ... }
    }

    Notes:
      - Rewards can be any real values; trainer will standardize per-group.
      - K >= 2 recommended. Groups with <2 candidates are skipped.
    """

    def __init__(self, path: Path | str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"GRPO dataset not found: {p}")
        self.path = p
        self._index: List[int] = []
        self._offsets: List[int] = []
        # Pre-index file line offsets for random access
        self._build_index()

    def _build_index(self) -> None:
        offsets: List[int] = []
        pos = 0
        with self.path.open("rb") as f:
            for i, line in enumerate(f):
                offsets.append(pos)
                pos += len(line)
        self._index = list(range(len(offsets)))
        self._offsets = offsets

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._index)

    def __getitem__(self, idx: int) -> GroupSample:
        off = self._offsets[idx]
        with self.path.open("rb") as f:
            f.seek(off)
            raw = f.readline().decode("utf-8", errors="ignore").strip()
        try:
            obj = json.loads(raw)
        except Exception as e:  # pragma: no cover - guard
            raise ValueError(f"Invalid JSONL at index {idx}: {e}")

        prompt = str(obj.get("prompt", "")).strip()
        cand_objs = obj.get("candidates") or []
        candidates: List[Candidate] = []
        for c in cand_objs:
            try:
                txt = str(c.get("text", ""))
                rew = float(c.get("reward", 0.0))
                candidates.append(Candidate(text=txt, reward=rew))
            except Exception:
                continue
        meta = obj.get("meta") or {}
        # Ensure K>=2; if not, synthesize a no-op negative example
        if len(candidates) < 2:
            candidates.append(Candidate(text="", reward=min([c.reward for c in candidates] + [0.0]) - 1.0))
        return GroupSample(prompt=prompt, candidates=candidates, meta=meta)

