from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from ..memory.manager import MemoryManager


AUTOSCALE_CFG_PATH = Path('.dspy_autoscale.json')
AUTOSCALE_PROPOSALS = Path('.dspy_scale_proposals.jsonl')
AUTOSCALE_REQUESTS = Path('.dspy_scale_requests.jsonl')


@dataclass
class AutoscaleConfig:
    storage_budget_gb: int = 20
    storage_ttl_days: int = 30
    stream_target_rps: float = 1.0  # target vectorization rate per sec
    gpu_hours_per_day: int = 0      # 0=disabled locally
    max_gpu_hours_per_day: int = 24
    max_storage_gb: int = 1024

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def load_config(path: Path = AUTOSCALE_CFG_PATH) -> AutoscaleConfig:
    try:
        if path.exists():
            data = json.loads(path.read_text())
            return AutoscaleConfig(**data)
    except Exception:
        pass
    return AutoscaleConfig()


def save_config(cfg: AutoscaleConfig, path: Path = AUTOSCALE_CFG_PATH) -> Path:
    path.write_text(json.dumps(cfg.to_dict(), indent=2))
    return path


def _read_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _now() -> float:
    return time.time()


def capacity_snapshot(root: Path) -> Dict[str, object]:
    mm = MemoryManager(root)
    st = mm.status()
    srl = _read_json(root / '.dspy_stream_rl.json')
    auto = _read_json(root / '.dspy_auto_status.json')
    return {
        'memory': st,
        'stream': srl,
        'auto': auto,
        'ts': _now(),
    }


def propose_scale(root: Path, cfg: Optional[AutoscaleConfig] = None) -> List[Dict[str, object]]:
    cfg = cfg or load_config()
    snap = capacity_snapshot(root)
    proposals: List[Dict[str, object]] = []

    # Storage proposal
    mem = snap.get('memory') or {}
    total_b = int(mem.get('total_bytes', 0) or 0)
    budget_b = int(mem.get('budget_bytes', cfg.storage_budget_gb * 1024 * 1024) or 0)
    util = (float(total_b) / float(budget_b)) if budget_b > 0 else 0.0
    if util >= 0.85 and (cfg.storage_budget_gb * 2) <= cfg.max_storage_gb:
        proposals.append({
            'kind': 'storage_budget_increase',
            'from_gb': cfg.storage_budget_gb,
            'to_gb': min(cfg.storage_budget_gb * 2, cfg.max_storage_gb),
            'reason': f'utilization={util:.2f} beyond 85% threshold',
        })

    # Stream throughput proposal (GPU hours)
    stream = snap.get('stream') or {}
    rps = float(stream.get('rate_per_sec', 0.0) or 0.0)
    target = float(cfg.stream_target_rps)
    if rps > (target * 1.5) and cfg.gpu_hours_per_day < cfg.max_gpu_hours_per_day:
        proposals.append({
            'kind': 'gpu_hours_increase',
            'from_hpd': cfg.gpu_hours_per_day,
            'to_hpd': min(cfg.gpu_hours_per_day + 2, cfg.max_gpu_hours_per_day),
            'reason': f'rps={rps:.2f} above target={target:.2f}',
        })

    return proposals


def persist_proposals(items: List[Dict[str, object]], path: Path = AUTOSCALE_PROPOSALS) -> None:
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        for it in items:
            rec = dict(it)
            rec['ts'] = _now()
            f.write(json.dumps(rec) + "\n")


def record_request(kind: str, params: Dict[str, object], *, approved: bool, path: Path = AUTOSCALE_REQUESTS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {'kind': kind, 'params': params, 'approved': bool(approved), 'ts': _now()}
    try:
        with path.open('a') as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def read_proposals(path: Path = AUTOSCALE_PROPOSALS) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    try:
        if not path.exists():
            return items
        with path.open('r') as f:
            for line in f:
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return items


def read_requests(path: Path = AUTOSCALE_REQUESTS) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    try:
        if not path.exists():
            return items
        with path.open('r') as f:
            for line in f:
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return items

