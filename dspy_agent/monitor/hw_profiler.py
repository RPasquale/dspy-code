from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


@dataclass
class HWSnapshot:
    ts: float
    cpu_percent: Optional[float] = None
    mem_total_mb: Optional[float] = None
    mem_used_mb: Optional[float] = None
    mem_percent: Optional[float] = None
    swap_total_mb: Optional[float] = None
    swap_used_mb: Optional[float] = None
    disk_used_mb: Optional[float] = None
    disk_total_mb: Optional[float] = None
    gpu_util_percent: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None
    gpu_power_watts: Optional[float] = None
    gpu_name: Optional[str] = None


def _nvidia_smi_query() -> Dict[str, Any]:
    if not shutil.which('nvidia-smi'):
        return {}
    try:
        q = [
            'utilization.gpu', 'memory.used', 'memory.total', 'power.draw', 'name'
        ]
        cmd = ['nvidia-smi', f"--query-gpu={','.join(q)}", '--format=csv,noheader,nounits']
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=2).decode('utf-8', errors='ignore').strip()
        # Only take first GPU for simplicity
        line = out.splitlines()[0] if out else ''
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            return {
                'gpu_util_percent': float(parts[0] or 0.0),
                'gpu_mem_used_mb': float(parts[1] or 0.0),
                'gpu_mem_total_mb': float(parts[2] or 0.0),
                'gpu_power_watts': float(parts[3] or 0.0),
                'gpu_name': parts[4],
            }
    except Exception:
        return {}
    return {}


def read_hw() -> HWSnapshot:
    now = time.time()
    snap = HWSnapshot(ts=now)
    # CPU/Mem
    if psutil is not None:
        try:
            snap.cpu_percent = float(psutil.cpu_percent(interval=None))
        except Exception:
            pass
        try:
            vm = psutil.virtual_memory()
            snap.mem_total_mb = float(vm.total) / (1024 * 1024)
            snap.mem_used_mb = float(vm.used) / (1024 * 1024)
            snap.mem_percent = float(vm.percent)
        except Exception:
            pass
        try:
            sw = psutil.swap_memory()
            snap.swap_total_mb = float(sw.total) / (1024 * 1024)
            snap.swap_used_mb = float(sw.used) / (1024 * 1024)
        except Exception:
            pass
        try:
            du = psutil.disk_usage(Path.cwd().anchor)
            snap.disk_total_mb = float(du.total) / (1024 * 1024)
            snap.disk_used_mb = float(du.used) / (1024 * 1024)
        except Exception:
            pass
    # GPU via nvidia-smi (best-effort)
    try:
        data = _nvidia_smi_query()
        for k, v in data.items():
            setattr(snap, k, v)
    except Exception:
        pass
    # Torch fallback for CUDA
    if snap.gpu_mem_total_mb is None:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                idx = 0
                prop = torch.cuda.get_device_properties(idx)
                snap.gpu_name = getattr(prop, 'name', '')
                snap.gpu_mem_total_mb = float(getattr(prop, 'total_memory', 0) / (1024 * 1024))
                snap.gpu_mem_used_mb = float(torch.cuda.memory_allocated(idx) / (1024 * 1024))
        except Exception:
            pass
    return snap


class HardwareProfiler(threading.Thread):
    def __init__(self, workspace: Path, interval_sec: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.workspace = Path(workspace)
        self.interval = float(max(1.0, interval_sec))
        self._stop = threading.Event()
        self._out = self.workspace / '.dspy_hw.json'
        try:
            from ..db.factory import get_storage as _get
            self._storage = _get()
        except Exception:
            self._storage = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # pragma: no cover
        while not self._stop.is_set():
            snap = read_hw()
            data = asdict(snap)
            try:
                self._out.write_text(json.dumps(data, indent=2))
            except Exception:
                pass
            if self._storage is not None:
                try:
                    self._storage.append('hw_metrics', data)  # type: ignore[attr-defined]
                except Exception:
                    pass
            time.sleep(self.interval)

