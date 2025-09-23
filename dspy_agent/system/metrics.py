from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_size_to_mb(txt: str) -> float:
    try:
        s = (txt or '').strip().upper()
        num = float(''.join(ch for ch in s if (ch.isdigit() or ch == '.')) or '0')
        if 'G' in s:
            return num * 1024.0
        if 'M' in s:
            return num
        if 'K' in s:
            return num / 1024.0
        if 'B' in s:
            return num / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def _docker_stats() -> List[Dict[str, Any]]:
    try:
        proc = subprocess.run(
            ['docker', 'stats', '--no-stream', '--format', '{{json .}}'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        out: List[Dict[str, Any]] = []
        for ln in lines:
            try:
                obj = json.loads(ln)
                name = obj.get('Name') or obj.get('Container')
                cpu = float((obj.get('CPUPerc') or '0').strip('% '))
                mem = (obj.get('MemUsage') or '').split('/')
                used_mb = _parse_size_to_mb(mem[0]) if len(mem) > 0 else 0.0
                lim_mb = _parse_size_to_mb(mem[1]) if len(mem) > 1 else 0.0
                mem_pct = float((obj.get('MemPerc') or '0').strip('% '))
                out.append(
                    {
                        'name': name,
                        'cpu_pct': cpu,
                        'mem_used_mb': used_mb,
                        'mem_limit_mb': lim_mb,
                        'mem_pct': mem_pct,
                        'net_io': obj.get('NetIO'),
                        'block_io': obj.get('BlockIO'),
                        'pids': obj.get('PIDs'),
                    }
                )
            except Exception:
                continue
        return out
    except Exception:
        return []


def _disk_usage(path: Path) -> Dict[str, Any]:
    try:
        total, used, free = shutil.disk_usage(str(path))
        to_gb = lambda b: round(b / (1024 ** 3), 2)
        pct = round((used / total) * 100.0, 1) if total else 0.0
        return {
            'path': str(path),
            'total_gb': to_gb(total),
            'used_gb': to_gb(used),
            'free_gb': to_gb(free),
            'pct_used': pct,
        }
    except Exception:
        return {'path': str(path), 'total_gb': 0, 'used_gb': 0, 'free_gb': 0, 'pct_used': 0}


def _gpu_info() -> List[Dict[str, Any]]:
    # Try nvidia-smi first, then torch.cuda
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=3,
        )
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        gpus = []
        for ln in lines:
            parts = [p.strip() for p in ln.split(',')]
            if len(parts) >= 4:
                gpus.append(
                    {
                        'name': parts[0],
                        'mem_used_mb': float(parts[1]),
                        'mem_total_mb': float(parts[2]),
                        'util_pct': float(parts[3]),
                    }
                )
        if gpus:
            return gpus
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mb = int(props.total_memory / (1024 ** 2))
                # best-effort used_mb via mem_allocated
                used_mb = int(torch.cuda.memory_allocated(i) / (1024 ** 2))
                gpus.append({'name': props.name, 'mem_used_mb': used_mb, 'mem_total_mb': total_mb, 'util_pct': None})
            return gpus
    except Exception:
        pass
    return []


def _host_cpu_mem() -> Dict[str, Any]:
    cpu_pct = None
    load_avg = None
    mem_total_gb = None
    mem_used_gb = None
    mem_free_gb = None
    mem_pct = None
    try:
        import psutil  # type: ignore

        cpu_pct = float(psutil.cpu_percent(interval=0.0))
        vm = psutil.virtual_memory()
        mem_total_gb = round(vm.total / (1024 ** 3), 2)
        mem_used_gb = round(vm.used / (1024 ** 3), 2)
        mem_free_gb = round(vm.available / (1024 ** 3), 2)
        mem_pct = float(vm.percent)
    except Exception:
        try:
            load_avg = os.getloadavg()[0]
        except Exception:
            load_avg = None
    return {
        'cpu_pct': cpu_pct,
        'load_avg_1m': load_avg,
        'mem_total_gb': mem_total_gb,
        'mem_used_gb': mem_used_gb,
        'mem_free_gb': mem_free_gb,
        'mem_pct': mem_pct,
    }


def get_system_resources(workspace: Optional[Path] = None) -> Dict[str, Any]:
    ws = Path(workspace or Path.cwd())
    containers = _docker_stats()
    disk = _disk_usage(ws)
    gpu = _gpu_info()
    host = _host_cpu_mem()
    # include a few common subpaths of interest
    extras = {}
    for name, rel in (
        ('grpo', ws / '.grpo'),
        ('vectorized', ws / 'vectorized'),
        ('logs', ws / 'logs'),
    ):
        try:
            extras[name] = _disk_usage(rel)
        except Exception:
            continue
    return {
        'host': {
            'cpu': {k: host.get(k) for k in ('cpu_pct', 'load_avg_1m')},
            'mem': {k: host.get(k) for k in ('mem_total_gb', 'mem_used_gb', 'mem_free_gb', 'mem_pct')},
            'disk': disk,
            'gpu': gpu,
            'timestamp': time.time(),
        },
        'paths': extras,
        'containers': containers,
    }


def guard_resources(
    *,
    workspace: Optional[Path] = None,
    min_free_gb: float = 2.0,
    min_ram_gb: float = 1.0,
    min_vram_mb: float = 0.0,
) -> Dict[str, Any]:
    res = get_system_resources(workspace)
    host = res.get('host', {})
    disk = host.get('disk') or {}
    mem = host.get('mem') or {}
    gpu = host.get('gpu') or []
    disk_ok = float(disk.get('free_gb') or 0.0) >= float(min_free_gb)
    ram_ok = float(mem.get('mem_free_gb') or 0.0) >= float(min_ram_gb)
    if min_vram_mb and gpu:
        v_ok = any(((g.get('mem_total_mb') or 0.0) - (g.get('mem_used_mb') or 0.0)) >= float(min_vram_mb) for g in gpu)
    else:
        v_ok = True
    ok = bool(disk_ok and ram_ok and v_ok)
    return {
        'ok': ok,
        'disk_ok': disk_ok,
        'ram_ok': ram_ok,
        'gpu_ok': v_ok,
        'thresholds': {
            'min_free_gb': float(min_free_gb),
            'min_ram_gb': float(min_ram_gb),
            'min_vram_mb': float(min_vram_mb),
        },
        'snapshot': res,
        'timestamp': time.time(),
    }

