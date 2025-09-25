from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class DirStat:
    path: Path
    bytes: int
    files: int


def _walk_bytes(root: Path) -> Tuple[int, int]:
    total = 0
    files = 0
    try:
        for dirpath, _, filenames in os.walk(root, followlinks=False):
            for name in filenames:
                try:
                    p = Path(dirpath) / name
                    total += p.stat().st_size
                    files += 1
                except Exception:
                    continue
    except Exception:
        pass
    return total, files


def _list_files(root: Path) -> List[Path]:
    out: List[Path] = []
    try:
        for dirpath, _, filenames in os.walk(root, followlinks=False):
            for name in filenames:
                out.append(Path(dirpath) / name)
    except Exception:
        pass
    return out


def _human(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"


class MemoryManager:
    """Simple disk-based memory manager with budgets and compression heuristics.

    - Tracks sizes of common agent directories (logs/, .dspy_reports/, .dspy_cache/)
    - Can trim oldest files to keep within a byte budget (dry-run by default)
    - Can compact .jsonl logs by downsampling to *_compacted.jsonl for older data
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        workspace: Path | None = None,
        include: Optional[Iterable[Path]] = None,
        budget_mb: int = 2048,
        ttl_days: int = 30,
    ) -> None:
        base = workspace or root or Path.cwd()
        self.root = Path(base)
        # Validate workspace/root path
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"Invalid workspace path: {self.root}")
        self.budget_bytes = int(max(1, budget_mb)) * 1024 * 1024
        self.ttl_sec = int(max(1, ttl_days)) * 86400
        self.targets: List[Path] = []
        if include:
            for p in include:
                try:
                    self.targets.append((p if p.is_absolute() else (root / p)).resolve())
                except Exception:
                    continue
        else:
            self.targets = [
                (self.root / 'logs').resolve(),
                (self.root / '.dspy_reports').resolve(),
                (self.root / '.dspy_cache').resolve(),
            ]

    def status(self) -> Dict[str, object]:
        dirs: List[Dict[str, object]] = []
        total = 0
        files_total = 0
        for p in self.targets:
            if not p.exists():
                continue
            b, f = _walk_bytes(p)
            dirs.append({"path": str(p), "bytes": b, "files": f, "human": _human(b)})
            total += b
            files_total += f
        return {
            "root": str(self.root),
            "targets": dirs,
            "total_bytes": total,
            "total_human": _human(total),
            "files": files_total,
            "budget_bytes": self.budget_bytes,
            "budget_human": _human(self.budget_bytes),
            "utilization": (float(total) / float(self.budget_bytes)) if self.budget_bytes > 0 else 0.0,
            "ttl_days": int(self.ttl_sec / 86400),
        }

    def _aged_files(self) -> List[Tuple[float, Path]]:
        now = time.time()
        out: List[Tuple[float, Path]] = []
        for p in self.targets:
            if not p.exists():
                continue
            for f in _list_files(p):
                try:
                    m = f.stat().st_mtime
                    if (now - m) >= self.ttl_sec:
                        out.append((m, f))
                except Exception:
                    continue
        out.sort(key=lambda t: t[0])  # oldest first
        return out

    def trim(self, *, apply: bool = False) -> Dict[str, object]:
        """Plan and optionally remove old files to stay under budget.

        Strategy: remove TTL-expired files by ascending mtime until under budget.
        """
        st = self.status()
        total = int(st.get("total_bytes", 0))
        removed: List[Dict[str, object]] = []
        saved = 0
        if total <= self.budget_bytes:
            return {"action": "noop", "total": total, "budget": self.budget_bytes, "removed": removed, "saved": 0}
        candidates = self._aged_files()
        for mtime, f in candidates:
            if total - saved <= self.budget_bytes:
                break
            try:
                size = f.stat().st_size
            except Exception:
                size = 0
            removed.append({"path": str(f), "bytes": size, "human": _human(size), "mtime": mtime})
            saved += size
            if apply:
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    continue
        return {"action": "trim", "total": total, "budget": self.budget_bytes, "removed": removed, "saved": saved, "saved_human": _human(saved)}

    def compact_logs(self, *, sample_every: int = 10, apply: bool = False, older_than_days: int = 7) -> Dict[str, object]:
        """Downsample .jsonl logs older than threshold into *_compacted.jsonl.

        For each *.jsonl, write a compacted file with ~1/N lines and delete the
        original only when apply=True.
        """
        threshold = time.time() - max(1, older_than_days) * 86400
        changed: List[Dict[str, object]] = []
        for p in self.targets:
            for f in _list_files(p):
                if not f.name.endswith('.jsonl'):
                    continue
                try:
                    m = f.stat().st_mtime
                except Exception:
                    continue
                if m > threshold:
                    continue
                comp = f.with_suffix('.compacted.jsonl')
                try:
                    kept = 0
                    total = 0
                    with f.open('r') as src, comp.open('w') as dst:
                        for i, line in enumerate(src):
                            total += 1
                            if i % max(1, sample_every) == 0:
                                try:
                                    json.loads(line)
                                    dst.write(line)
                                except Exception:
                                    # keep malformed lines sparsely anyway
                                    dst.write(line)
                                kept += 1
                    changed.append({"source": str(f), "compacted": str(comp), "kept": kept, "total": total})
                    if apply:
                        try:
                            f.unlink(missing_ok=True)
                        except Exception:
                            pass
                except Exception:
                    continue
        return {"action": "compact", "changes": changed}

    # --- Simple key-value memory API for tests ---
    def _mem_file(self) -> Path:
        return (self.root / '.dspy_memory.json')

    def _load_mem(self) -> Dict[str, object]:
        p = self._mem_file()
        try:
            if p.exists():
                return json.loads(p.read_text())
        except Exception:
            pass
        return {}

    def _save_mem(self, data: Dict[str, object]) -> None:
        try:
            self._mem_file().write_text(json.dumps(data))
        except Exception:
            pass

    def store(self, key: str, value: object) -> None:
        data = self._load_mem(); data[str(key)] = value; self._save_mem(data)

    def retrieve(self, key: str) -> object | None:
        return self._load_mem().get(str(key))

    def search(self, query: str) -> list[object]:
        q = (query or '').lower(); data = self._load_mem(); out: list[object] = []
        for k, v in data.items():
            if q in str(k).lower() or q in str(v).lower():
                out.append(v)
        return out

    def get_metrics(self) -> Dict[str, object]:
        data = self._load_mem(); total_items = len(data)
        # approximate memory usage as file size
        try:
            usage = self._mem_file().stat().st_size
        except Exception:
            usage = 0
        return {"total_items": total_items, "memory_usage": usage}
