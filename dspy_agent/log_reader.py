from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

from .config import get_settings


def iter_log_paths(roots: Iterable[Path]) -> Iterable[Path]:
    exts = {".log", ".out", ".err", ".txt"}
    for root in roots:
        if root.is_file():
            yield root
        elif root.is_dir():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    yield p


def read_capped(path: Path, max_bytes: int) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    if len(data) <= max_bytes:
        return data.decode(errors="ignore")
    # Return head and tail if file is large
    head = data[: max_bytes // 2]
    tail = data[-max_bytes // 2 :]
    return head.decode(errors="ignore") + "\n... [truncated] ...\n" + tail.decode(errors="ignore")


def extract_key_events(text: str, max_lines: int = 120) -> str:
    # Simple heuristic: collect lines with errors, tracebacks, warnings, and timestamps
    patterns = [
        r"\bERROR\b",
        r"\bFATAL\b",
        r"\bEXCEPTION\b",
        r"Traceback \(most recent call last\):",
        r"\bWARNING\b",
        r"\bWARN\b",
        r"\bfailed\b",
        r"\bfailures?\b",
        r"\btimeout\b",
        r"\bnot found\b",
    ]
    rx = re.compile("|".join(patterns), re.IGNORECASE)
    lines = text.splitlines()
    hits: List[str] = []
    for i, line in enumerate(lines):
        if rx.search(line):
            # Include some context around the hit
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            hits.extend(lines[start:end])
            hits.append("")
            if len(hits) >= max_lines:
                break
    if not hits:
        # fallback to the last lines if nothing matched
        hits = lines[-max_lines:]
    return "\n".join(hits)


def load_logs(paths: Iterable[str | Path]) -> Tuple[str, int]:
    settings = get_settings()
    roots = [Path(p) for p in paths]
    collected: List[str] = []
    total = 0
    for p in iter_log_paths(roots):
        content = read_capped(p, settings.max_log_bytes)
        if not content:
            continue
        collected.append(f"===== {p} =====\n" + content)
        total += 1
    bundle = "\n\n".join(collected) if collected else ""
    return bundle, total

