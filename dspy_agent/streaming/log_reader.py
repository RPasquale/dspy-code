from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

from ..config import get_settings


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
    # Heuristic: collect error-like lines with light de-noising and compression
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
    # Noise filters: drop extremely spammy warnings
    noise = re.compile(
        r"dspy\.adapters\.json_adapter.*structured output format|json_adapter: Failed to use structured|"
        r"litellm\.Timeout: Connection timed out after 600\.0 seconds|litellm\.exceptions\.Timeout:.*Connection timed out after 600\.0 seconds|httpcore\.ReadTimeout",
        re.IGNORECASE,
    )
    lines = text.splitlines()
    hits: List[str] = []
    for i, line in enumerate(lines):
        if noise.search(line):
            continue
        if rx.search(line):
            # Include some context around the hit
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            # Compress Python stack traces
            block = lines[start:end]
            out_block: List[str] = []
            skip_stack = False
            for j, ln in enumerate(block):
                if skip_stack:
                    if ln.strip() == "":
                        skip_stack = False
                    continue
                out_block.append(ln)
                if ln.strip().startswith("Traceback ("):
                    out_block.append("... stack trace omitted ...")
                    skip_stack = True
            hits.extend(out_block)
            hits.append("")
            if len(hits) >= max_lines:
                break
    if not hits:
        # fallback to the last lines if nothing matched
        hits = lines[-max_lines:]
    # Collapse immediate duplicates
    comp: List[str] = []
    repeat = 0
    for i, ln in enumerate(hits):
        if i > 0 and ln == hits[i - 1]:
            repeat += 1
            continue
        if repeat > 0 and comp:
            comp.append(f"... repeated {repeat} times ...")
            repeat = 0
        comp.append(ln)
    if repeat > 0 and comp:
        comp.append(f"... repeated {repeat} times ...")
    return "\n".join(comp[:max_lines])


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
