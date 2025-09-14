from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple


def _list_files(root: Path, max_files: int = 200) -> List[Path]:
    ex_dirs = {".git", ".venv", "node_modules", "dist", "build", ".mypy_cache", ".pytest_cache"}
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            if p.name in ex_dirs:
                # skip subtree
                for _ in p.rglob("*"):
                    pass
                continue
            continue
        out.append(p)
        if len(out) >= max_files:
            break
    return out


def _read_head_tail(path: Path, max_bytes: int) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    if len(data) <= max_bytes:
        return data.decode(errors="ignore")
    head = data[: max_bytes // 2]
    tail = data[-max_bytes // 2 :]
    return head.decode(errors="ignore") + "\n... [truncated] ...\n" + tail.decode(errors="ignore")


def _py_outline(text: str, max_lines: int = 200) -> List[str]:
    lines = text.splitlines()[:max_lines]
    pat = re.compile(r"^(class|def)\s+([\w_]+)")
    out = []
    for i, line in enumerate(lines, start=1):
        m = pat.match(line.strip())
        if m:
            out.append(f"L{i}: {m.group(1)} {m.group(2)}")
    return out


def build_code_snapshot(target: Path, max_files: int = 50, max_file_bytes: int = 20000) -> str:
    if target.is_file():
        text = _read_head_tail(target, max_file_bytes)
        outline = _py_outline(text) if target.suffix == ".py" else []
        header = f"===== FILE {target} ====="
        blocks = [header]
        if outline:
            blocks.append("-- Outline --\n" + "\n".join(outline))
        blocks.append(text)
        return "\n".join(blocks)

    # Directory snapshot
    files = _list_files(target, max_files=max_files)
    blocks: List[str] = [f"===== DIRECTORY {target} ({len(files)} files sampled) ====="]
    for p in files:
        text = _read_head_tail(p, max_file_bytes // 2)
        outline = _py_outline(text, max_lines=150) if p.suffix == ".py" else []
        blocks.append(f"--- {p} ---")
        if outline:
            blocks.append("Outline:\n" + "\n".join(outline))
        # Limit content included for dirs to keep snapshot compact
        snippet = "\n".join(text.splitlines()[:200])
        blocks.append(snippet)
    return "\n\n".join(blocks)

