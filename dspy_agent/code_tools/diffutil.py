from __future__ import annotations

import difflib
from pathlib import Path
from typing import Optional


def unified_diff_from_texts(
    a_text: str,
    b_text: str,
    a_path: str = "a/file",
    b_path: str = "b/file",
    n: int = 3,
) -> str:
    a_lines = a_text.splitlines(keepends=True)
    b_lines = b_text.splitlines(keepends=True)
    diff = difflib.unified_diff(a_lines, b_lines, fromfile=a_path, tofile=b_path, n=n)
    return "".join(diff)


def unified_diff_file_vs_text(file_path: Path, new_text: str, n: int = 3) -> str:
    try:
        old_text = file_path.read_text(errors="ignore")
    except Exception:
        old_text = ""
    return unified_diff_from_texts(old_text, new_text, a_path=str(file_path), b_path=str(file_path), n=n)

