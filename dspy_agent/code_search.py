from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple


def _iter_files(root: Path, include_globs: Sequence[str] | None = None, exclude_globs: Sequence[str] | None = None) -> Iterator[Path]:
    include_globs = include_globs or ["**/*"]
    exclude_globs = exclude_globs or [
        "**/.git/**", "**/.venv/**", "**/node_modules/**", "**/dist/**", "**/build/**",
        "**/.mypy_cache/**", "**/.pytest_cache/**",
    ]
    seen: set[Path] = set()
    for pat in include_globs:
        for p in root.glob(pat):
            if p.is_file():
                skip = False
                for ex in exclude_globs:
                    if p.match(ex):
                        skip = True
                        break
                if not skip and p not in seen:
                    seen.add(p)
                    yield p


@dataclass
class LineHit:
    path: Path
    line_no: int
    line: str


def search_text(
    root: Path,
    pattern: str,
    regex: bool = True,
    include_globs: Sequence[str] | None = None,
    exclude_globs: Sequence[str] | None = None,
    encoding: str = "utf-8",
) -> List[LineHit]:
    hits: List[LineHit] = []
    rx = re.compile(pattern) if regex else None
    for f in _iter_files(root, include_globs, exclude_globs):
        try:
            text = f.read_text(encoding=encoding, errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            ok = (rx.search(line) if rx else (pattern in line))
            if ok:
                hits.append(LineHit(path=f, line_no=i, line=line))
    return hits


def search_file(
    path: Path,
    pattern: str,
    regex: bool = True,
    encoding: str = "utf-8",
) -> List[LineHit]:
    try:
        text = path.read_text(encoding=encoding, errors="ignore")
    except Exception:
        return []
    rx = re.compile(pattern) if regex else None
    out: List[LineHit] = []
    for i, line in enumerate(text.splitlines(), start=1):
        ok = (rx.search(line) if rx else (pattern in line))
        if ok:
            out.append(LineHit(path=path, line_no=i, line=line))
    return out


def extract_context(text: str, line_no: int, before: int, after: int) -> Tuple[int, int, str]:
    lines = text.splitlines()
    start = max(1, line_no - before)
    end = min(len(lines), line_no + after)
    segment = "\n".join(lines[start - 1 : end])
    return start, end, segment


def python_extract_symbol(path: Path, name: str, encoding: str = "utf-8") -> Optional[Tuple[int, int, str]]:
    try:
        src = path.read_text(encoding=encoding)
    except Exception:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    match_node: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                match_node = node
                break
    if not match_node:
        return None

    lineno = getattr(match_node, "lineno", None)
    end_lineno = getattr(match_node, "end_lineno", None)
    if not lineno or not end_lineno:
        return None
    lines = src.splitlines()
    segment = "\n".join(lines[lineno - 1 : end_lineno])
    return lineno, end_lineno, segment


def ast_grep_available() -> Optional[str]:
    for exe in ("ast-grep", "sg"):
        if shutil.which(exe):
            return exe
    return None


def run_ast_grep(
    root: Path,
    pattern: Optional[str] = None,
    lang: Optional[str] = None,
    rule_file: Optional[Path] = None,
    json: bool = False,
) -> tuple[int, str, str]:
    exe = ast_grep_available()
    if not exe:
        return 127, "", (
            "ast-grep not found on PATH. Install:\n"
            "  - brew install ast-grep\n  - or: curl -fsSL https://raw.githubusercontent.com/ast-grep/ast-grep/main/install.sh | bash\n"
        )
    cmd = [exe, "-U"]
    if json:
        cmd += ["--json"]
    if rule_file:
        cmd += ["-r", str(rule_file)]
    elif pattern:
        cmd += ["-p", pattern]
    if lang:
        cmd += ["-l", lang]
    cmd += [str(root)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

