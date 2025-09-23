#!/usr/bin/env python3
"""
Repository cleanup utility.

Performs a conservative scan for redundant artifacts (patch rejects, backups,
bytecode caches, and accidental home-folder copies) and optionally deletes them.

Usage:
  python scripts/cleanup_repo.py           # dry run (prints what would be removed)
  python scripts/cleanup_repo.py --apply   # actually remove safe artifacts
  python scripts/cleanup_repo.py --apply --aggressive  # also remove stray Users/ and Library/ trees

What it targets (safe set by default):
  - *.orig, *.rej
  - __pycache__/ and *.pyc, *.pyo
  - .DS_Store
  - repo-root "~/" tree (accidental home-ish folder)

Aggressive (requires --aggressive):
  - repo-root "Users/" tree (if present)
  - repo-root "Library/" tree (if present)

Never touches: source files, tests, docs, docker templates, or datasets.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]


SAFE_FILE_GLOBS: List[str] = [
    "**/*.orig",
    "**/*.rej",
    "**/.DS_Store",
    "**/*.pyc",
    "**/*.pyo",
]

SAFE_DIR_NAMES: List[str] = [
    "__pycache__",
]

ROOT_STRAY_DIRS_SAFE: List[str] = [
    "~",  # accidental home-ish folder in repo root
]

ROOT_STRAY_DIRS_AGGRESSIVE: List[str] = [
    "Users",
    "Library",
]


def _iter_matches(globs: Iterable[str]) -> Iterable[Path]:
    for pat in globs:
        yield from ROOT.glob(pat)


def _collect_candidates(aggressive: bool = False) -> List[Path]:
    cands: List[Path] = []
    # Files
    for p in _iter_matches(SAFE_FILE_GLOBS):
        if p.is_file():
            cands.append(p)
    # Directories by name anywhere
    for p in ROOT.rglob("*"):
        if p.is_dir() and p.name in SAFE_DIR_NAMES:
            cands.append(p)
    # Root-level stray dirs
    for name in ROOT_STRAY_DIRS_SAFE:
        p = ROOT / name
        if p.exists():
            cands.append(p)
    if aggressive:
        for name in ROOT_STRAY_DIRS_AGGRESSIVE:
            p = ROOT / name
            if p.exists():
                cands.append(p)
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in cands:
        try:
            key = p.resolve()
        except Exception:
            key = p
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _remove_path(p: Path) -> None:
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually delete files/directories")
    ap.add_argument("--aggressive", action="store_true", help="Also remove repo-root Users/ and Library/ trees")
    args = ap.parse_args()

    cands = _collect_candidates(aggressive=bool(args.aggressive))
    if not cands:
        print("✅ No removable artifacts found.")
        return 0

    print("🧹 Cleanup candidates (safe):")
    for p in cands:
        rel = p.relative_to(ROOT)
        print(f" - {rel}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to remove the above.")
        return 0

    # Apply deletions
    for p in cands:
        _remove_path(p)
    print("\n✅ Cleanup applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

