from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def apply_unified_patch(patch_text: str, cwd: Path) -> Tuple[bool, str]:
    """Attempt to apply a unified diff patch using git or patch.

    Returns (ok, message).
    """
    cwd = cwd.resolve()
    git = _which("git")
    patch = _which("patch")

    # Write patch to a temp file for tooling to read
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch") as tf:
        tf.write(patch_text)
        tf.flush()
        patch_path = Path(tf.name)

    try:
        if git and (cwd / ".git").exists():
            # Try git apply with whitespace fix
            proc = subprocess.run(
                [git, "apply", "--reject", "--whitespace=fix", str(patch_path)],
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0:
                return True, proc.stdout or "Applied patch with git apply."
            else:
                # fall through to patch
                last_err = proc.stderr
        else:
            last_err = "git not available or not a git repo"

        if patch:
            proc = subprocess.run(
                [patch, "-p0", "-i", str(patch_path)],
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0:
                return True, proc.stdout or "Applied patch with patch(1)."
            return False, proc.stderr or "patch failed"
        return False, f"No patch tool available. Last git error: {last_err}"
    finally:
        try:
            os.unlink(patch_path)
        except Exception:
            pass


def summarize_patch(patch_text: str) -> Dict[str, int]:
    """Summarize a unified diff: count files touched and added/removed lines.

    Returns a dict with keys: files, added_lines, removed_lines.
    """
    files = set()
    added = 0
    removed = 0
    for line in patch_text.splitlines():
        if line.startswith('+++ ') or line.startswith('--- '):
            # ignore /dev/null markers; capture filename tokens
            parts = line.split()
            if len(parts) >= 2 and parts[1] not in {'/dev/null'}:
                files.add(parts[1])
        elif line.startswith('+') and not line.startswith('+++'):
            added += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed += 1
    return {"files": max(0, len(files) // 2) if files else 0, "added_lines": added, "removed_lines": removed}
