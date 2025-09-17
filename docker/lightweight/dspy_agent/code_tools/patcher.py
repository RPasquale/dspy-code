from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict
import subprocess
import shlex


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


def revert_unified_patch(patch_text: str, cwd: Path) -> Tuple[bool, str]:
    """Attempt to reverse-apply a unified diff patch (undo changes)."""
    cwd = cwd.resolve()
    git = _which("git")
    patch = _which("patch")
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch") as tf:
        tf.write(patch_text)
        tf.flush()
        patch_path = Path(tf.name)
    try:
        if git and (cwd / ".git").exists():
            proc = subprocess.run([git, "apply", "-R", str(patch_path)], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0:
                return True, proc.stdout or "Reverted patch with git apply -R."
            last_err = proc.stderr
        else:
            last_err = "git not available or not a git repo"
        if patch:
            proc = subprocess.run([patch, "-R", "-p0", "-i", str(patch_path)], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0:
                return True, proc.stdout or "Reverted patch with patch -R."
            return False, proc.stderr or "patch -R failed"
        return False, f"No patch tool available. Last git error: {last_err}"
    finally:
        try:
            os.unlink(patch_path)
        except Exception:
            pass


def run_shell(cmd: str, cwd: Path, timeout: int = 600) -> Tuple[int, str, str]:
    """Run a shell command in cwd, return (code, stdout, stderr)."""
    try:
        proc = subprocess.run(cmd if isinstance(cmd, list) else shlex.split(cmd), cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", e.stderr or "timeout"
    except Exception as e:
        return 1, "", str(e)


def git_commit(cwd: Path, message: str) -> Tuple[bool, str]:
    """Attempt to git add/commit changes with message."""
    cwd = cwd.resolve()
    git = _which("git")
    if not git or not (cwd / ".git").exists():
        return False, "Not a git repo or git not available"
    try:
        p1 = subprocess.run([git, "add", "-A"], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p1.returncode != 0:
            return False, p1.stderr
        p2 = subprocess.run([git, "commit", "-m", message], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p2.returncode != 0:
            return False, p2.stderr or "git commit failed"
        return True, p2.stdout or "Committed"
    except Exception as e:
        return False, str(e)


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
