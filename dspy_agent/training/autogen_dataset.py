from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import random

from ..embedding.indexer import tokenize
from ..code_tools.code_search import python_extract_symbol
from ..streaming.log_reader import load_logs, extract_key_events
from ..code_tools.code_snapshot import build_code_snapshot


def _iter_files(root: Path, include: Sequence[str] | None = None, exclude: Sequence[str] | None = None) -> Iterable[Path]:
    include = include or ["**/*"]
    exclude = exclude or ["**/.git/**", "**/.venv/**", "**/node_modules/**", "**/dist/**", "**/build/**"]
    seen = set()
    for pat in include:
        for p in root.glob(pat):
            if p.is_file() and not any(p.match(ex) for ex in exclude):
                if p not in seen:
                    seen.add(p)
                    yield p


def extract_identifiers(root: Path, max_files: int = 500) -> List[str]:
    cnt = Counter()
    for i, f in enumerate(_iter_files(root)):
        if i >= max_files:
            break
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        cnt.update(tokenize(text))
    # Filter out overly common tokens
    tokens = [t for t, c in cnt.most_common(200) if len(t) > 2 and not t.isdigit()]
    return tokens[:50]


def list_python_symbols(root: Path, limit: int = 50) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    for f in _iter_files(root, include=["**/*.py"]):
        if len(items) >= limit:
            break
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        for m in re.finditer(r"^(?:def|class)\s+([A-Za-z_]\w*)", text, flags=re.M):
            items.append((f, m.group(1)))
            if len(items) >= limit:
                break
    return items


def extract_error_phrases(events: str, limit: int = 10) -> List[str]:
    lines = events.splitlines()
    phrases: List[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if any(k in ln.lower() for k in ["error", "exception", "failed", "timeout", "not exist", "not found", "traceback"]):
            phrases.append(ln)
    # Pick representative tokens from the phrases
    cnt = Counter()
    for p in phrases:
        for tok in tokenize(p):
            if len(tok) > 2:
                cnt[tok] += 1
    return [t for t, _ in cnt.most_common(limit)]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: Path, rows: List[Dict]):
    ensure_dir(path.parent)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _split_indices(n: int, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train = idx[:n_train]
    val = idx[n_train:n_train + n_val]
    test = idx[n_train + n_val:]
    return train, val, test


def _dedup_rows(rows: List[Dict], key_fields: Optional[Sequence[str]] = None) -> List[Dict]:
    seen: set[str] = set()
    out: List[Dict] = []
    for r in rows:
        if key_fields:
            key = "\u0001".join(str(r.get(k, "")) for k in key_fields)
        else:
            # Fallback: stable JSON for entire row
            key = json.dumps(r, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def split_jsonl_rows(
    rows: List[Dict],
    out_dir: Path,
    stem: str,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    *,
    dedup: bool = True,
    stratify_by: Optional[str] = None,
) -> Dict[str, Path]:
    ensure_dir(out_dir)
    src_rows = _dedup_rows(rows) if dedup else list(rows)
    # Optional stratified splitting
    parts: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    n = len(src_rows)
    if n <= 3 and n > 0:
        # Small-sample guard: put all into train to avoid empty splits
        parts["train"] = list(src_rows)
        parts["val"] = []
        parts["test"] = []
    elif stratify_by:
        groups: Dict[str, List[Dict]] = {}
        for r in src_rows:
            key = str(r.get(stratify_by, "_none"))
            groups.setdefault(key, []).append(r)
        for _, grp in groups.items():
            train_idx, val_idx, test_idx = _split_indices(len(grp), ratios=ratios, seed=seed)
            parts["train"].extend(grp[i] for i in train_idx)
            parts["val"].extend(grp[i] for i in val_idx)
            parts["test"].extend(grp[i] for i in test_idx)
    else:
        train_idx, val_idx, test_idx = _split_indices(len(src_rows), ratios=ratios, seed=seed)
        parts["train"] = [src_rows[i] for i in train_idx]
        parts["val"] = [src_rows[i] for i in val_idx]
        parts["test"] = [src_rows[i] for i in test_idx]

    paths: Dict[str, Path] = {}
    for name, part in parts.items():
        p = out_dir / f"{stem}_{name}.jsonl"
        write_jsonl(p, part)
        paths[name] = p
    return paths


def make_orchestrator_dataset(workspace: Path, logs: Optional[Path], out: Path, max_examples: int = 30) -> Path:
    identifiers = extract_identifiers(workspace)
    sym_list = list_python_symbols(workspace)
    bundle, _ = load_logs([logs or (workspace / 'logs')])
    key = extract_key_events(bundle) if bundle else ""
    errors = extract_error_phrases(key)

    examples: List[Dict] = []
    # Search tasks
    for tok in identifiers[:10]:
        examples.append({
            "query": f"find references to {tok}",
            "workspace": str(workspace),
            "logs": str(logs) if logs else None,
            "targets": [tok],
            "task_type": "orchestrator.search",
        })
    # Extract tasks
    for f, sym in sym_list[:10]:
        rel = str(f.relative_to(workspace)) if f.exists() and workspace in f.parents else str(f)
        examples.append({
            "query": f"extract symbol {sym} from {rel}",
            "workspace": str(workspace),
            "logs": str(logs) if logs else None,
            "targets": [sym],
            "task_type": "orchestrator.extract",
        })
    # Logs analysis tasks
    if errors:
        for e in errors[:5]:
            examples.append({
                "query": f"investigate in logs: {e}",
                "workspace": str(workspace),
                "logs": str(logs) if logs else None,
                "targets": [e.split()[0]],
                "task_type": "orchestrator.logs",
            })
    # Code summary tasks
    examples.append({
        "query": "summarize the codebase architecture",
        "workspace": str(workspace),
        "logs": str(logs) if logs else None,
        "targets": identifiers[:5],
        "task_type": "orchestrator.summary",
    })

    path = out / "orch_train.jsonl"
    write_jsonl(path, examples[:max_examples])
    return path


def make_context_dataset(workspace: Path, logs: Optional[Path], out: Path, max_examples: int = 20) -> Path:
    bundle, _ = load_logs([logs or (workspace / 'logs')])
    key = extract_key_events(bundle) if bundle else ""
    if not key:
        # fallback to code snapshot keywords
        snap = build_code_snapshot(workspace)
        ids = extract_identifiers(workspace)
        rows = [{
            "task": "Summarize logs for debugging",
            "logs_preview": snap[:2000],
            "context_keywords": ids[:5],
            "key_points_keywords": ids[5:10],
            "task_type": "context",
        }]
    else:
        errs = extract_error_phrases(key)
        rows = [{
            "task": "Summarize logs for debugging",
            "logs_preview": key[:4000],
            "context_keywords": errs[:5],
            "key_points_keywords": errs[5:10],
            "task_type": "context",
        }]
    path = out / "context_train.jsonl"
    write_jsonl(path, rows[:max_examples])
    return path


def make_code_dataset(workspace: Path, out: Path, max_examples: int = 10) -> Path:
    rows: List[Dict] = []
    history_path = workspace / '.dspy_patches' / 'history.jsonl'
    if history_path.exists():
        try:
            lines = history_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            lines = []
        for line in reversed(lines):
            if len(rows) >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            patch = rec.get('patch_content') or rec.get('patch') or ''
            if not patch:
                continue
            test_results = rec.get('test_results') if isinstance(rec.get('test_results'), dict) else {}
            row = {
                "task": rec.get('task') or rec.get('prompt_hash') or "code_fix",
                "context": rec.get('context') or rec.get('logs') or "",
                "file_hints": rec.get('target_files') or test_results.get('target_files_summary') or "",
                "patch": patch,
                "reasoning_plan": test_results.get('reasoning_plan', ''),
                "test_plan": test_results.get('test_plan', ''),
                "metrics": {
                    k: v for k, v in test_results.items()
                    if isinstance(k, str) and isinstance(v, (int, float, str))
                },
                "task_type": "code_patch",
            }
            rows.append(row)
    if not rows:
        snap = build_code_snapshot(workspace)
        ids = extract_identifiers(workspace)
        rows = [{"snapshot": snap[:8000], "ask": "Summarize this code snapshot.", "keywords": ids[:10], "task_type": "code"}]
    path = out / "code_train.jsonl"
    write_jsonl(path, rows[:max_examples])
    return path


def make_task_dataset(workspace: Path, logs: Optional[Path], out: Path, max_examples: int = 10) -> Path:
    bundle, _ = load_logs([logs or (workspace / 'logs')])
    key = extract_key_events(bundle) if bundle else ""
    context = key[:4000] if key else build_code_snapshot(workspace)[:4000]
    plan_k = ["check logs", "retry connection", "increase timeout", "apply migration", "run tests"]
    cmd_k = ["uv run", "pytest -q", "make migrate", "git status"]
    rows = [{
        "task": "Investigate failures and propose safe steps.",
        "context": context,
        "plan_keywords": plan_k,
        "commands_keywords": cmd_k,
        "task_type": "task",
    }]
    path = out / "task_train.jsonl"
    write_jsonl(path, rows[:max_examples])
    return path


def bootstrap_datasets(workspace: Path, logs: Optional[Path], out_dir: Optional[Path] = None) -> Dict[str, Path]:
    out = out_dir or (workspace / ".dspy_data")
    ensure_dir(out)
    paths = {
        "orchestrator": make_orchestrator_dataset(workspace, logs, out),
        "context": make_context_dataset(workspace, logs, out),
        "code": make_code_dataset(workspace, out),
        "task": make_task_dataset(workspace, logs, out),
    }
    return paths


def bootstrap_datasets_with_splits(
    workspace: Path,
    logs: Optional[Path],
    out_dir: Optional[Path] = None,
    *,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    dedup: bool = True,
    stratify_by: Optional[str] = "task_type",
) -> Dict[str, Dict[str, Path]]:
    base = out_dir or (workspace / ".dspy_data")
    ensure_dir(base)
    raw = bootstrap_datasets(workspace, logs, base)

    result: Dict[str, Dict[str, Path]] = {}
    for module, raw_path in raw.items():
        rows: List[Dict] = []
        try:
            with open(raw_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception:
            rows = []
        split_dir = base / "splits"
        paths = split_jsonl_rows(rows, split_dir, stem=module, ratios=ratios, seed=seed, dedup=dedup, stratify_by=stratify_by)
        result[module] = paths

    # Write manifest for reproducibility
    manifest = {
        "workspace": str(workspace),
        "logs": str(logs) if logs else None,
        "seed": seed,
        "ratios": list(ratios),
        "dedup": dedup,
        "stratify_by": stratify_by,
        "splits": {k: {n: str(p) for n, p in v.items()} for k, v in result.items()},
    }
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return result
