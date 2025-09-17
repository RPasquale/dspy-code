from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..streaming.log_reader import extract_key_events, load_logs
from ..code_tools.code_snapshot import build_code_snapshot


def _tail_jsonl(path: Path, limit: int = 10) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return []
    items: List[Dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    return items


class ContextManager:
    """Aggregate rich context signals for patch generation and training."""

    def __init__(self, workspace: Path, logs: Optional[Path] = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.logs = Path(logs) if logs else (self.workspace / 'logs')
        self.history_path = self.workspace / '.dspy_patches' / 'history.jsonl'

    def build_patch_context(self, task: str, max_patches: int = 5) -> Dict[str, Any]:
        logs_text = self._recent_logs()
        patches = self._recent_patches(max_patches)
        summary = self._code_summary()
        segments: List[str] = []
        if logs_text:
            segments.append("Recent errors and logs:\n" + logs_text)
        file_candidates: List[str] = []
        if patches:
            patch_lines = []
            for rec in patches:
                ts = rec.get('human_ts') or rec.get('ts')
                pid = rec.get('prompt_id') or rec.get('prompt_hash')
                status = 'pass' if rec.get('high_confidence') else rec.get('result') or 'fail'
                file_hint = rec.get('file_candidates') or ''
                if file_hint:
                    file_candidates.extend([x.strip() for x in str(file_hint).split(',') if x.strip()])
                patch_lines.append(f"- {ts} [{status}] prompt={pid} pass={rec.get('metrics', {}).get('pass_rate', 0)} files={file_hint}")
            segments.append("Recent fixes:\n" + "\n".join(patch_lines))
        if summary:
            segments.append("Code summary:\n" + summary)
        combined = ("\n\n".join(segments)).strip()
        stats = self._history_stats(patches)
        unique_hints = list(dict.fromkeys([hint for hint in file_candidates if hint]))
        return {
            'text': combined,
            'logs': logs_text,
            'patches': patches,
            'summary': summary,
            'stats': stats,
            'task': task,
            'file_hints': ','.join(unique_hints),
        }

    def _recent_logs(self, max_chars: int = 4000) -> str:
        try:
            bundle, _ = load_logs([self.logs])
        except Exception:
            bundle = ""
        key = extract_key_events(bundle) if bundle else ""
        return key[:max_chars]

    def _recent_patches(self, max_items: int = 5) -> List[Dict[str, Any]]:
        rows = _tail_jsonl(self.history_path, limit=max_items)
        patched: List[Dict[str, Any]] = []
        for rec in rows:
            ts = rec.get('timestamp')
            if ts:
                try:
                    human = datetime.fromtimestamp(float(ts)).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    human = str(ts)
            else:
                human = ''
            item = dict(rec)
            item['human_ts'] = human
            patched.append(item)
        return patched

    def _code_summary(self, max_chars: int = 1200) -> str:
        try:
            snap = build_code_snapshot(self.workspace)
            return snap[:max_chars]
        except Exception:
            return ""

    def _history_stats(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        total = 0
        high_conf = 0
        failures = 0
        avg_pass = 0.0
        avg_blast = 0.0
        for rec in rows:
            total += 1
            if rec.get('high_confidence'):
                high_conf += 1
            if rec.get('result') == 'failure' or not rec.get('high_confidence'):
                failures += 1
            metrics = rec.get('metrics') or {}
            try:
                avg_pass += float(metrics.get('pass_rate', 0.0))
            except Exception:
                pass
            try:
                avg_blast += float(metrics.get('blast_radius', 0.0))
            except Exception:
                pass
        if total:
            avg_pass /= total
            avg_blast /= total
        return {
            'total': float(total),
            'high_confidence': float(high_conf),
            'failures': float(failures),
            'recent_success_rate': float(high_conf / total) if total else 0.0,
            'recent_failure_rate': float(failures / total) if total else 0.0,
            'avg_pass_rate': avg_pass,
            'avg_blast_radius': avg_blast,
        }

    def stats_for_features(self) -> Dict[str, float]:
        return self._history_stats(self._recent_patches(max_items=10))

