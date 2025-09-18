from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..streaming.log_reader import extract_key_events, load_logs
from ..code_tools.code_snapshot import build_code_snapshot
from ..db import (
    get_enhanced_data_manager, ContextState, PatchRecord, LogEntry,
    Environment, AgentState, create_log_entry
)
from ..agentic import compute_retrieval_features, load_retrieval_events


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

    def __init__(self, workspace: Path, logs: Optional[Path] = None, environment: Environment = Environment.DEVELOPMENT) -> None:
        self.workspace = Path(workspace).resolve()
        self.logs = Path(logs) if logs else (self.workspace / 'logs')
        self.history_path = self.workspace / '.dspy_patches' / 'history.jsonl'
        self.environment = environment
        self.data_manager = get_enhanced_data_manager()
        
        # Log context manager initialization
        init_log = create_log_entry(
            level="INFO",
            source="context_manager",
            message=f"Context manager initialized for workspace: {self.workspace}",
            context={"workspace": str(self.workspace), "environment": environment.value},
            environment=environment
        )
        self.data_manager.log(init_log)

    def build_patch_context(self, task: str, max_patches: int = 5) -> Dict[str, Any]:
        logs_text = self._recent_logs()
        patches = self._recent_patches(max_patches)
        retrieval_events = self._recent_retrieval_events(max_patches)
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
        if retrieval_events:
            retrieval_lines = []
            for ev in retrieval_events[:max_patches]:
                hits = ev.get('hits') or []
                retrieval_lines.append(f"- {ev.get('query', '')} → {len(hits)} hit(s)")
            segments.append("Recent retrievals:\n" + "\n".join(retrieval_lines))
        combined = ("\n\n".join(segments)).strip()
        stats = self._history_stats(patches)
        unique_hints = list(dict.fromkeys([hint for hint in file_candidates if hint]))
        kg_features = compute_retrieval_features(self.workspace, patches, retrieval_events)
        return {
            'text': combined,
            'logs': logs_text,
            'patches': patches,
            'summary': summary,
            'stats': stats,
            'task': task,
            'file_hints': ','.join(unique_hints),
            'retrieval_events': retrieval_events,
            'kg_features': kg_features,
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

    def agentic_features(self, max_items: int = 10) -> List[float]:
        patches = self._recent_patches(max_items)
        retrieval_events = self._recent_retrieval_events(max_items)
        return compute_retrieval_features(self.workspace, patches, retrieval_events)

    def _recent_retrieval_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        try:
            events.extend(load_retrieval_events(self.workspace, limit=limit))
        except Exception:
            pass
        try:
            dm = get_enhanced_data_manager()
            reddb_events = dm.get_recent_retrieval_events(limit)
            for ev in reddb_events:
                events.append(ev.to_dict())
        except Exception:
            pass
        # Deduplicate by event_id if present
        seen: Dict[str, Dict[str, Any]] = {}
        for ev in events:
            eid = str(ev.get('event_id') or '')
            if eid:
                seen[eid] = ev
            else:
                seen[str(len(seen)) + '_'] = ev
        merged = list(seen.values())
        if len(merged) > limit:
            merged = merged[-limit:]
        return merged
    
    def store_current_context(self, agent_state: AgentState, current_task: Optional[str] = None, 
                             active_files: List[str] = None, recent_actions: List[str] = None) -> ContextState:
        """Store current context state in RedDB"""
        import uuid
        
        if active_files is None:
            active_files = []
        if recent_actions is None:
            recent_actions = []
        
        # Get current performance snapshot
        performance_summary = self.data_manager.get_performance_summary(hours=1)
        performance_snapshot = {
            'avg_signature_performance': performance_summary.get('signature_performance', {}).get('avg_score', 0),
            'avg_verifier_accuracy': performance_summary.get('verifier_performance', {}).get('avg_accuracy', 0),
            'total_actions': performance_summary.get('action_performance', {}).get('total_actions', 0)
        }
        
        # Create context state
        context_state = ContextState(
            context_id=str(uuid.uuid4()),
            timestamp=time.time(),
            agent_state=agent_state,
            current_task=current_task,
            workspace_path=str(self.workspace),
            active_files=active_files,
            recent_actions=recent_actions,
            memory_usage={'workspace_size': self._get_workspace_size()},
            performance_snapshot=performance_snapshot,
            environment=self.environment
        )
        
        # Store in RedDB
        self.data_manager.store_context_state(context_state)
        
        # Log context storage
        agent_state_value = agent_state.value if hasattr(agent_state, 'value') else str(agent_state)
        context_log = create_log_entry(
            level="INFO",
            source="context_manager",
            message=f"Context state stored: {agent_state_value}",
            context={
                "context_id": context_state.context_id,
                "agent_state": agent_state_value,
                "current_task": current_task,
                "active_files_count": len(active_files)
            },
            environment=self.environment
        )
        self.data_manager.log(context_log)
        
        return context_state
    
    def get_current_context(self) -> Optional[ContextState]:
        """Get current context state from RedDB"""
        return self.data_manager.get_current_context()
    
    def get_recent_patches_from_reddb(self, limit: int = 10) -> List[PatchRecord]:
        """Get recent patches from RedDB instead of local files"""
        return self.data_manager.get_patch_history(limit=limit)
    
    def store_patch_record(self, patch_content: str, target_files: List[str], 
                          applied: bool, test_results: Optional[Dict[str, Any]] = None,
                          confidence_score: float = 0.0, blast_radius: float = 0.0) -> PatchRecord:
        """Store a patch record in RedDB"""
        import uuid
        import hashlib
        
        # Generate patch hash for prompt_hash
        prompt_hash = hashlib.md5(patch_content.encode()).hexdigest()
        
        patch_record = PatchRecord(
            patch_id=str(uuid.uuid4()),
            timestamp=time.time(),
            prompt_hash=prompt_hash,
            target_files=target_files,
            patch_content=patch_content,
            applied=applied,
            test_results=test_results,
            confidence_score=confidence_score,
            blast_radius=blast_radius,
            rollback_info=None,  # Can be populated later if needed
            environment=self.environment
        )
        
        # Store in RedDB
        self.data_manager.store_patch_record(patch_record)
        
        # Log patch storage
        patch_log = create_log_entry(
            level="INFO",
            source="context_manager",
            message=f"Patch record stored: {len(target_files)} files, applied: {applied}",
            context={
                "patch_id": patch_record.patch_id,
                "target_files": target_files,
                "applied": applied,
                "confidence_score": confidence_score,
                "blast_radius": blast_radius
            },
            environment=self.environment
        )
        self.data_manager.log(patch_log)
        
        return patch_record
    
    def _get_workspace_size(self) -> int:
        """Get approximate workspace size in bytes"""
        try:
            total_size = 0
            for path in self.workspace.rglob('*'):
                if path.is_file() and not path.name.startswith('.'):
                    try:
                        total_size += path.stat().st_size
                    except (OSError, FileNotFoundError):
                        pass
            return total_size
        except Exception:
            return 0
    
    def build_enhanced_context(self, task: str, agent_state: AgentState = AgentState.ANALYZING) -> Dict[str, Any]:
        """Build enhanced context using both local files and RedDB data"""
        # Store current context state
        context_state = self.store_current_context(agent_state, task)
        
        # Get traditional context
        traditional_context = self.build_patch_context(task)
        
        # Get RedDB patches
        reddb_patches = self.get_recent_patches_from_reddb(limit=10)
        
        # Get recent logs from RedDB
        recent_logs = self.data_manager.get_recent_logs(limit=50)
        
        # Get performance data
        performance_summary = self.data_manager.get_performance_summary(hours=24)
        
        # Combine all context sources
        enhanced_context = {
            **traditional_context,
            'context_state': context_state.to_dict(),
            'reddb_patches': [patch.to_dict() for patch in reddb_patches],
            'recent_reddb_logs': [log.to_dict() for log in recent_logs[-10:]],  # Last 10 logs
            'performance_summary': performance_summary,
            'workspace_size': self._get_workspace_size(),
            'environment': self.environment.value,
            'timestamp': time.time()
        }
        
        return enhanced_context
