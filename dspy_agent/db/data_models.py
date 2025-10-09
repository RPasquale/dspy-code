"""
Comprehensive data models for DSPy Agent using RedDB storage.

This module defines all the data structures and schemas needed for the DSPy coding agent,
including embeddings, training metrics, logs, environments, states, actions, and performance data.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from enum import Enum

from .factory import get_storage


class AgentState(str, Enum):
    """Agent operational states"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    TRAINING = "training"
    ERROR = "error"
    OPTIMIZING = "optimizing"


class ActionType(str, Enum):
    """Types of actions the agent can perform"""
    CODE_ANALYSIS = "code_analysis"
    CODE_EDIT = "code_edit"
    FILE_SEARCH = "file_search"
    TEST_EXECUTION = "test_execution"
    PATCH_GENERATION = "patch_generation"
    OPTIMIZATION = "optimization"
    VERIFICATION = "verification"
    CONTEXT_BUILDING = "context_building"
    TOOL_SELECTION = "tool_selection"


class Environment(str, Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"


@dataclass
class EmbeddingVector:
    """Vector embedding for code chunks or text"""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    timestamp: float
    source_type: str  # 'code', 'documentation', 'log', 'context'
    source_path: Optional[str] = None
    chunk_start: Optional[int] = None
    chunk_end: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingVector':
        return cls(**data)


@dataclass
class SignatureMetrics:
    """Performance metrics for DSPy signatures"""
    signature_name: str
    performance_score: float
    success_rate: float
    avg_response_time: float
    memory_usage: str
    iterations: int
    last_updated: str
    signature_type: str  # 'analysis', 'execution', 'coordination', etc.
    active: bool = True
    optimization_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.optimization_history is None:
            self.optimization_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignatureMetrics':
        return cls(**data)


@dataclass
class VerifierMetrics:
    """Performance metrics for code verifiers"""
    verifier_name: str
    accuracy: float
    status: str
    checks_performed: int
    issues_found: int
    last_run: str
    avg_execution_time: float
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerifierMetrics':
        return cls(**data)


@dataclass
class TrainingMetrics:
    """Training and learning performance data"""
    session_id: str
    timestamp: float
    epoch: int
    training_accuracy: float
    validation_accuracy: float
    loss: float
    learning_rate: float
    batch_size: int
    model_type: str  # 'gepa', 'rl', 'signature_optimizer'
    environment: Environment
    hyperparameters: Dict[str, Any]
    convergence_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['environment'] = self.environment.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetrics':
        data['environment'] = Environment(data['environment'])
        return cls(**data)


@dataclass
class ActionRecord:
    """Record of agent actions for RL training"""
    action_id: str
    timestamp: float
    action_type: ActionType
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    reward: float
    confidence: float
    execution_time: float
    environment: Environment
    context_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['action_type'] = self.action_type.value
        data['environment'] = self.environment.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionRecord':
        data['action_type'] = ActionType(data['action_type'])
        data['environment'] = Environment(data['environment'])
        return cls(**data)


@dataclass
class LogEntry:
    """Structured log entry"""
    log_id: str
    timestamp: float
    level: str  # DEBUG, INFO, WARN, ERROR, CRITICAL
    source: str  # component that generated the log
    message: str
    context: Dict[str, Any]
    environment: Environment
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['environment'] = self.environment.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        data['environment'] = Environment(data['environment'])
        return cls(**data)


@dataclass
class ContextState:
    """Agent's current context and state"""
    context_id: str
    timestamp: float
    agent_state: AgentState
    current_task: Optional[str]
    workspace_path: str
    active_files: List[str]
    recent_actions: List[str]  # Action IDs
    memory_usage: Dict[str, Any]
    performance_snapshot: Dict[str, float]
    environment: Environment
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Handle agent_state - could be enum or already a string/dict
        if hasattr(self.agent_state, 'value'):
            data['agent_state'] = self.agent_state.value
        else:
            data['agent_state'] = self.agent_state
        
        # Handle environment - could be enum or already a string
        if hasattr(self.environment, 'value'):
            data['environment'] = self.environment.value
        else:
            data['environment'] = self.environment
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextState':
        data['agent_state'] = AgentState(data['agent_state'])
        data['environment'] = Environment(data['environment'])
        return cls(**data)


@dataclass
class PatchRecord:
    """Code patch generation and application record"""
    patch_id: str
    timestamp: float
    prompt_hash: str
    target_files: List[str]
    patch_content: str
    applied: bool
    test_results: Optional[Dict[str, Any]]
    confidence_score: float
    blast_radius: float
    rollback_info: Optional[Dict[str, Any]]
    environment: Environment
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['environment'] = self.environment.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatchRecord':
        data['environment'] = Environment(data['environment'])
        return cls(**data)


@dataclass
class RetrievalEventRecord:
    """Agent retrieval events for knowledge graph enrichment."""

    event_id: str
    timestamp: float
    workspace_path: str
    query: str
    hits: List[Dict[str, Any]]
    environment: Environment

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['environment'] = self.environment.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalEventRecord':
        data['environment'] = Environment(data['environment'])
        return cls(**data)


class RedDBDataManager:
    """High-level data access layer for RedDB storage"""
    
    def __init__(self, namespace: str = "dspy_agent"):
        self.storage = get_storage()
        self.namespace = namespace
        
    def _key(self, category: str, identifier: str) -> str:
        """Generate namespaced key"""
        return f"{self.namespace}:{category}:{identifier}"
    
    def _stream_key(self, stream_name: str) -> str:
        """Generate stream key"""
        return f"{self.namespace}:stream:{stream_name}"
    
    # Embedding Operations
    def store_embedding(self, embedding: EmbeddingVector) -> None:
        """Store an embedding vector"""
        key = self._key("embeddings", embedding.id)
        self.storage.put(key, embedding.to_dict())
    
    def get_embedding(self, embedding_id: str) -> Optional[EmbeddingVector]:
        """Retrieve an embedding by ID"""
        key = self._key("embeddings", embedding_id)
        data = self.storage.get(key)
        return EmbeddingVector.from_dict(data) if data else None
    
    def store_embeddings_batch(self, embeddings: List[EmbeddingVector]) -> None:
        """Store multiple embeddings efficiently"""
        for embedding in embeddings:
            self.store_embedding(embedding)
    
    # Signature Metrics Operations
    def store_signature_metrics(self, metrics: SignatureMetrics) -> None:
        """Store signature performance metrics"""
        key = self._key("signatures", metrics.signature_name)
        self.storage.put(key, metrics.to_dict())
        # Keep registry in sync so UI queries can discover this signature without
        # requiring an extra manual registration call.
        self.register_signature(metrics.signature_name)

        # Also append to time series stream
        self.storage.append("signature_metrics", {
            "signature_name": metrics.signature_name,
            "performance_score": metrics.performance_score,
            "timestamp": time.time(),
            "success_rate": metrics.success_rate,
            "avg_response_time": metrics.avg_response_time
        })
    
    def get_signature_metrics(self, signature_name: str) -> Optional[SignatureMetrics]:
        """Get current signature metrics"""
        key = self._key("signatures", signature_name)
        data = self.storage.get(key)
        return SignatureMetrics.from_dict(data) if data else None
    
    def get_all_signature_metrics(self) -> List[SignatureMetrics]:
        """Get all signature metrics (would need key scanning in real implementation)"""
        # This would require RedDB to support key pattern matching
        # For now, we maintain a registry
        registry_key = self._key("registries", "signatures")
        signature_names = self.storage.get(registry_key) or []
        
        metrics = []
        for name in signature_names:
            metric = self.get_signature_metrics(name)
            if metric:
                metrics.append(metric)
        return metrics
    
    def register_signature(self, signature_name: str) -> None:
        """Register a signature in the registry"""
        registry_key = self._key("registries", "signatures")
        signatures = self.storage.get(registry_key) or []
        if signature_name not in signatures:
            signatures.append(signature_name)
            self.storage.put(registry_key, signatures)
    
    # Verifier Metrics Operations
    def store_verifier_metrics(self, metrics: VerifierMetrics) -> None:
        """Store verifier performance metrics"""
        key = self._key("verifiers", metrics.verifier_name)
        self.storage.put(key, metrics.to_dict())
        # Automatically track verifiers in the registry for discovery in the UI.
        self.register_verifier(metrics.verifier_name)

        # Append to time series
        self.storage.append("verifier_metrics", {
            "verifier_name": metrics.verifier_name,
            "accuracy": metrics.accuracy,
            "timestamp": time.time(),
            "checks_performed": metrics.checks_performed,
            "issues_found": metrics.issues_found
        })
    
    def get_verifier_metrics(self, verifier_name: str) -> Optional[VerifierMetrics]:
        """Get current verifier metrics"""
        key = self._key("verifiers", verifier_name)
        data = self.storage.get(key)
        return VerifierMetrics.from_dict(data) if data else None
    
    def get_all_verifier_metrics(self) -> List[VerifierMetrics]:
        """Get all verifier metrics"""
        registry_key = self._key("registries", "verifiers")
        verifier_names = self.storage.get(registry_key) or []
        
        metrics = []
        for name in verifier_names:
            metric = self.get_verifier_metrics(name)
            if metric:
                metrics.append(metric)
        return metrics
    
    def register_verifier(self, verifier_name: str) -> None:
        """Register a verifier in the registry"""
        registry_key = self._key("registries", "verifiers")
        verifiers = self.storage.get(registry_key) or []
        if verifier_name not in verifiers:
            verifiers.append(verifier_name)
            self.storage.put(registry_key, verifiers)
    
    # Training Operations
    def store_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Store training session metrics"""
        key = self._key("training", metrics.session_id)
        self.storage.put(key, metrics.to_dict())
        
        # Append to training history stream
        self.storage.append("training_history", metrics.to_dict())
    
    def get_training_history(self, limit: int = 100) -> List[TrainingMetrics]:
        """Get recent training history"""
        history = []
        for offset, data in self.storage.read("training_history", count=limit):
            history.append(TrainingMetrics.from_dict(data))
        return history
    
    # Action Recording for RL
    def record_action(self, action: ActionRecord) -> None:
        """Record an agent action for RL training"""
        key = self._key("actions", action.action_id)
        self.storage.put(key, action.to_dict())
        
        # Append to action stream for RL processing
        self.storage.append("rl_actions", action.to_dict())
    
    def get_recent_actions(self, limit: int = 100) -> List[ActionRecord]:
        """Get recent actions for RL training"""
        actions = []
        for offset, data in self.storage.read("rl_actions", count=limit):
            actions.append(ActionRecord.from_dict(data))
        return actions
    
    # Logging Operations
    def log(self, entry: LogEntry) -> None:
        """Store a log entry"""
        key = self._key("logs", entry.log_id)
        self.storage.put(key, entry.to_dict())
        
        # Append to log stream
        self.storage.append("system_logs", entry.to_dict())

    def get_recent_logs(self, level: Optional[str] = None, limit: int = 100) -> List[LogEntry]:
        """Get recent log entries, optionally filtered by level"""
        logs = []
        for offset, data in self.storage.read("system_logs", count=limit):
            log_entry = LogEntry.from_dict(data)
            if level is None or log_entry.level == level:
                logs.append(log_entry)
        return logs

    # Retrieval events
    def record_retrieval_event(self, event: RetrievalEventRecord) -> None:
        """Record a retrieval event for knowledge tracking."""

        key = self._key("retrieval", event.event_id)
        self.storage.put(key, event.to_dict())
        self.storage.append("retrieval_events", event.to_dict())

    def get_recent_retrieval_events(self, limit: int = 50) -> List[RetrievalEventRecord]:
        events: List[RetrievalEventRecord] = []
        for offset, data in self.storage.read("retrieval_events", count=limit):
            try:
                events.append(RetrievalEventRecord.from_dict(data))
            except Exception:
                continue
        return events
    
    # Context State Management
    def store_context_state(self, context: ContextState) -> None:
        """Store current agent context state"""
        key = self._key("context", "current")
        self.storage.put(key, context.to_dict())
        
        # Also append to context history
        self.storage.append("context_history", context.to_dict())
    
    def get_current_context(self) -> Optional[ContextState]:
        """Get current agent context"""
        key = self._key("context", "current")
        data = self.storage.get(key)
        return ContextState.from_dict(data) if data else None
    
    # Patch Management
    def store_patch_record(self, patch: PatchRecord) -> None:
        """Store a patch generation/application record"""
        key = self._key("patches", patch.patch_id)
        self.storage.put(key, patch.to_dict())
        
        # Append to patch history stream
        self.storage.append("patch_history", patch.to_dict())
    
    def get_patch_history(self, limit: int = 50) -> List[PatchRecord]:
        """Get recent patch history"""
        patches = []
        for offset, data in self.storage.read("patch_history", count=limit):
            patches.append(PatchRecord.from_dict(data))
        return patches
    
    # Performance Analytics
    def get_performance_trends(self, metric_type: str, hours: int = 24) -> Dict[str, List[float]]:
        """Get performance trends over time"""
        cutoff = time.time() - (hours * 3600)
        trends = {}
        
        stream_name = f"{metric_type}_metrics"
        for offset, data in self.storage.read(stream_name, count=1000):
            timestamp = data.get('timestamp', 0)
            if timestamp >= cutoff:
                for key, value in data.items():
                    if key != 'timestamp' and isinstance(value, (int, float)):
                        if key not in trends:
                            trends[key] = []
                        trends[key].append(value)
        
        return trends
    
    # System Health
    def store_system_health(self, health_data: Dict[str, Any]) -> None:
        """Store system health metrics"""
        key = self._key("health", "current")
        health_data['timestamp'] = time.time()
        self.storage.put(key, health_data)
        
        # Append to health history
        self.storage.append("health_history", health_data)
    
    def get_system_health(self) -> Optional[Dict[str, Any]]:
        """Get current system health"""
        key = self._key("health", "current")
        return self.storage.get(key)


# Convenience functions for common operations
def create_embedding_vector(
    vector_id: str,
    vector: List[float],
    source_type: str,
    metadata: Dict[str, Any],
    source_path: Optional[str] = None,
    chunk_start: Optional[int] = None,
    chunk_end: Optional[int] = None
) -> EmbeddingVector:
    """Create an embedding vector with current timestamp"""
    return EmbeddingVector(
        id=vector_id,
        vector=vector,
        metadata=metadata,
        timestamp=time.time(),
        source_type=source_type,
        source_path=source_path,
        chunk_start=chunk_start,
        chunk_end=chunk_end
    )


def create_log_entry(
    level: str,
    source: str,
    message: str,
    context: Dict[str, Any],
    environment: Environment = Environment.DEVELOPMENT,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None
) -> LogEntry:
    """Create a log entry with auto-generated ID and timestamp"""
    import uuid
    return LogEntry(
        log_id=str(uuid.uuid4()),
        timestamp=time.time(),
        level=level,
        source=source,
        message=message,
        context=context,
        environment=environment,
        session_id=session_id,
        trace_id=trace_id
    )


def create_action_record(
    action_type: ActionType,
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
    parameters: Dict[str, Any],
    result: Dict[str, Any],
    reward: float,
    confidence: float,
    execution_time: float,
    environment: Environment = Environment.DEVELOPMENT,
    context_hash: Optional[str] = None
) -> ActionRecord:
    """Create an action record with auto-generated ID and timestamp"""
    import uuid
    return ActionRecord(
        action_id=str(uuid.uuid4()),
        timestamp=time.time(),
        action_type=action_type,
        state_before=state_before,
        state_after=state_after,
        parameters=parameters,
        result=result,
        reward=reward,
        confidence=confidence,
        execution_time=execution_time,
        environment=environment,
        context_hash=context_hash
    )


def create_retrieval_event(
    workspace_path: str,
    query: str,
    hits: List[Dict[str, Any]],
    environment: Environment = Environment.DEVELOPMENT,
    event_id: Optional[str] = None,
) -> RetrievalEventRecord:
    import uuid
    return RetrievalEventRecord(
        event_id=event_id or str(uuid.uuid4()),
        timestamp=time.time(),
        workspace_path=workspace_path,
        query=query,
        hits=hits,
        environment=environment,
    )


# Global data manager instance
_data_manager: Optional[RedDBDataManager] = None

def get_data_manager() -> RedDBDataManager:
    """Get the global data manager instance"""
    global _data_manager
    if _data_manager is None:
        _data_manager = RedDBDataManager()
    return _data_manager
