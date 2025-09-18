"""
Storage adapters and data models for DSPy Agent.

This package provides comprehensive data management for the DSPy coding agent,
including RedDB storage, caching, migrations, and data models.
"""

from .base import Storage
from .factory import get_storage
from .reddb import RedDBStorage
from .data_models import (
    # Enums
    AgentState, ActionType, Environment,
    
    # Data Models
    EmbeddingVector, SignatureMetrics, VerifierMetrics, TrainingMetrics,
    ActionRecord, LogEntry, ContextState, PatchRecord, RetrievalEventRecord,
    
    # Data Manager
    RedDBDataManager, get_data_manager,
    
    # Utility Functions
    create_embedding_vector, create_log_entry, create_action_record, create_retrieval_event
)
from .enhanced_storage import (
    EnhancedDataManager, get_enhanced_data_manager,
    QueryBuilder, QueryResult, LRUCache,
    get_top_performing_signatures, get_recent_high_reward_actions, get_error_patterns
)
from .migrations import (
    Migration, MigrationManager, get_migration_manager, initialize_database
)

__all__ = [
    # Base storage
    "Storage", "get_storage", "RedDBStorage",
    
    # Enums
    "AgentState", "ActionType", "Environment",
    
    # Data Models
    "EmbeddingVector", "SignatureMetrics", "VerifierMetrics", "TrainingMetrics",
    "ActionRecord", "LogEntry", "ContextState", "PatchRecord", "RetrievalEventRecord",
    
    # Data Managers
    "RedDBDataManager", "get_data_manager",
    "EnhancedDataManager", "get_enhanced_data_manager",
    
    # Query and Cache
    "QueryBuilder", "QueryResult", "LRUCache",
    
    # Migrations
    "Migration", "MigrationManager", "get_migration_manager", "initialize_database",
    
    # Utility Functions
    "create_embedding_vector", "create_log_entry", "create_action_record", "create_retrieval_event",
    "get_top_performing_signatures", "get_recent_high_reward_actions", "get_error_patterns"
]
