"""
Database migration and schema management for RedDB.

This module handles versioning, migrations, and schema evolution for the DSPy agent's RedDB data.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from .factory import get_storage
from .data_models import get_data_manager


@dataclass
class Migration:
    """Database migration definition"""
    version: str
    description: str
    up_function: Callable[[], None]
    down_function: Optional[Callable[[], None]] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class MigrationManager:
    """Manages database migrations and schema versions"""
    
    def __init__(self, namespace: str = "dspy_agent"):
        self.storage = get_storage()
        self.namespace = namespace
        self.data_manager = get_data_manager()
        self.migrations: Dict[str, Migration] = {}
        
    def _key(self, key: str) -> str:
        """Generate namespaced key"""
        return f"{self.namespace}:migrations:{key}"
    
    def register_migration(self, migration: Migration) -> None:
        """Register a migration"""
        self.migrations[migration.version] = migration
    
    def get_current_version(self) -> Optional[str]:
        """Get the current database version"""
        return self.storage.get(self._key("current_version"))
    
    def set_current_version(self, version: str) -> None:
        """Set the current database version"""
        self.storage.put(self._key("current_version"), version)
        self.storage.append("migration_history", {
            "version": version,
            "timestamp": time.time(),
            "action": "upgrade"
        })
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions"""
        return self.storage.get(self._key("applied_migrations")) or []
    
    def mark_migration_applied(self, version: str) -> None:
        """Mark a migration as applied"""
        applied = self.get_applied_migrations()
        if version not in applied:
            applied.append(version)
            self.storage.put(self._key("applied_migrations"), applied)
    
    def is_migration_applied(self, version: str) -> bool:
        """Check if a migration has been applied"""
        return version in self.get_applied_migrations()
    
    def get_pending_migrations(self) -> List[Migration]:
        """Get migrations that need to be applied"""
        applied = self.get_applied_migrations()
        pending = []
        
        for version, migration in self.migrations.items():
            if version not in applied:
                # Check if dependencies are satisfied
                if all(dep in applied for dep in migration.dependencies):
                    pending.append(migration)
        
        # Sort by version (assuming semantic versioning)
        pending.sort(key=lambda m: m.version)
        return pending
    
    def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration"""
        try:
            print(f"Applying migration {migration.version}: {migration.description}")
            migration.up_function()
            self.mark_migration_applied(migration.version)
            self.set_current_version(migration.version)
            print(f"Migration {migration.version} applied successfully")
            return True
        except Exception as e:
            print(f"Error applying migration {migration.version}: {e}")
            return False
    
    def apply_pending_migrations(self) -> bool:
        """Apply all pending migrations"""
        pending = self.get_pending_migrations()
        
        if not pending:
            print("No pending migrations")
            return True
        
        print(f"Applying {len(pending)} pending migrations")
        
        for migration in pending:
            if not self.apply_migration(migration):
                return False
        
        return True
    
    def rollback_migration(self, version: str) -> bool:
        """Rollback a specific migration"""
        migration = self.migrations.get(version)
        if not migration or not migration.down_function:
            print(f"Cannot rollback migration {version}: no rollback function")
            return False
        
        try:
            print(f"Rolling back migration {version}")
            migration.down_function()
            
            # Remove from applied migrations
            applied = self.get_applied_migrations()
            if version in applied:
                applied.remove(version)
                self.storage.put(self._key("applied_migrations"), applied)
            
            # Log rollback
            self.storage.append("migration_history", {
                "version": version,
                "timestamp": time.time(),
                "action": "rollback"
            })
            
            print(f"Migration {version} rolled back successfully")
            return True
        except Exception as e:
            print(f"Error rolling back migration {version}: {e}")
            return False


# Migration functions
def migration_001_initial_schema():
    """Initial schema setup"""
    from .data_models import Environment
    
    dm = get_data_manager()
    
    # Initialize registries
    dm.storage.put(f"{dm.namespace}:registries:signatures", [])
    dm.storage.put(f"{dm.namespace}:registries:verifiers", [])
    dm.storage.put(f"{dm.namespace}:registries:environments", [env.value for env in Environment])
    
    # Initialize system health
    dm.store_system_health({
        "status": "healthy",
        "version": "1.0.0",
        "initialized": True
    })
    
    print("Initial schema created")


def migration_002_add_embedding_indices():
    """Add embedding vector indices and metadata"""
    dm = get_data_manager()
    
    # Create embedding index registry
    dm.storage.put(f"{dm.namespace}:registries:embeddings", [])
    
    # Initialize embedding metadata
    dm.storage.put(f"{dm.namespace}:embeddings:metadata", {
        "total_vectors": 0,
        "vector_dimension": 0,
        "last_updated": time.time()
    })
    
    print("Embedding indices added")


def migration_003_add_rl_training_streams():
    """Add reinforcement learning training streams"""
    from .data_models import ActionType
    
    dm = get_data_manager()
    
    # Initialize RL configuration
    rl_config = {
        "enabled": True,
        "reward_scaling": 1.0,
        "exploration_rate": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32,
        "memory_size": 10000
    }
    
    dm.storage.put(f"{dm.namespace}:config:rl", rl_config)
    
    # Initialize action space registry
    dm.storage.put(f"{dm.namespace}:registries:actions", [action.value for action in ActionType])
    
    print("RL training streams added")


def migration_004_add_performance_analytics():
    """Add performance analytics and monitoring"""
    dm = get_data_manager()
    
    # Initialize performance baselines
    baselines = {
        "signature_performance_threshold": 85.0,
        "verifier_accuracy_threshold": 90.0,
        "response_time_threshold": 5.0,
        "success_rate_threshold": 95.0
    }
    
    dm.storage.put(f"{dm.namespace}:config:performance", baselines)
    
    # Initialize alert thresholds
    alerts = {
        "error_rate_threshold": 0.05,
        "memory_usage_threshold": 0.85,
        "disk_usage_threshold": 0.90,
        "response_time_p95_threshold": 10.0
    }
    
    dm.storage.put(f"{dm.namespace}:config:alerts", alerts)
    
    print("Performance analytics added")


def migration_005_add_context_caching():
    """Add context caching and optimization"""
    dm = get_data_manager()
    
    # Initialize context cache configuration
    cache_config = {
        "enabled": True,
        "max_size": 1000,
        "ttl_seconds": 3600,
        "compression_enabled": True
    }
    
    dm.storage.put(f"{dm.namespace}:config:context_cache", cache_config)
    
    # Initialize context templates
    templates = {
        "code_analysis": {
            "required_fields": ["file_path", "content", "language"],
            "optional_fields": ["dependencies", "imports", "exports"]
        },
        "patch_generation": {
            "required_fields": ["target_files", "change_description", "test_files"],
            "optional_fields": ["rollback_plan", "impact_analysis"]
        }
    }
    
    dm.storage.put(f"{dm.namespace}:config:context_templates", templates)
    
    print("Context caching added")


# Create migration manager instance and register migrations
def get_migration_manager() -> MigrationManager:
    """Get configured migration manager with all migrations registered"""
    manager = MigrationManager()
    
    # Register all migrations
    manager.register_migration(Migration(
        version="0.1.0",
        description="Initial schema setup",
        up_function=migration_001_initial_schema
    ))
    
    manager.register_migration(Migration(
        version="0.2.0",
        description="Add embedding indices and metadata",
        up_function=migration_002_add_embedding_indices,
        dependencies=["0.1.0"]
    ))
    
    manager.register_migration(Migration(
        version="0.3.0",
        description="Add RL training streams",
        up_function=migration_003_add_rl_training_streams,
        dependencies=["0.2.0"]
    ))
    
    manager.register_migration(Migration(
        version="0.4.0",
        description="Add performance analytics",
        up_function=migration_004_add_performance_analytics,
        dependencies=["0.3.0"]
    ))
    
    manager.register_migration(Migration(
        version="0.5.0",
        description="Add context caching and optimization",
        up_function=migration_005_add_context_caching,
        dependencies=["0.4.0"]
    ))
    
    return manager


def initialize_database() -> bool:
    """Initialize database with all migrations"""
    try:
        manager = get_migration_manager()
        return manager.apply_pending_migrations()
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False


if __name__ == "__main__":
    # Run migrations when called directly
    print("Initializing DSPy Agent database...")
    success = initialize_database()
    if success:
        print("Database initialization completed successfully")
    else:
        print("Database initialization failed")
        exit(1)
