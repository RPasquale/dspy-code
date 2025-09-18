# DSPy Agent RedDB Data Model

## Overview

The DSPy Agent uses a comprehensive data model built on top of RedDB (not Redis) to store all the essential data points for the coding agent system. This includes embeddings, training metrics, logs, environments, states, actions, and performance data.

## Architecture

The data model is organized into several layers:

1. **Storage Layer** (`dbkit.py`) - Low-level RedDB interface with HTTP and in-memory fallback
2. **Data Models** (`data_models.py`) - Structured data classes and schemas
3. **Enhanced Storage** (`enhanced_storage.py`) - High-performance access with caching and querying
4. **Migrations** (`migrations.py`) - Schema versioning and evolution

## Data Models

### Core Enums

- **AgentState**: IDLE, ANALYZING, PLANNING, EXECUTING, TRAINING, ERROR, OPTIMIZING
- **ActionType**: CODE_ANALYSIS, CODE_EDIT, FILE_SEARCH, TEST_EXECUTION, PATCH_GENERATION, OPTIMIZATION, VERIFICATION, CONTEXT_BUILDING
- **Environment**: DEVELOPMENT, TESTING, STAGING, PRODUCTION, LOCAL

### Primary Data Structures

#### EmbeddingVector
Stores vector embeddings for code chunks, documentation, logs, and context:
```python
@dataclass
class EmbeddingVector:
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    timestamp: float
    source_type: str  # 'code', 'documentation', 'log', 'context'
    source_path: Optional[str]
    chunk_start: Optional[int]
    chunk_end: Optional[int]
```

#### SignatureMetrics
Performance metrics for DSPy signatures:
```python
@dataclass
class SignatureMetrics:
    signature_name: str
    performance_score: float
    success_rate: float
    avg_response_time: float
    memory_usage: str
    iterations: int
    last_updated: str
    signature_type: str
    active: bool
    optimization_history: List[Dict[str, Any]]
```

#### VerifierMetrics
Performance data for code verifiers:
```python
@dataclass
class VerifierMetrics:
    verifier_name: str
    accuracy: float
    status: str
    checks_performed: int
    issues_found: int
    last_run: str
    avg_execution_time: float
    false_positive_rate: Optional[float]
    false_negative_rate: Optional[float]
```

#### TrainingMetrics
Training and learning performance data:
```python
@dataclass
class TrainingMetrics:
    session_id: str
    timestamp: float
    epoch: int
    training_accuracy: float
    validation_accuracy: float
    loss: float
    learning_rate: float
    batch_size: int
    model_type: str
    environment: Environment
    hyperparameters: Dict[str, Any]
    convergence_metrics: Dict[str, float]
```

#### ActionRecord
Records of agent actions for RL training:
```python
@dataclass
class ActionRecord:
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
    context_hash: Optional[str]
```

#### LogEntry
Structured log entries:
```python
@dataclass
class LogEntry:
    log_id: str
    timestamp: float
    level: str
    source: str
    message: str
    context: Dict[str, Any]
    environment: Environment
    session_id: Optional[str]
    trace_id: Optional[str]
```

#### ContextState
Agent's current context and state:
```python
@dataclass
class ContextState:
    context_id: str
    timestamp: float
    agent_state: AgentState
    current_task: Optional[str]
    workspace_path: str
    active_files: List[str]
    recent_actions: List[str]
    memory_usage: Dict[str, Any]
    performance_snapshot: Dict[str, float]
    environment: Environment
```

#### PatchRecord
Code patch generation and application records:
```python
@dataclass
class PatchRecord:
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
```

## Storage Patterns

### Key-Value Storage
Used for current state and configuration data:
- `{namespace}:signatures:{signature_name}` - Current signature metrics
- `{namespace}:verifiers:{verifier_name}` - Current verifier metrics
- `{namespace}:context:current` - Current agent context
- `{namespace}:config:{config_type}` - Configuration settings

### Stream Storage
Used for time-series and append-only data:
- `signature_metrics` - Historical signature performance
- `verifier_metrics` - Historical verifier performance
- `training_history` - Training session records
- `rl_actions` - Action records for RL training
- `system_logs` - Application logs
- `context_history` - Context state history
- `patch_history` - Patch application history

### Registry Pattern
Maintains lists of active components:
- `{namespace}:registries:signatures` - List of active signatures
- `{namespace}:registries:verifiers` - List of active verifiers
- `{namespace}:registries:environments` - Available environments
- `{namespace}:registries:actions` - Available action types

## Enhanced Features

### Caching System
- **LRU Cache** with TTL support for frequently accessed data
- **Query Cache** for expensive analytical queries
- **Automatic cache invalidation** on data updates
- **Cache statistics** and monitoring

### Query Builder
Advanced querying capabilities:
```python
query = QueryBuilder(data_manager)
query.filter_by_field("action_type", ActionType.CODE_EDIT)
query.filter_by_range("reward", 0.7, 1.0)
query.sort_by_field("timestamp", reverse=True)
query.limit(10)
result = query.execute(data)
```

### Performance Analytics
- **Performance summaries** across time ranges
- **Trend analysis** for signatures and verifiers
- **Learning progress** tracking
- **Error pattern analysis**

### Migration System
- **Schema versioning** with dependency management
- **Automatic migrations** on startup
- **Rollback capabilities** for safe schema changes
- **Migration history** tracking

## Usage Examples

### Basic Operations
```python
from dspy_agent.db import get_enhanced_data_manager, create_action_record

# Get data manager
dm = get_enhanced_data_manager()

# Store signature metrics
metrics = SignatureMetrics(
    signature_name="CodeAnalyzer",
    performance_score=89.5,
    success_rate=94.2,
    avg_response_time=2.1,
    memory_usage="256MB",
    iterations=150,
    last_updated=datetime.now().isoformat(),
    signature_type="analysis"
)
dm.store_signature_metrics(metrics)

# Record an action
action = create_action_record(
    action_type=ActionType.CODE_ANALYSIS,
    state_before={"files": ["app.py"]},
    state_after={"files": ["app.py"], "analyzed": True},
    parameters={"depth": "full"},
    result={"issues": 3, "suggestions": 5},
    reward=0.85,
    confidence=0.92,
    execution_time=3.4
)
dm.record_action(action)
```

### Advanced Queries
```python
# Get top performing signatures
top_signatures = get_top_performing_signatures(limit=5)

# Get high reward actions from last 24 hours
high_reward_actions = get_recent_high_reward_actions(min_reward=0.8, hours=24)

# Get performance summary
summary = dm.get_performance_summary(hours=24)

# Get learning progress
progress = dm.get_learning_progress(sessions=10)
```

### Cache Management
```python
# Get cache statistics
stats = dm.get_cache_stats()

# Warm cache with commonly accessed data
dm.warm_cache(signatures=["CodeAnalyzer", "TestRunner"])

# Clear cache if needed
dm.clear_cache()
```

## Configuration

The system automatically detects RedDB configuration through environment variables:

```bash
# RedDB Configuration
export REDDB_URL="https://your-reddb-instance.com"
export REDDB_TOKEN="your-auth-token"
export REDDB_NAMESPACE="dspy_agent"

# Database Backend
export DB_BACKEND="reddb"  # or "none" for in-memory fallback
```

## Migration and Initialization

Initialize the database with all migrations:

```bash
python3 scripts/init_reddb.py
```

Or programmatically:

```python
from dspy_agent.db import initialize_database

success = initialize_database()
if success:
    print("Database ready!")
```

## Performance Considerations

1. **Caching**: Frequently accessed data is cached with configurable TTL
2. **Batch Operations**: Use batch methods for bulk data operations
3. **Query Optimization**: Use query builder for complex filtering and sorting
4. **Stream Processing**: Time-series data uses efficient stream storage
5. **Index Management**: Registries provide fast lookups for active components

## Monitoring and Observability

The system provides comprehensive monitoring:

- **Cache hit/miss rates** and performance statistics
- **Query execution times** and optimization opportunities
- **Data volume metrics** across different data types
- **Migration status** and schema version tracking
- **Error patterns** and system health indicators

## Future Enhancements

1. **Vector Similarity Search**: Implement efficient similarity search for embeddings
2. **Data Compression**: Add compression for large data objects
3. **Horizontal Scaling**: Support for distributed RedDB instances
4. **Real-time Analytics**: Stream processing for real-time insights
5. **Data Archival**: Automatic archiving of old data based on retention policies

This comprehensive data model provides a solid foundation for the DSPy coding agent, supporting all aspects of learning, performance tracking, and operational monitoring while maintaining high performance through intelligent caching and optimization strategies.
