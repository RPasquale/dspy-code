"""
Streaming and real-time processing components for DSPy agent.
"""

from .streamkit import *
from .streaming_config import *
from .streaming_kafka import *
from .streaming_runtime import *
from .kafka_log import *
from .event_bus import EventBus, get_event_bus, memory_tail, memory_delta, memory_last_seq, memory_topics
from .events import (
    publish_event,
    log_ui_action,
    log_backend_api,
    log_agent_action,
    log_ingest_decision,
    log_training_trigger,
    log_training_result,
    log_training_dataset,
    log_spark_app,
    log_spark_log,
)
from .log_reader import *
from .vectorized_pipeline import *
from .feature_store import *

__all__ = [
    'FileTailer', 'Aggregator', 'Worker', 'LocalBus', 'Trainer',
    'StreamConfig', 'KafkaParams', 'WorkerLoop', 
    'VectorizedRecord', 'RLVectorizer', 'VectorizedStreamOrchestrator',
    'KafkaTopicStatus', 'KafkaHealthSnapshot', 'KafkaStreamInspector',
    'SparkCheckpointStatus', 'SparkCheckpointMonitor',
    'FeatureStore', 'FeatureSnapshot',
    'get_kafka_logger', 'EventBus', 'get_event_bus', 'memory_tail', 'memory_delta', 'memory_last_seq', 'memory_topics',
    'publish_event', 'log_ui_action', 'log_backend_api', 'log_agent_action',
    'log_ingest_decision', 'log_training_trigger', 'log_training_result', 'log_training_dataset',
    'log_spark_app', 'log_spark_log',
    'iter_log_paths', 'read_capped', 'load_logs', 'extract_key_events'
]
