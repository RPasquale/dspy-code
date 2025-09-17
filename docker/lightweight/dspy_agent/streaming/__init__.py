"""
Streaming and real-time processing components for DSPy agent.
"""

from .streamkit import *
from .streaming_config import *
from .streaming_kafka import *
from .streaming_runtime import *
from .kafka_log import *
from .log_reader import *

__all__ = [
    'FileTailer', 'Aggregator', 'Worker', 'LocalBus', 'Trainer',
    'StreamConfig', 'KafkaParams', 'WorkerLoop', 
    'get_kafka_logger', 'iter_log_paths', 'read_capped', 'load_logs', 'extract_key_events'
]
