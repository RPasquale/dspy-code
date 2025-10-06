#!/usr/bin/env python3
"""
Parquet file monitor that publishes to Kafka embedding_input topic.
This runs as a background thread in the embed-worker.
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from kafka import KafkaProducer
from typing import Set


def log(msg: str) -> None:
    print(f"[parquet-monitor] {msg}", flush=True)


class ParquetMonitor:
    def __init__(self):
        self.kafka_bootstrap = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.input_topic = os.getenv('EMBED_INPUT_TOPIC', 'embedding_input')
        self.parquet_dir = os.getenv('EMBED_PARQUET_DIR', '/workspace/vectorized/embeddings')
        
        # Create Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=[self.kafka_bootstrap],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3,
            retry_backoff_ms=100
        )
        
        # Track processed files
        self.processed_files: Set[str] = set()
        
        log(f"Initialized with bootstrap={self.kafka_bootstrap}, topic={self.input_topic}, dir={self.parquet_dir}")
    
    def monitor(self):
        """Monitor Parquet directory for new files and publish to Kafka."""
        try:
            while True:
                # Find new Parquet files
                parquet_path = Path(self.parquet_dir)
                if not parquet_path.exists():
                    log(f"Directory {self.parquet_dir} does not exist, waiting...")
                    time.sleep(5)
                    continue
                    
                parquet_files = list(parquet_path.glob("*.parquet"))
                new_files = [f for f in parquet_files if str(f) not in self.processed_files]
                
                if new_files:
                    log(f"Found {len(new_files)} new Parquet files")
                    
                    for file_path in new_files:
                        try:
                            log(f"Processing {file_path}")
                            
                            # Read Parquet file
                            df = pd.read_parquet(file_path)
                            
                            # Convert to JSON and send to Kafka
                            for _, row in df.iterrows():
                                # Handle kafka_ts conversion for Timestamp objects
                                kafka_ts = row.get('kafka_ts', 0)
                                if hasattr(kafka_ts, 'timestamp'):
                                    kafka_ts = int(kafka_ts.timestamp())
                                else:
                                    kafka_ts = int(kafka_ts) if kafka_ts else 0
                                
                                message = {
                                    'topic': row.get('topic', 'code.fs.events'),
                                    'kafka_ts': kafka_ts,
                                    'text': row.get('text', ''),
                                    'doc_id': row.get('doc_id', '')
                                }
                                
                                self.producer.send(self.input_topic, message)
                                log(f"Sent message: {message['text'][:50]}...")
                            
                            # Mark file as processed
                            self.processed_files.add(str(file_path))
                            log(f"Processed {file_path}")
                            
                        except Exception as e:
                            log(f"Error processing {file_path}: {e}")
                            continue
                    
                    # Flush producer
                    self.producer.flush()
                    log(f"Flushed {len(new_files)} files to Kafka")
                
                # Wait before checking again
                time.sleep(10)
                
        except KeyboardInterrupt:
            log("Shutting down...")
        finally:
            self.producer.close()
    
    def start_background(self):
        """Start monitoring in a background thread."""
        import threading
        thread = threading.Thread(target=self.monitor, daemon=True)
        thread.start()
        log("Started parquet monitoring in background thread")
        return thread


def start_parquet_monitor():
    """Start the parquet monitor."""
    monitor = ParquetMonitor()
    return monitor.start_background()


if __name__ == '__main__':
    monitor = ParquetMonitor()
    monitor.monitor()
