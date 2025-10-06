#!/bin/bash
set -e

echo "[parquet-to-kafka] Starting parquet-to-kafka service..."

# Wait for Kafka to be ready
echo "[parquet-to-kafka] Waiting for Kafka to be ready..."
while ! nc -z kafka 9092; do
  echo "[parquet-to-kafka] Kafka not ready, waiting..."
  sleep 2
done
echo "[parquet-to-kafka] Kafka is ready!"

# Wait for Parquet directory to exist
PARQUET_DIR="/workspace/vectorized/embeddings"
echo "[parquet-to-kafka] Waiting for Parquet directory: $PARQUET_DIR"
while [ ! -d "$PARQUET_DIR" ]; do
  echo "[parquet-to-kafka] Parquet directory not found, waiting..."
  sleep 5
done
echo "[parquet-to-kafka] Parquet directory found!"

# Start the parquet-to-kafka service
echo "[parquet-to-kafka] Starting parquet-to-kafka service..."
cd /app
python -c "
import os
import json
import time
import pandas as pd
from kafka import KafkaProducer
from pathlib import Path

def main():
    # Configuration
    kafka_bootstrap = os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092')
    input_topic = os.getenv('SINK_INPUT_TOPIC', 'embedding_input')
    parquet_dir = os.getenv('PARQUET_DIR', '/workspace/vectorized/embeddings')
    
    print(f'[parquet-to-kafka] Starting with bootstrap={kafka_bootstrap}, topic={input_topic}, dir={parquet_dir}')
    
    # Create Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=[kafka_bootstrap],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        retries=3,
        retry_backoff_ms=100
    )
    
    # Track processed files
    processed_files = set()
    
    try:
        while True:
            # Find new Parquet files
            parquet_path = Path(parquet_dir)
            if not parquet_path.exists():
                print(f'[parquet-to-kafka] Directory {parquet_dir} does not exist, waiting...')
                time.sleep(5)
                continue
                
            parquet_files = list(parquet_path.glob('*.parquet'))
            new_files = [f for f in parquet_files if str(f) not in processed_files]
            
            if new_files:
                print(f'[parquet-to-kafka] Found {len(new_files)} new Parquet files')
                
                for file_path in new_files:
                    try:
                        print(f'[parquet-to-kafka] Processing {file_path}')
                        
                        # Read Parquet file
                        df = pd.read_parquet(file_path)
                        
                        # Convert to JSON and send to Kafka
                        for _, row in df.iterrows():
                            message = {
                                'topic': row.get('topic', 'code.fs.events'),
                                'kafka_ts': int(row.get('kafka_ts', 0)),
                                'text': row.get('text', ''),
                                'doc_id': row.get('doc_id', '')
                            }
                            
                            producer.send(input_topic, message)
                            print(f'[parquet-to-kafka] Sent message: {message[\"text\"][:50]}...')
                        
                        # Mark file as processed
                        processed_files.add(str(file_path))
                        print(f'[parquet-to-kafka] Processed {file_path}')
                        
                    except Exception as e:
                        print(f'[parquet-to-kafka] Error processing {file_path}: {e}')
                        continue
                
                # Flush producer
                producer.flush()
                print(f'[parquet-to-kafka] Flushed {len(new_files)} files to Kafka')
            
            # Wait before checking again
            time.sleep(10)
            
    except KeyboardInterrupt:
        print('[parquet-to-kafka] Shutting down...')
    finally:
        producer.close()

if __name__ == '__main__':
    main()
"
