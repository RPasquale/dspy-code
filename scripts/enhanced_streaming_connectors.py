#!/usr/bin/env python3
"""
Enhanced Streaming Connectors - Universal Data Ingestion

This enhances the existing streaming system to support universal data ingestion
from any source, leveraging RedDB for storage and the existing vectorization pipeline.
"""

import os
import sys
import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import existing components
from dspy_agent.streaming import LocalBus, get_kafka_logger
from dspy_agent.db import get_enhanced_data_manager, ActionRecord, LogEntry, create_action_record, create_log_entry
from dspy_agent.streaming.vectorized_pipeline import RLVectorizer, VectorizedStreamOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSourceConfig:
    """Configuration for a data source connector"""
    name: str
    type: str  # api, database, file, webhook, etc.
    config: Dict[str, Any]
    enabled: bool = True
    poll_interval: float = 1.0
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 30.0
    kafka_topic: Optional[str] = None  # Custom topic, defaults to data.source.{name}

class StreamingDataConnector(ABC):
    """Abstract base class for streaming data connectors"""
    
    def __init__(self, source_config: DataSourceConfig, data_manager, kafka_logger):
        self.source_config = source_config
        self.data_manager = data_manager
        self.kafka_logger = kafka_logger
        self.running = False
        self.last_poll = 0
        self.error_count = 0
        self.total_records = 0
        
        # Determine Kafka topic
        self.kafka_topic = source_config.kafka_topic or f"data.source.{source_config.name}"
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data source"""
        pass
    
    @abstractmethod
    async def poll(self) -> List[Dict[str, Any]]:
        """Poll for new data"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the data source"""
        pass
    
    def should_poll(self) -> bool:
        """Check if it's time to poll"""
        return time.time() - self.last_poll >= self.source_config.poll_interval
    
    async def process_data(self, data_list: List[Dict[str, Any]]):
        """Process polled data through the streaming pipeline"""
        if not data_list:
            return
        
        self.total_records += len(data_list)
        
        for data_item in data_list:
            try:
                # 1. Store in RedDB
                await self._store_in_reddb(data_item)
                
                # 2. Publish to Kafka for vectorization
                await self._publish_to_kafka(data_item)
                
                # 3. Create action record for RL training
                await self._create_action_record(data_item)
                
            except Exception as e:
                logger.error(f"❌ Error processing data from {self.source_config.name}: {e}")
    
    async def _store_in_reddb(self, data_item: Dict[str, Any]):
        """Store data in RedDB"""
        try:
            # Create a log entry for the data ingestion
            log_entry = create_log_entry(
                level="INFO",
                source=self.source_config.name,
                message=f"Data ingested from {self.source_config.name}",
                context={
                    "data_type": self.source_config.type,
                    "data_size": len(str(data_item)),
                    "timestamp": time.time()
                }
            )
            
            self.data_manager.log(log_entry)
            
        except Exception as e:
            logger.error(f"❌ Error storing in RedDB: {e}")
    
    async def _publish_to_kafka(self, data_item: Dict[str, Any]):
        """Publish data to Kafka for vectorization"""
        if not self.kafka_logger:
            return
        
        try:
            payload = {
                "source": self.source_config.name,
                "source_type": self.source_config.type,
                "timestamp": time.time(),
                "data": data_item,
                "metadata": {
                    "connector": self.source_config.name,
                    "total_records": self.total_records
                }
            }
            
            self.kafka_logger.publish(self.kafka_topic, payload)
            
        except Exception as e:
            logger.error(f"❌ Error publishing to Kafka: {e}")
    
    async def _create_action_record(self, data_item: Dict[str, Any]):
        """Create action record for RL training"""
        try:
            from dspy_agent.db.data_models import ActionType, Environment
            
            action_record = create_action_record(
                action_type=ActionType.CODE_ANALYSIS,
                state_before={"source": self.source_config.name, "status": "pending"},
                state_after={"source": self.source_config.name, "status": "ingested"},
                parameters={
                    "source": self.source_config.name,
                    "data_type": self.source_config.type,
                    "data_size": len(str(data_item))
                },
                result={"success": True, "message": f"Successfully ingested data from {self.source_config.name}"},
                reward=0.7,  # Default reward for successful ingestion
                confidence=0.8,
                execution_time=0.1,
                environment=Environment.DEVELOPMENT
            )
            
            self.data_manager.record_action(action_record)
            
        except Exception as e:
            logger.error(f"❌ Error creating action record: {e}")

class APIConnector(StreamingDataConnector):
    """Connects to REST/GraphQL APIs"""
    
    async def connect(self) -> bool:
        try:
            import aiohttp
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.source_config.timeout)
            )
            logger.info(f"✅ Connected to API: {self.source_config.name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to API {self.source_config.name}: {e}")
            return False
    
    async def poll(self) -> List[Dict[str, Any]]:
        if not self.should_poll():
            return []
        
        try:
            url = self.source_config.config.get("url")
            method = self.source_config.config.get("method", "GET")
            headers = self.source_config.config.get("headers", {})
            params = self.source_config.config.get("params", {})
            
            async with self.session.request(method, url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.last_poll = time.time()
                    self.error_count = 0
                    
                    # Convert to list of records
                    if isinstance(data, list):
                        return data[:self.source_config.batch_size]
                    else:
                        return [data]
                else:
                    logger.warning(f"API {self.source_config.name} returned status {response.status}")
                    return []
                    
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Error polling API {self.source_config.name}: {e}")
            return []
    
    async def disconnect(self):
        if hasattr(self, 'session'):
            await self.session.close()

class DatabaseConnector(StreamingDataConnector):
    """Connects to databases"""
    
    async def connect(self) -> bool:
        try:
            db_type = self.source_config.config.get("type", "postgresql")
            
            if db_type == "postgresql":
                import asyncpg
                self.connection = await asyncpg.connect(
                    host=self.source_config.config.get("host", "localhost"),
                    port=self.source_config.config.get("port", 5432),
                    user=self.source_config.config.get("user"),
                    password=self.source_config.config.get("password"),
                    database=self.source_config.config.get("database")
                )
            elif db_type == "mongodb":
                import motor.motor_asyncio
                client = motor.motor_asyncio.AsyncIOMotorClient(
                    self.source_config.config.get("connection_string")
                )
                self.connection = client[self.source_config.config.get("database")]
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            logger.info(f"✅ Connected to database: {self.source_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to database {self.source_config.name}: {e}")
            return False
    
    async def poll(self) -> List[Dict[str, Any]]:
        if not self.should_poll():
            return []
        
        try:
            query = self.source_config.config.get("query")
            db_type = self.source_config.config.get("type", "postgresql")
            
            if db_type == "postgresql":
                rows = await self.connection.fetch(query)
                data = [dict(row) for row in rows]
            elif db_type == "mongodb":
                collection = self.connection[self.source_config.config.get("collection")]
                cursor = collection.find(self.source_config.config.get("filter", {}))
                data = await cursor.to_list(length=self.source_config.batch_size)
            
            self.last_poll = time.time()
            self.error_count = 0
            
            return data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Error polling database {self.source_config.name}: {e}")
            return []
    
    async def disconnect(self):
        if hasattr(self, 'connection'):
            if hasattr(self.connection, 'close'):
                await self.connection.close()

class FileConnector(StreamingDataConnector):
    """Connects to file systems"""
    
    async def connect(self) -> bool:
        try:
            self.file_path = Path(self.source_config.config.get("path"))
            if not self.file_path.exists():
                logger.error(f"❌ File path does not exist: {self.file_path}")
                return False
            
            self.last_size = self.file_path.stat().st_size
            logger.info(f"✅ Connected to file: {self.source_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to file {self.source_config.name}: {e}")
            return False
    
    async def poll(self) -> List[Dict[str, Any]]:
        if not self.should_poll():
            return []
        
        try:
            current_size = self.file_path.stat().st_size
            
            if current_size > self.last_size:
                # Read new content
                with open(self.file_path, 'r') as f:
                    f.seek(self.last_size)
                    new_content = f.read(current_size - self.last_size)
                
                self.last_size = current_size
                self.last_poll = time.time()
                self.error_count = 0
                
                # Parse content based on file type
                data = self._parse_content(new_content)
                
                return data
            
            return []
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Error polling file {self.source_config.name}: {e}")
            return []
    
    def _parse_content(self, content: str) -> List[Dict[str, Any]]:
        """Parse file content based on type"""
        file_type = self.file_path.suffix.lower()
        
        if file_type == '.json':
            try:
                return [json.loads(content)]
            except:
                return [{"raw_content": content}]
        elif file_type == '.csv':
            import csv
            from io import StringIO
            reader = csv.DictReader(StringIO(content))
            return list(reader)
        elif file_type in ['.log', '.txt']:
            return [{"line": line.strip()} for line in content.split('\n') if line.strip()]
        else:
            return [{"raw_content": content}]
    
    async def disconnect(self):
        pass

class WebhookConnector(StreamingDataConnector):
    """Connects to webhooks"""
    
    async def connect(self) -> bool:
        try:
            from aiohttp import web
            self.app = web.Application()
            self.app.router.add_post(self.source_config.config.get("path", "/webhook"), self.handle_webhook)
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            site = web.TCPSite(
                self.runner,
                self.source_config.config.get("host", "localhost"),
                self.source_config.config.get("port", 8080)
            )
            await site.start()
            
            logger.info(f"✅ Webhook server started: {self.source_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start webhook server {self.source_config.name}: {e}")
            return False
    
    async def handle_webhook(self, request):
        """Handle incoming webhook data"""
        try:
            data = await request.json()
            self.recent_data = {
                "webhook_data": data,
                "headers": dict(request.headers),
                "timestamp": time.time()
            }
            return web.Response(text="OK")
        except Exception as e:
            logger.error(f"❌ Error handling webhook: {e}")
            return web.Response(text="Error", status=500)
    
    async def poll(self) -> List[Dict[str, Any]]:
        if hasattr(self, 'recent_data'):
            data = self.recent_data
            delattr(self, 'recent_data')
            return [data]
        return []
    
    async def disconnect(self):
        if hasattr(self, 'runner'):
            await self.runner.cleanup()

class EnhancedStreamingOrchestrator:
    """Enhanced orchestrator that manages multiple data sources and integrates with existing streaming"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_manager = get_enhanced_data_manager()
        self.kafka_logger = get_kafka_logger()
        self.connectors: Dict[str, StreamingDataConnector] = {}
        self.running = False
        
        # Integration with existing streaming
        self.bus: Optional[LocalBus] = None
        self.vectorizer: Optional[RLVectorizer] = None
        self.streaming_threads: List[threading.Thread] = []
        
    def add_data_source(self, source_config: DataSourceConfig) -> bool:
        """Add a new data source"""
        try:
            connector = self._create_connector(source_config)
            if connector:
                self.connectors[source_config.name] = connector
                logger.info(f"✅ Added data source: {source_config.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to add data source {source_config.name}: {e}")
            return False
    
    def _create_connector(self, source_config: DataSourceConfig) -> Optional[StreamingDataConnector]:
        """Create appropriate connector for data source"""
        source_type = source_config.type.lower()
        
        if source_type == "api":
            return APIConnector(source_config, self.data_manager, self.kafka_logger)
        elif source_type == "database":
            return DatabaseConnector(source_config, self.data_manager, self.kafka_logger)
        elif source_type == "file":
            return FileConnector(source_config, self.data_manager, self.kafka_logger)
        elif source_type == "webhook":
            return WebhookConnector(source_config, self.data_manager, self.kafka_logger)
        else:
            logger.error(f"❌ Unsupported data source type: {source_type}")
            return None
    
    async def setup_streaming_infrastructure(self):
        """Set up the streaming infrastructure using existing components"""
        try:
            from dspy_agent.streaming import start_local_stack
            
            # Start the existing local streaming stack
            self.streaming_threads, self.bus = start_local_stack(
                root=self.project_root,
                cfg=None,
                storage=None,
                kafka=None
            )
            
            # Get the existing vectorizer from the bus
            self.vectorizer = getattr(self.bus, 'vectorizer', None)
            
            logger.info("✅ Enhanced streaming infrastructure setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup streaming infrastructure: {e}")
            return False
    
    async def start_data_collection(self):
        """Start collecting data from all sources"""
        logger.info("🚀 Starting enhanced data collection from all sources...")
        
        # Connect to all data sources
        for name, connector in self.connectors.items():
            if connector.source_config.enabled:
                success = await connector.connect()
                if not success:
                    logger.warning(f"⚠️  Failed to connect to {name}, skipping")
        
        # Start polling loop
        self.running = True
        while self.running:
            try:
                # Poll all connectors
                for name, connector in self.connectors.items():
                    if connector.source_config.enabled:
                        try:
                            data_list = await connector.poll()
                            if data_list:
                                await connector.process_data(data_list)
                        except Exception as e:
                            logger.error(f"❌ Error polling {name}: {e}")
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error in data collection loop: {e}")
                await asyncio.sleep(1)
    
    async def start(self):
        """Start the enhanced streaming orchestrator"""
        logger.info("🚀 Starting Enhanced Streaming Orchestrator...")
        
        # Setup streaming infrastructure
        if not await self.setup_streaming_infrastructure():
            logger.error("❌ Failed to setup streaming infrastructure")
            return False
        
        # Start data collection
        try:
            await self.start_data_collection()
        except KeyboardInterrupt:
            logger.info("🛑 Stopping Enhanced Streaming Orchestrator...")
            self.running = False
            
            # Disconnect all connectors
            for connector in self.connectors.values():
                await connector.disconnect()
    
    def stop(self):
        """Stop the enhanced streaming orchestrator"""
        self.running = False

def create_sample_data_sources() -> List[DataSourceConfig]:
    """Create sample data sources for testing"""
    return [
        # GitHub API
        DataSourceConfig(
            name="github_commits",
            type="api",
            config={
                "url": "https://api.github.com/repos/microsoft/vscode/commits",
                "method": "GET",
                "headers": {"Accept": "application/vnd.github.v3+json"},
                "params": {"per_page": 10}
            },
            poll_interval=30.0
        ),
        
        # JSON file
        DataSourceConfig(
            name="sample_data",
            type="file",
            config={
                "path": str(project_root / "sample_data.json")
            },
            poll_interval=5.0
        ),
        
        # Webhook
        DataSourceConfig(
            name="webhook_receiver",
            type="webhook",
            config={
                "host": "localhost",
                "port": 8081,
                "path": "/webhook"
            }
        )
    ]

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Streaming Connectors")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--sources", "-s", help="JSON file with data sources")
    parser.add_argument("--demo", "-d", action="store_true", help="Run with demo data sources")
    
    args = parser.parse_args()
    
    orchestrator = EnhancedStreamingOrchestrator(project_root)
    
    if args.demo:
        # Add demo data sources
        demo_sources = create_sample_data_sources()
        for source in demo_sources:
            orchestrator.add_data_source(source)
    
    elif args.sources:
        # Load data sources from file
        try:
            with open(args.sources) as f:
                sources_data = json.load(f)
            
            for source_config in sources_data:
                source = DataSourceConfig(**source_config)
                orchestrator.add_data_source(source)
                
        except Exception as e:
            logger.error(f"❌ Error loading data sources: {e}")
            return
    
    # Start the orchestrator
    await orchestrator.start()

if __name__ == "__main__":
    asyncio.run(main())
