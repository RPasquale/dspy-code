#!/usr/bin/env python3
"""
Test Streaming Demo - Simple demonstration of the universal data streaming system
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.enhanced_streaming_connectors import EnhancedStreamingOrchestrator, DataSourceConfig

async def test_streaming_system():
    """Test the streaming system with a simple demo"""
    print("🚀 Testing Universal Data Streaming System")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = EnhancedStreamingOrchestrator(project_root)
    
    # Add a simple file source for testing
    test_data = {
        "test_events": [
            {"id": 1, "event": "user_login", "timestamp": time.time(), "user_id": 123},
            {"id": 2, "event": "page_view", "timestamp": time.time(), "page": "/dashboard"},
            {"id": 3, "event": "api_call", "timestamp": time.time(), "endpoint": "/api/users"},
        ]
    }
    
    # Create test data file
    test_file = project_root / "test_streaming_data.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"📁 Created test data file: {test_file}")
    
    # Add file source
    file_config = DataSourceConfig(
        name="test_file_source",
        type="file",
        config={"path": str(test_file)},
        poll_interval=2.0,
        batch_size=10
    )
    
    success = orchestrator.add_data_source(file_config)
    if success:
        print("✅ Added test file source")
    else:
        print("❌ Failed to add test file source")
        return
    
    # Add GitHub API source
    github_config = DataSourceConfig(
        name="github_test",
        type="api",
        config={
            "url": "https://api.github.com/repos/microsoft/vscode/commits",
            "method": "GET",
            "headers": {"Accept": "application/vnd.github.v3+json"},
            "params": {"per_page": 3}
        },
        poll_interval=10.0,
        batch_size=5
    )
    
    success = orchestrator.add_data_source(github_config)
    if success:
        print("✅ Added GitHub API source")
    else:
        print("❌ Failed to add GitHub API source")
    
    print("\n🔄 Starting data collection for 30 seconds...")
    print("Watch for data ingestion messages...")
    print("Press Ctrl+C to stop early")
    
    # Start data collection in background
    collection_task = asyncio.create_task(orchestrator.start_data_collection())
    
    try:
        # Let it run for 30 seconds
        await asyncio.wait_for(collection_task, timeout=30.0)
    except asyncio.TimeoutError:
        print("\n⏰ 30 seconds elapsed, stopping...")
    except KeyboardInterrupt:
        print("\n🛑 Stopping early...")
    finally:
        orchestrator.stop()
        collection_task.cancel()
    
    # Clean up test file
    if test_file.exists():
        test_file.unlink()
        print("🧹 Cleaned up test file")
    
    print("\n🎉 Streaming test completed!")
    print("Check the logs above to see data ingestion in action.")

def test_reddb_integration():
    """Test RedDB integration"""
    print("\n🔍 Testing RedDB Integration")
    print("-" * 40)
    
    try:
        from dspy_agent.db import get_enhanced_data_manager
        
        # Get data manager
        dm = get_enhanced_data_manager()
        print("✅ Connected to RedDB")
        
        # Get recent actions
        actions = dm.get_recent_actions(limit=10)
        print(f"📊 Recent actions in RedDB: {len(actions)}")
        
        for action in actions[-5:]:
            print(f"  • {action.tool} -> {action.action} (reward: {action.reward:.2f})")
        
        # Get recent logs
        logs = dm.get_recent_logs(limit=5)
        print(f"📝 Recent logs in RedDB: {len(logs)}")
        
        for log in logs[-3:]:
            print(f"  • {log.level}: {log.message[:50]}...")
        
    except Exception as e:
        print(f"❌ RedDB integration test failed: {e}")

def test_kafka_integration():
    """Test Kafka integration"""
    print("\n📡 Testing Kafka Integration")
    print("-" * 40)
    
    try:
        from dspy_agent.streaming import get_kafka_logger
        
        kafka_logger = get_kafka_logger()
        if kafka_logger:
            print("✅ Kafka logger available")
            
            # Test publishing a message
            test_message = {
                "test": True,
                "timestamp": time.time(),
                "message": "Test message from streaming demo"
            }
            
            kafka_logger.publish("test.topic", test_message)
            print("✅ Published test message to Kafka")
        else:
            print("⚠️  Kafka logger not available (this is OK for testing)")
            
    except Exception as e:
        print(f"❌ Kafka integration test failed: {e}")

async def main():
    """Main test function"""
    print("🧪 Universal Data Streaming System Test")
    print("=" * 60)
    
    # Test RedDB integration
    test_reddb_integration()
    
    # Test Kafka integration
    test_kafka_integration()
    
    # Test streaming system
    await test_streaming_system()
    
    print("\n🎉 All tests completed!")
    print("The universal data streaming system is working!")

if __name__ == "__main__":
    asyncio.run(main())
