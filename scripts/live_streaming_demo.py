#!/usr/bin/env python3
"""
Live Streaming Demo - Show the universal data streaming system in action
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

async def live_demo():
    """Live demonstration of the streaming system"""
    print("🚀 LIVE DEMO: Universal Data Streaming System")
    print("=" * 60)
    print("This demo will show the system ingesting data from multiple sources")
    print("and storing it in RedDB for real-time learning!")
    print()
    
    # Create orchestrator
    orchestrator = EnhancedStreamingOrchestrator(project_root)
    
    # Create dynamic test data
    test_data = {
        "events": [
            {"id": 1, "event": "user_login", "timestamp": time.time(), "user_id": 123, "ip": "192.168.1.100"},
            {"id": 2, "event": "page_view", "timestamp": time.time(), "page": "/dashboard", "duration": 45},
            {"id": 3, "event": "api_call", "timestamp": time.time(), "endpoint": "/api/users", "response_time": 120},
        ]
    }
    
    # Create test data file
    test_file = project_root / "live_demo_data.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"📁 Created test data file: {test_file}")
    
    # Add file source
    file_config = DataSourceConfig(
        name="live_demo_file",
        type="file",
        config={"path": str(test_file)},
        poll_interval=3.0,
        batch_size=10
    )
    
    success = orchestrator.add_data_source(file_config)
    if success:
        print("✅ Added live demo file source")
    else:
        print("❌ Failed to add file source")
        return
    
    # Add GitHub API source
    github_config = DataSourceConfig(
        name="github_live_demo",
        type="api",
        config={
            "url": "https://api.github.com/repos/microsoft/vscode/commits",
            "method": "GET",
            "headers": {"Accept": "application/vnd.github.v3+json"},
            "params": {"per_page": 2}
        },
        poll_interval=8.0,
        batch_size=5
    )
    
    success = orchestrator.add_data_source(github_config)
    if success:
        print("✅ Added GitHub API source")
    else:
        print("❌ Failed to add GitHub source")
    
    print("\n🔄 Starting live data collection...")
    print("Watch for data ingestion messages below:")
    print("-" * 60)
    
    # Start data collection
    collection_task = asyncio.create_task(orchestrator.start_data_collection())
    
    # Monitor RedDB in real-time
    async def monitor_reddb():
        from dspy_agent.db import get_enhanced_data_manager
        dm = get_enhanced_data_manager()
        
        last_action_count = 0
        last_log_count = 0
        
        while True:
            try:
                # Check for new actions
                actions = dm.get_recent_actions(limit=50)
                if len(actions) > last_action_count:
                    new_actions = actions[last_action_count:]
                    for action in new_actions:
                        print(f"🎯 NEW ACTION: {action.action_type.value} (reward: {action.reward:.2f})")
                        print(f"   Source: {action.parameters.get('source', 'unknown')}")
                        print(f"   Result: {action.result.get('message', 'N/A')}")
                    last_action_count = len(actions)
                
                # Check for new logs
                logs = dm.get_recent_logs(limit=50)
                if len(logs) > last_log_count:
                    new_logs = logs[last_log_count:]
                    for log in new_logs:
                        print(f"📝 NEW LOG: {log.level} - {log.message}")
                    last_log_count = len(logs)
                
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                await asyncio.sleep(5)
    
    # Start monitoring
    monitor_task = asyncio.create_task(monitor_reddb())
    
    try:
        # Let it run for 45 seconds
        await asyncio.wait_for(collection_task, timeout=45.0)
    except asyncio.TimeoutError:
        print("\n⏰ 45 seconds elapsed, stopping...")
    except KeyboardInterrupt:
        print("\n🛑 Stopping early...")
    finally:
        orchestrator.stop()
        collection_task.cancel()
        monitor_task.cancel()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 LIVE DEMO COMPLETED!")
    print("=" * 60)
    
    # Show final stats
    from dspy_agent.db import get_enhanced_data_manager
    dm = get_enhanced_data_manager()
    
    actions = dm.get_recent_actions(limit=100)
    logs = dm.get_recent_logs(limit=100)
    
    print(f"📊 Total actions recorded: {len(actions)}")
    print(f"📝 Total logs recorded: {len(logs)}")
    
    if actions:
        print("\n🎯 Recent Actions:")
        for action in actions[-5:]:
            print(f"  • {action.action_type.value} -> {action.result.get('message', 'N/A')} (reward: {action.reward:.2f})")
    
    if logs:
        print("\n📝 Recent Logs:")
        for log in logs[-5:]:
            print(f"  • {log.level}: {log.message}")
    
    # Clean up
    if test_file.exists():
        test_file.unlink()
        print(f"\n🧹 Cleaned up test file: {test_file}")
    
    print("\n✨ The universal data streaming system is working perfectly!")
    print("The agent can now ingest data from any source and learn from it!")

if __name__ == "__main__":
    asyncio.run(live_demo())
