#!/usr/bin/env python3
"""
Watch the DSPy Agent Learn in Real-Time

This script shows you exactly what the agent is learning by running
commands and displaying the learning events as they happen.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def get_learning_events() -> List[Dict[str, Any]]:
    """Get all learning events"""
    events_file = project_root / ".dspy_rl_events.jsonl"
    events = []
    if events_file.exists():
        try:
            with open(events_file) as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        except Exception as e:
            print(f"⚠️  Error reading events: {e}")
    return events

def run_agent_command(command: str) -> bool:
    """Run a command with the agent"""
    try:
        full_command = f"uv run dspy-agent --workspace {project_root} --approval auto"
        
        process = subprocess.run(
            ["bash", "-c", f"echo '{command}' | {full_command}"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(project_root)
        )
        
        return process.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def show_learning_events(events: List[Dict[str, Any]], start_index: int = 0):
    """Show learning events starting from a specific index"""
    for i, event in enumerate(events[start_index:], start_index):
        timestamp = event.get('timestamp', 'Unknown')
        action = event.get('action', 'Unknown')
        tool = event.get('tool', 'Unknown')
        reward = event.get('reward', 0)
        info = event.get('info', {})
        
        print(f"📊 Event {i+1}:")
        print(f"   🕒 Time: {timestamp}")
        print(f"   ⚡ Action: {action}")
        print(f"   🔧 Tool: {tool}")
        print(f"   🎯 Reward: {reward:.3f}")
        if info:
            print(f"   📝 Info: {info}")
        print()

def main():
    """Main function"""
    print("🚀 Watch the DSPy Agent Learn in Real-Time")
    print("="*60)
    
    # Get initial events
    initial_events = get_learning_events()
    print(f"📊 Initial learning events: {len(initial_events)}")
    
    if initial_events:
        print("\n📚 Recent learning history:")
        show_learning_events(initial_events[-5:])  # Show last 5 events
    
    # Demo commands to show learning
    demo_commands = [
        "help",
        "ls",
        "grep \"def test_\"",
        "esearch \"database\"",
        "ctx"
    ]
    
    print("🎯 Running demo commands to show learning...")
    print("Watch as new learning events are created!")
    print()
    
    for i, command in enumerate(demo_commands, 1):
        print(f"💻 Command {i}: {command}")
        
        # Run the command
        success = run_agent_command(command)
        
        if success:
            print("✅ Command completed")
        else:
            print("❌ Command failed")
        
        # Wait for learning events to be processed
        time.sleep(3)
        
        # Get updated events
        current_events = get_learning_events()
        new_events = current_events[len(initial_events):]
        
        if new_events:
            print(f"🆕 New learning events created: {len(new_events)}")
            show_learning_events(new_events)
        else:
            print("⚠️  No new learning events created")
        
        print("-" * 60)
        
        if i < len(demo_commands):
            input("Press Enter to continue to next command...")
    
    # Final summary
    final_events = get_learning_events()
    total_new_events = len(final_events) - len(initial_events)
    
    print("🎉 Learning Demonstration Complete!")
    print("="*60)
    print(f"📊 Total new learning events: {total_new_events}")
    print(f"📈 Total learning events: {len(final_events)}")
    
    if final_events:
        # Calculate success rate
        rewards = [e.get('reward', 0) for e in final_events if 'reward' in e]
        if rewards:
            positive_rewards = sum(1 for r in rewards if r > 0)
            success_rate = positive_rewards / len(rewards)
            print(f"🎯 Success rate: {success_rate:.1%}")
            
            avg_reward = sum(rewards) / len(rewards)
            print(f"📊 Average reward: {avg_reward:.3f}")
        
        # Show tool usage
        tools = [e.get('tool', 'unknown') for e in final_events if 'tool' in e]
        if tools:
            from collections import Counter
            tool_counts = Counter(tools)
            print(f"\n🔧 Tool usage:")
            for tool, count in tool_counts.most_common(5):
                print(f"   {tool}: {count} times")
    
    print("\n💡 The agent is learning from every interaction!")
    print("   - It tracks which tools work best for different tasks")
    print("   - It learns from rewards and feedback")
    print("   - It improves its decision-making over time")
    
    print("\n🔍 To monitor learning in real-time:")
    print("   uv run python scripts/monitor_rl_learning.py --monitor")
    
    print("\n🧪 To run comprehensive learning tests:")
    print("   uv run python scripts/test_rl_learning.py")

if __name__ == "__main__":
    main()
