#!/usr/bin/env python3
"""
Demo script showing how the DSPy Agent learns
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def show_learning_state():
    """Show the current learning state"""
    print("🧠 DSPy Agent Learning State")
    print("=" * 40)
    
    # Check for learning files
    state_file = project_root / ".dspy_rl_state.json"
    events_file = project_root / ".dspy_rl_events.jsonl"
    
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            print("📊 Current Learning State:")
            print(f"  - Learning enabled: {state.get('enabled', 'Unknown')}")
            print(f"  - Total events: {state.get('total_events', 0)}")
            print(f"  - Last updated: {state.get('last_updated', 'Unknown')}")
        except Exception as e:
            print(f"  ❌ Error reading state: {e}")
    else:
        print("📊 No learning state found (agent hasn't learned yet)")
    
    if events_file.exists():
        try:
            with open(events_file) as f:
                events = [json.loads(line) for line in f if line.strip()]
            print(f"📝 Learning Events: {len(events)} events recorded")
            
            if events:
                print("  Recent events:")
                for event in events[-3:]:  # Show last 3 events
                    print(f"    - {event.get('timestamp', 'Unknown')}: {event.get('action', 'Unknown')}")
        except Exception as e:
            print(f"  ❌ Error reading events: {e}")
    else:
        print("📝 No learning events found (start using the agent!)")

def demo_learning_commands():
    """Show example commands that help the agent learn"""
    print("\n🎯 Commands That Help the Agent Learn")
    print("=" * 40)
    
    commands = [
        ("plan \"add user authentication\"", "Agent learns to break down tasks"),
        ("grep \"def test_\"", "Agent learns code search patterns"),
        ("esearch \"database connection\"", "Agent learns semantic search"),
        ("edit \"fix the bug\" --apply", "Agent learns what changes work"),
        ("ctx", "Agent learns to extract context"),
        ("stats", "Agent shows learning progress"),
    ]
    
    for cmd, explanation in commands:
        print(f"  {cmd:<30} # {explanation}")

def show_learning_tips():
    """Show tips for effective learning"""
    print("\n💡 Tips for Effective Learning")
    print("=" * 40)
    
    tips = [
        "Give clear, specific tasks (not vague requests)",
        "Use consistent language and patterns",
        "Let the agent try different approaches",
        "Provide feedback through your actions",
        "Use the agent regularly to build learning data",
        "Check learning progress with 'stats' command",
        "Don't micromanage - let it explore and learn",
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")

def show_learning_files():
    """Show what learning files are created"""
    print("\n📁 Learning Files Created")
    print("=" * 40)
    
    files = [
        (".dspy_rl_state.json", "Current learning state and statistics"),
        (".dspy_rl_events.jsonl", "Learning events and rewards"),
        ("logs/", "Detailed execution logs"),
    ]
    
    for filename, description in files:
        filepath = project_root / filename
        status = "✅ Exists" if filepath.exists() else "❌ Not found"
        print(f"  {filename:<25} {status:<12} # {description}")

def main():
    """Main demo function"""
    print("🚀 DSPy Agent Learning Demo")
    print("=" * 50)
    
    show_learning_state()
    demo_learning_commands()
    show_learning_tips()
    show_learning_files()
    
    print("\n🎉 Ready to Start Learning!")
    print("=" * 40)
    print("1. Start the agent: ./scripts/start-agent.sh")
    print("2. Give it tasks to learn from")
    print("3. Check learning progress with 'stats'")
    print("4. Watch it get smarter over time!")
    
    print("\n📚 For more details, see:")
    print("  - AGENT_LEARNING_GUIDE.md")
    print("  - docs/TESTING.md")
    print("  - USAGE_GUIDE.md")

if __name__ == "__main__":
    main()
