#!/usr/bin/env python3
"""
Real-time RL Learning Monitor for DSPy Agent

This script monitors the agent's learning process in real-time,
showing you exactly what the agent is doing and learning.
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class RLLearningMonitor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.state_file = project_root / ".dspy_rl_state.json"
        self.events_file = project_root / ".dspy_rl_events.jsonl"
        self.logs_dir = project_root / "logs"
        
        # Monitoring data
        self.last_state = {}
        self.last_event_count = 0
        self.learning_stats = {
            "total_events": 0,
            "recent_rewards": deque(maxlen=100),
            "action_counts": defaultdict(int),
            "tool_usage": defaultdict(int),
            "success_rate": 0.0,
            "learning_trend": "stable"
        }
        
        # Real-time monitoring
        self.monitoring = False
        self.monitor_thread = None
        
    def load_learning_state(self) -> Dict[str, Any]:
        """Load current learning state"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading state: {e}")
        return {}
    
    def load_learning_events(self) -> List[Dict[str, Any]]:
        """Load learning events"""
        events = []
        if self.events_file.exists():
            try:
                with open(self.events_file) as f:
                    for line in f:
                        if line.strip():
                            events.append(json.loads(line))
            except Exception as e:
                print(f"⚠️  Error loading events: {e}")
        return events
    
    def analyze_learning_progress(self) -> Dict[str, Any]:
        """Analyze learning progress from events"""
        events = self.load_learning_events()
        
        if not events:
            return self.learning_stats
        
        # Update basic stats
        self.learning_stats["total_events"] = len(events)
        
        # Analyze recent events (last 50)
        recent_events = events[-50:] if len(events) > 50 else events
        
        # Count actions and tools
        action_counts = defaultdict(int)
        tool_usage = defaultdict(int)
        rewards = []
        
        for event in recent_events:
            if "action" in event:
                action_counts[event["action"]] += 1
            if "tool" in event:
                tool_usage[event["tool"]] += 1
            if "reward" in event:
                rewards.append(event["reward"])
        
        self.learning_stats["action_counts"] = dict(action_counts)
        self.learning_stats["tool_usage"] = dict(tool_usage)
        self.learning_stats["recent_rewards"] = deque(rewards, maxlen=100)
        
        # Calculate success rate
        if rewards:
            positive_rewards = sum(1 for r in rewards if r > 0)
            self.learning_stats["success_rate"] = positive_rewards / len(rewards)
        
        # Determine learning trend
        if len(rewards) >= 10:
            recent_avg = sum(rewards[-10:]) / 10
            older_avg = sum(rewards[-20:-10]) / 10 if len(rewards) >= 20 else recent_avg
            if recent_avg > older_avg * 1.1:
                self.learning_stats["learning_trend"] = "improving"
            elif recent_avg < older_avg * 0.9:
                self.learning_stats["learning_trend"] = "declining"
            else:
                self.learning_stats["learning_trend"] = "stable"
        
        return self.learning_stats
    
    def display_learning_status(self):
        """Display current learning status"""
        state = self.load_learning_state()
        stats = self.analyze_learning_progress()
        
        print("\n" + "="*80)
        print("🧠 DSPy Agent RL Learning Monitor")
        print("="*80)
        print(f"📊 Total Learning Events: {stats['total_events']}")
        print(f"📈 Success Rate: {stats['success_rate']:.2%}")
        print(f"📊 Learning Trend: {stats['learning_trend']}")
        
        if stats['recent_rewards']:
            avg_reward = sum(stats['recent_rewards']) / len(stats['recent_rewards'])
            print(f"🎯 Average Reward: {avg_reward:.3f}")
        
        print("\n🔧 Tool Usage (Recent):")
        for tool, count in sorted(stats['tool_usage'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {tool}: {count} times")
        
        print("\n⚡ Action Distribution (Recent):")
        for action, count in sorted(stats['action_counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {action}: {count} times")
        
        if state:
            print(f"\n💾 Learning State: {state.get('enabled', 'Unknown')}")
            print(f"🕒 Last Updated: {state.get('last_updated', 'Unknown')}")
        
        print("="*80)
    
    def monitor_learning_events(self):
        """Monitor learning events in real-time"""
        last_event_count = 0
        
        while self.monitoring:
            try:
                events = self.load_learning_events()
                current_count = len(events)
                
                if current_count > last_event_count:
                    new_events = events[last_event_count:]
                    print(f"\n🆕 New Learning Events ({len(new_events)}):")
                    
                    for event in new_events:
                        timestamp = event.get('timestamp', 'Unknown')
                        action = event.get('action', 'Unknown')
                        tool = event.get('tool', 'Unknown')
                        reward = event.get('reward', 0)
                        
                        print(f"  [{timestamp}] {action} ({tool}) → Reward: {reward:.3f}")
                    
                    last_event_count = current_count
                    
                    # Update and display stats
                    self.analyze_learning_progress()
                    self.display_learning_status()
                
                time.sleep(2)  # Check every 2 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(5)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        print("🚀 Starting RL Learning Monitor...")
        print("Press Ctrl+C to stop monitoring")
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_learning_events)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        try:
            while self.monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitor...")
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()
    
    def show_learning_history(self, limit: int = 20):
        """Show recent learning history"""
        events = self.load_learning_events()
        
        if not events:
            print("📝 No learning events found yet.")
            return
        
        print(f"\n📚 Recent Learning History (Last {min(limit, len(events))} events):")
        print("-" * 80)
        
        for event in events[-limit:]:
            timestamp = event.get('timestamp', 'Unknown')
            action = event.get('action', 'Unknown')
            tool = event.get('tool', 'Unknown')
            reward = event.get('reward', 0)
            info = event.get('info', {})
            
            print(f"🕒 {timestamp}")
            print(f"   Action: {action} ({tool})")
            print(f"   Reward: {reward:.3f}")
            if info:
                print(f"   Info: {info}")
            print()
    
    def run_learning_demo(self):
        """Run a demonstration of the learning system"""
        print("🎯 DSPy Agent RL Learning Demonstration")
        print("="*60)
        
        # Show current state
        self.display_learning_status()
        
        # Show history
        self.show_learning_history(10)
        
        # Start monitoring
        print("\n🔄 Starting real-time monitoring...")
        print("💡 Tip: Open another terminal and run the agent to see learning in action!")
        print("   Command: uv run dspy-agent --workspace $(pwd)")
        
        self.start_monitoring()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor DSPy Agent RL Learning")
    parser.add_argument("--history", type=int, default=20, help="Show learning history (default: 20)")
    parser.add_argument("--monitor", "-m", action="store_true", help="Start real-time monitoring")
    parser.add_argument("--demo", "-d", action="store_true", help="Run learning demonstration")
    
    args = parser.parse_args()
    
    monitor = RLLearningMonitor(project_root)
    
    if args.demo:
        monitor.run_learning_demo()
    elif args.monitor:
        monitor.start_monitoring()
    else:
        monitor.display_learning_status()
        monitor.show_learning_history(args.history)

if __name__ == "__main__":
    main()
