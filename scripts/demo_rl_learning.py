#!/usr/bin/env python3
"""
Interactive RL Learning Demonstration for DSPy Agent

This script demonstrates the agent's learning capabilities by running
a series of tasks and showing how the agent learns and improves.
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

class RLLearningDemo:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.state_file = project_root / ".dspy_rl_state.json"
        self.events_file = project_root / ".dspy_rl_events.jsonl"
        
        # Demo tasks designed to show learning
        self.demo_tasks = [
            {
                "name": "Explore Codebase",
                "command": "plan \"explore the project structure and understand the main components\"",
                "description": "Agent learns to break down exploration tasks"
            },
            {
                "name": "Search for Functions",
                "command": "grep \"def test_\"",
                "description": "Agent learns code search patterns"
            },
            {
                "name": "Semantic Search",
                "command": "esearch \"database connection\"",
                "description": "Agent learns semantic search capabilities"
            },
            {
                "name": "Get Context",
                "command": "ctx",
                "description": "Agent learns to extract and summarize context"
            },
            {
                "name": "List Files",
                "command": "ls",
                "description": "Agent learns file system navigation"
            },
            {
                "name": "Show Tree",
                "command": "tree -d 2",
                "description": "Agent learns directory structure visualization"
            }
        ]
        
        # Learning metrics
        self.initial_stats = {}
        self.final_stats = {}
        
    def run_command(self, command: str, timeout: int = 30) -> tuple[bool, str, str]:
        """Run a command and return success, stdout, stderr"""
        try:
            # Use uv run to ensure we're in the right environment
            full_command = f"uv run dspy-agent --workspace {self.project_root} --approval auto"
            
            process = subprocess.run(
                ["bash", "-c", f"echo '{command}' | {full_command}"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_root)
            )
            
            return process.returncode == 0, process.stdout, process.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        stats = {
            "total_events": 0,
            "recent_rewards": [],
            "action_counts": {},
            "tool_usage": {},
            "success_rate": 0.0
        }
        
        if self.events_file.exists():
            try:
                with open(self.events_file) as f:
                    events = [json.loads(line) for line in f if line.strip()]
                
                stats["total_events"] = len(events)
                
                # Analyze recent events
                recent_events = events[-20:] if len(events) > 20 else events
                
                action_counts = {}
                tool_usage = {}
                rewards = []
                
                for event in recent_events:
                    if "action" in event:
                        action_counts[event["action"]] = action_counts.get(event["action"], 0) + 1
                    if "tool" in event:
                        tool_usage[event["tool"]] = tool_usage.get(event["tool"], 0) + 1
                    if "reward" in event:
                        rewards.append(event["reward"])
                
                stats["action_counts"] = action_counts
                stats["tool_usage"] = tool_usage
                stats["recent_rewards"] = rewards
                
                if rewards:
                    positive_rewards = sum(1 for r in rewards if r > 0)
                    stats["success_rate"] = positive_rewards / len(rewards)
                
            except Exception as e:
                print(f"⚠️  Error reading learning stats: {e}")
        
        return stats
    
    def display_stats(self, stats: Dict[str, Any], title: str):
        """Display learning statistics"""
        print(f"\n📊 {title}")
        print("-" * 50)
        print(f"Total Events: {stats['total_events']}")
        print(f"Success Rate: {stats['success_rate']:.2%}")
        
        if stats['recent_rewards']:
            avg_reward = sum(stats['recent_rewards']) / len(stats['recent_rewards'])
            print(f"Average Reward: {avg_reward:.3f}")
        
        if stats['tool_usage']:
            print("\n🔧 Tool Usage:")
            for tool, count in sorted(stats['tool_usage'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool}: {count} times")
        
        if stats['action_counts']:
            print("\n⚡ Action Distribution:")
            for action, count in sorted(stats['action_counts'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {action}: {count} times")
    
    def run_demo_task(self, task: Dict[str, str], task_num: int, total_tasks: int):
        """Run a single demo task"""
        print(f"\n🎯 Task {task_num}/{total_tasks}: {task['name']}")
        print(f"📝 Description: {task['description']}")
        print(f"💻 Command: {task['command']}")
        print("-" * 60)
        
        # Run the command
        success, stdout, stderr = self.run_command(task['command'])
        
        if success:
            print("✅ Task completed successfully")
            if stdout:
                # Show first few lines of output
                lines = stdout.split('\n')[:5]
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
                if len(stdout.split('\n')) > 5:
                    print("   ... (output truncated)")
        else:
            print("❌ Task failed")
            if stderr:
                print(f"   Error: {stderr}")
        
        # Show learning progress
        current_stats = self.get_learning_stats()
        print(f"\n📈 Learning Progress:")
        print(f"   Events: {current_stats['total_events']}")
        print(f"   Success Rate: {current_stats['success_rate']:.2%}")
        
        # Wait a moment for learning to be processed
        time.sleep(2)
    
    def run_full_demo(self):
        """Run the complete learning demonstration"""
        print("🚀 DSPy Agent RL Learning Demonstration")
        print("="*60)
        print("This demo will show you how the agent learns from interactions.")
        print("Watch as the agent's performance improves with each task!")
        print()
        
        # Get initial stats
        self.initial_stats = self.get_learning_stats()
        self.display_stats(self.initial_stats, "Initial Learning State")
        
        input("\nPress Enter to start the demonstration...")
        
        # Run demo tasks
        for i, task in enumerate(self.demo_tasks, 1):
            self.run_demo_task(task, i, len(self.demo_tasks))
            
            if i < len(self.demo_tasks):
                input("\nPress Enter to continue to next task...")
        
        # Get final stats
        self.final_stats = self.get_learning_stats()
        self.display_stats(self.final_stats, "Final Learning State")
        
        # Show learning progress
        print("\n🎉 Learning Demonstration Complete!")
        print("="*60)
        
        events_gained = self.final_stats['total_events'] - self.initial_stats['total_events']
        print(f"📈 Events Gained: {events_gained}")
        
        if self.initial_stats['total_events'] > 0:
            success_improvement = self.final_stats['success_rate'] - self.initial_stats['success_rate']
            print(f"📊 Success Rate Change: {success_improvement:+.2%}")
        
        print("\n💡 The agent has learned from these interactions!")
        print("   - It now understands your coding patterns better")
        print("   - It has improved its tool selection")
        print("   - It has built up learning data for future tasks")
        
        print("\n🔍 To see detailed learning data:")
        print("   uv run python scripts/monitor_rl_learning.py --history 50")
        
        print("\n🔄 To monitor learning in real-time:")
        print("   uv run python scripts/monitor_rl_learning.py --monitor")

def main():
    """Main function"""
    demo = RLLearningDemo(project_root)
    demo.run_full_demo()

if __name__ == "__main__":
    main()
