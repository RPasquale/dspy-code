#!/usr/bin/env python3
"""
Real-Time Agent Monitor

This script provides real-time monitoring of the DSPy Agent's actions,
thoughts, and learning progress through a rich CLI interface.
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import deque
import subprocess

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class AgentMonitor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.events_file = project_root / ".dspy_rl_events.jsonl"
        self.state_file = project_root / ".dspy_rl_state.json"
        self.logs_dir = project_root / "logs"
        
        # Monitoring data
        self.recent_actions = deque(maxlen=50)
        self.recent_thoughts = deque(maxlen=50)
        self.learning_stats = {}
        self.agent_status = "idle"
        self.current_task = None
        
        # Real-time monitoring
        self.monitoring = False
        self.monitor_thread = None
        
        # Rich display components
        self.setup_display()
    
    def setup_display(self):
        """Set up the display components"""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.live import Live
            from rich.layout import Layout
            from rich.text import Text
            from rich.progress import Progress, SpinnerColumn, TextColumn
            from rich.align import Align
            
            self.console = Console()
            self.layout = Layout()
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            )
            
            # Configure layout
            self.layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main", ratio=1),
                Layout(name="footer", size=3)
            )
            
            self.layout["main"].split_row(
                Layout(name="left", ratio=1),
                Layout(name="right", ratio=1)
            )
            
            self.layout["left"].split_column(
                Layout(name="actions", ratio=1),
                Layout(name="thoughts", ratio=1)
            )
            
            self.layout["right"].split_column(
                Layout(name="stats", ratio=1),
                Layout(name="learning", ratio=1)
            )
            
        except ImportError:
            print("⚠️  Rich library not available. Install with: pip install rich")
            self.console = None
            self.layout = None
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        status = {
            "state": "idle",
            "current_task": None,
            "last_action": None,
            "learning_events": 0,
            "success_rate": 0.0,
            "avg_reward": 0.0
        }
        
        # Check for learning events
        if self.events_file.exists():
            try:
                with open(self.events_file) as f:
                    events = [json.loads(line) for line in f if line.strip()]
                status["learning_events"] = len(events)
                
                if events:
                    recent_events = events[-10:]
                    rewards = [e.get("reward", 0) for e in recent_events if "reward" in e]
                    if rewards:
                        positive_rewards = sum(1 for r in rewards if r > 0)
                        status["success_rate"] = positive_rewards / len(rewards)
                        status["avg_reward"] = sum(rewards) / len(rewards)
                    
                    last_event = events[-1]
                    status["last_action"] = last_event.get("tool", "unknown")
            except Exception:
                pass
        
        # Check for learning state
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                status["state"] = state.get("enabled", "idle")
            except Exception:
                pass
        
        return status
    
    def get_recent_actions(self) -> List[Dict[str, Any]]:
        """Get recent agent actions"""
        actions = []
        
        if self.events_file.exists():
            try:
                with open(self.events_file) as f:
                    events = [json.loads(line) for line in f if line.strip()]
                
                for event in events[-20:]:  # Last 20 events
                    action = {
                        "timestamp": event.get("timestamp", "Unknown"),
                        "tool": event.get("tool", "unknown"),
                        "action": event.get("action", "unknown"),
                        "reward": event.get("reward", 0.0),
                        "info": event.get("info", {})
                    }
                    actions.append(action)
            except Exception:
                pass
        
        return actions
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get learning progress information"""
        progress = {
            "total_events": 0,
            "recent_rewards": [],
            "tool_usage": {},
            "action_distribution": {},
            "learning_trend": "stable"
        }
        
        if self.events_file.exists():
            try:
                with open(self.events_file) as f:
                    events = [json.loads(line) for line in f if line.strip()]
                
                progress["total_events"] = len(events)
                
                if events:
                    # Recent rewards
                    recent_events = events[-20:]
                    rewards = [e.get("reward", 0) for e in recent_events if "reward" in e]
                    progress["recent_rewards"] = rewards
                    
                    # Tool usage
                    tool_counts = {}
                    for event in events:
                        tool = event.get("tool", "unknown")
                        tool_counts[tool] = tool_counts.get(tool, 0) + 1
                    progress["tool_usage"] = tool_counts
                    
                    # Action distribution
                    action_counts = {}
                    for event in events:
                        action = event.get("action", "unknown")
                        action_counts[action] = action_counts.get(action, 0) + 1
                    progress["action_distribution"] = action_counts
                    
                    # Learning trend
                    if len(rewards) >= 10:
                        recent_avg = sum(rewards[-10:]) / 10
                        older_avg = sum(rewards[-20:-10]) / 10 if len(rewards) >= 20 else recent_avg
                        if recent_avg > older_avg * 1.1:
                            progress["learning_trend"] = "improving"
                        elif recent_avg < older_avg * 0.9:
                            progress["learning_trend"] = "declining"
            except Exception:
                pass
        
        return progress
    
    def create_header_panel(self) -> Panel:
        """Create the header panel"""
        status = self.get_agent_status()
        
        header_text = f"🤖 DSPy Agent Monitor | Events: {status['learning_events']} | Success: {status['success_rate']:.1%} | Avg Reward: {status['avg_reward']:.3f}"
        
        return Panel(
            Align.center(Text(header_text, style="bold blue")),
            style="blue"
        )
    
    def create_actions_panel(self) -> Panel:
        """Create the actions panel"""
        actions = self.get_recent_actions()
        
        if not actions:
            content = Text("No recent actions", style="dim")
        else:
            content = Text()
            for action in actions[-10:]:  # Show last 10 actions
                timestamp = action["timestamp"][:19] if len(action["timestamp"]) > 19 else action["timestamp"]
                tool = action["tool"]
                reward = action["reward"]
                
                # Color code based on reward
                if reward > 0.7:
                    style = "green"
                elif reward > 0.3:
                    style = "yellow"
                else:
                    style = "red"
                
                content.append(f"{timestamp} {tool} ({reward:.2f})\n", style=style)
        
        return Panel(
            content,
            title="Recent Actions",
            border_style="green"
        )
    
    def create_thoughts_panel(self) -> Panel:
        """Create the thoughts panel"""
        # This would be populated from agent's internal thoughts
        # For now, we'll show action details as "thoughts"
        actions = self.get_recent_actions()
        
        if not actions:
            content = Text("No recent thoughts", style="dim")
        else:
            content = Text()
            for action in actions[-5:]:  # Show last 5 thoughts
                tool = action["tool"]
                info = action.get("info", {})
                
                # Extract "thoughts" from info
                thought = info.get("reasoning", info.get("plan", f"Executing {tool}"))
                content.append(f"💭 {thought[:50]}...\n", style="cyan")
        
        return Panel(
            content,
            title="Agent Thoughts",
            border_style="cyan"
        )
    
    def create_stats_panel(self) -> Panel:
        """Create the stats panel"""
        progress = self.get_learning_progress()
        
        content = Text()
        content.append(f"Total Events: {progress['total_events']}\n", style="bold")
        content.append(f"Learning Trend: {progress['learning_trend']}\n", style="bold")
        
        if progress["recent_rewards"]:
            avg_reward = sum(progress["recent_rewards"]) / len(progress["recent_rewards"])
            content.append(f"Recent Avg Reward: {avg_reward:.3f}\n", style="bold")
        
        content.append("\nTool Usage:\n", style="bold")
        for tool, count in sorted(progress["tool_usage"].items(), key=lambda x: x[1], reverse=True)[:5]:
            content.append(f"  {tool}: {count}\n", style="white")
        
        return Panel(
            content,
            title="Learning Stats",
            border_style="yellow"
        )
    
    def create_learning_panel(self) -> Panel:
        """Create the learning panel"""
        progress = self.get_learning_progress()
        
        content = Text()
        
        if progress["recent_rewards"]:
            content.append("Recent Rewards:\n", style="bold")
            for i, reward in enumerate(progress["recent_rewards"][-10:]):
                if reward > 0.7:
                    style = "green"
                elif reward > 0.3:
                    style = "yellow"
                else:
                    style = "red"
                content.append(f"  {reward:.2f} ", style=style)
                if (i + 1) % 5 == 0:
                    content.append("\n")
            content.append("\n")
        
        content.append("Action Distribution:\n", style="bold")
        for action, count in sorted(progress["action_distribution"].items(), key=lambda x: x[1], reverse=True)[:5]:
            content.append(f"  {action}: {count}\n", style="white")
        
        return Panel(
            content,
            title="Learning Progress",
            border_style="magenta"
        )
    
    def create_footer_panel(self) -> Panel:
        """Create the footer panel"""
        footer_text = "Press Ctrl+C to stop monitoring | Use 'stats' command in agent session for more details"
        
        return Panel(
            Align.center(Text(footer_text, style="dim")),
            style="dim"
        )
    
    def update_display(self):
        """Update the display with current information"""
        if not self.console or not self.layout:
            return
        
        # Update layout components
        self.layout["header"].update(self.create_header_panel())
        self.layout["actions"].update(self.create_actions_panel())
        self.layout["thoughts"].update(self.create_thoughts_panel())
        self.layout["stats"].update(self.create_stats_panel())
        self.layout["learning"].update(self.create_learning_panel())
        self.layout["footer"].update(self.create_footer_panel())
    
    def monitor_agent(self):
        """Monitor the agent in real-time"""
        if not self.console or not self.layout:
            print("❌ Rich library not available. Install with: pip install rich")
            return
        
        print("🚀 Starting real-time agent monitoring...")
        print("Press Ctrl+C to stop")
        
        with Live(self.layout, refresh_per_second=2, console=self.console) as live:
            try:
                while self.monitoring:
                    self.update_display()
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n🛑 Stopping monitor...")
                self.monitoring = False
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring = True
        self.monitor_agent()
    
    def show_current_status(self):
        """Show current agent status"""
        status = self.get_agent_status()
        progress = self.get_learning_progress()
        
        print("🤖 DSPy Agent Current Status")
        print("=" * 50)
        print(f"State: {status['state']}")
        print(f"Learning Events: {status['learning_events']}")
        print(f"Success Rate: {status['success_rate']:.1%}")
        print(f"Average Reward: {status['avg_reward']:.3f}")
        print(f"Learning Trend: {progress['learning_trend']}")
        
        if progress["tool_usage"]:
            print(f"\nTool Usage:")
            for tool, count in sorted(progress["tool_usage"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool}: {count} times")
        
        if progress["recent_rewards"]:
            print(f"\nRecent Rewards: {progress['recent_rewards'][-5:]}")
    
    def show_agent_thoughts(self, limit: int = 10):
        """Show recent agent thoughts"""
        actions = self.get_recent_actions()
        
        print(f"\n💭 Recent Agent Thoughts (Last {limit})")
        print("-" * 50)
        
        if not actions:
            print("No recent thoughts available")
            return
        
        for action in actions[-limit:]:
            timestamp = action["timestamp"]
            tool = action["tool"]
            info = action.get("info", {})
            
            # Extract thoughts from info
            thought = info.get("reasoning", info.get("plan", f"Executing {tool}"))
            
            print(f"🕒 {timestamp}")
            print(f"🔧 Tool: {tool}")
            print(f"💭 Thought: {thought}")
            print(f"🎯 Reward: {action['reward']:.3f}")
            print()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor DSPy Agent in Real-Time")
    parser.add_argument("--monitor", "-m", action="store_true", help="Start real-time monitoring")
    parser.add_argument("--status", "-s", action="store_true", help="Show current status")
    parser.add_argument("--thoughts", "-t", type=int, default=10, help="Show recent thoughts (default: 10)")
    
    args = parser.parse_args()
    
    monitor = AgentMonitor(project_root)
    
    if args.monitor:
        monitor.start_monitoring()
    elif args.status:
        monitor.show_current_status()
    else:
        monitor.show_current_status()
        monitor.show_agent_thoughts(args.thoughts)

if __name__ == "__main__":
    main()
