#!/usr/bin/env python3
"""
Real-Time Agent Monitor with Kafka/Spark Integration

This script provides real-time monitoring of the DSPy Agent's actions,
thoughts, and learning progress using the existing Kafka/Spark streaming
infrastructure to provide live data to both frontend and CLI.
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import streaming components
from dspy_agent.streaming import LocalBus, StreamConfig, start_local_stack
from dspy_agent.streaming.vectorized_pipeline import RLVectorizer, VectorizedStreamOrchestrator

@dataclass
class AgentAction:
    """Represents an agent action with full context"""
    timestamp: float
    tool: str
    action: str
    reward: float
    reasoning: str
    context: Dict[str, Any]
    metrics: Dict[str, float]
    thoughts: List[str]

@dataclass
class LearningMetrics:
    """Learning progress metrics"""
    total_events: int
    success_rate: float
    avg_reward: float
    learning_trend: str
    tool_usage: Dict[str, int]
    action_distribution: Dict[str, int]
    recent_rewards: List[float]

class RealtimeAgentMonitor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.bus: Optional[LocalBus] = None
        self.streaming_threads: List[threading.Thread] = []
        
        # Monitoring data
        self.recent_actions = deque(maxlen=100)
        self.learning_metrics = LearningMetrics(
            total_events=0,
            success_rate=0.0,
            avg_reward=0.0,
            learning_trend="stable",
            tool_usage={},
            action_distribution={},
            recent_rewards=[]
        )
        
        # Real-time monitoring
        self.monitoring = False
        self.monitor_thread = None
        
        # Subscribers for real-time updates
        self.subscribers: List[Callable[[AgentAction], None]] = []
        
        # Kafka topics for monitoring
        self.monitor_topics = [
            "agent.results",
            "agent.metrics", 
            "agent.rl.vectorized",
            "agent.tasks",
            "agent.patches",
            "agent.errors"
        ]
    
    def setup_streaming_infrastructure(self):
        """Set up the Kafka/Spark streaming infrastructure"""
        print("🚀 Setting up Kafka/Spark streaming infrastructure...")
        
        try:
            # Start the local streaming stack
            self.streaming_threads, self.bus = start_local_stack(
                root=self.project_root,
                cfg=None,  # Use default config
                storage=None,
                kafka=None
            )
            
            print(f"✅ Streaming infrastructure started with {len(self.streaming_threads)} threads")
            
            # Subscribe to monitoring topics
            for topic in self.monitor_topics:
                try:
                    self.bus.subscribe(topic)
                    print(f"📡 Subscribed to topic: {topic}")
                except Exception as e:
                    print(f"⚠️  Could not subscribe to {topic}: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup streaming infrastructure: {e}")
            return False
    
    def process_agent_action(self, topic: str, payload: Dict[str, Any]):
        """Process an agent action from Kafka stream"""
        try:
            # Extract action information
            timestamp = payload.get("timestamp", time.time())
            tool = payload.get("tool", "unknown")
            action = payload.get("action", "unknown")
            reward = payload.get("reward", 0.0)
            reasoning = payload.get("reasoning", payload.get("plan", f"Executing {tool}"))
            context = payload.get("context", {})
            metrics = payload.get("metrics", {})
            thoughts = payload.get("thoughts", [])
            
            # Create agent action
            agent_action = AgentAction(
                timestamp=timestamp,
                tool=tool,
                action=action,
                reward=reward,
                reasoning=reasoning,
                context=context,
                metrics=metrics,
                thoughts=thoughts
            )
            
            # Add to recent actions
            self.recent_actions.append(agent_action)
            
            # Update learning metrics
            self.update_learning_metrics(agent_action)
            
            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber(agent_action)
                except Exception as e:
                    print(f"⚠️  Subscriber error: {e}")
            
            # Publish to frontend via Kafka
            self.publish_to_frontend(agent_action)
            
        except Exception as e:
            print(f"❌ Error processing agent action: {e}")
    
    def update_learning_metrics(self, action: AgentAction):
        """Update learning metrics based on new action"""
        # Update basic metrics
        self.learning_metrics.total_events += 1
        self.learning_metrics.recent_rewards.append(action.reward)
        
        # Keep only recent rewards (last 50)
        if len(self.learning_metrics.recent_rewards) > 50:
            self.learning_metrics.recent_rewards = self.learning_metrics.recent_rewards[-50:]
        
        # Update tool usage
        self.learning_metrics.tool_usage[action.tool] = self.learning_metrics.tool_usage.get(action.tool, 0) + 1
        
        # Update action distribution
        self.learning_metrics.action_distribution[action.action] = self.learning_metrics.action_distribution.get(action.action, 0) + 1
        
        # Calculate success rate
        if self.learning_metrics.recent_rewards:
            positive_rewards = sum(1 for r in self.learning_metrics.recent_rewards if r > 0)
            self.learning_metrics.success_rate = positive_rewards / len(self.learning_metrics.recent_rewards)
            self.learning_metrics.avg_reward = sum(self.learning_metrics.recent_rewards) / len(self.learning_metrics.recent_rewards)
        
        # Determine learning trend
        if len(self.learning_metrics.recent_rewards) >= 10:
            recent_avg = sum(self.learning_metrics.recent_rewards[-10:]) / 10
            older_avg = sum(self.learning_metrics.recent_rewards[-20:-10]) / 10 if len(self.learning_metrics.recent_rewards) >= 20 else recent_avg
            if recent_avg > older_avg * 1.1:
                self.learning_metrics.learning_trend = "improving"
            elif recent_avg < older_avg * 0.9:
                self.learning_metrics.learning_trend = "declining"
            else:
                self.learning_metrics.learning_trend = "stable"
    
    def publish_to_frontend(self, action: AgentAction):
        """Publish agent action to frontend via Kafka"""
        if not self.bus:
            return
        
        try:
            # Create frontend payload
            frontend_payload = {
                "type": "agent_action",
                "timestamp": action.timestamp,
                "data": asdict(action),
                "metrics": asdict(self.learning_metrics)
            }
            
            # Publish to frontend topic
            self.bus.publish("agent.monitor.frontend", frontend_payload)
            
        except Exception as e:
            print(f"⚠️  Error publishing to frontend: {e}")
    
    def monitor_kafka_streams(self):
        """Monitor Kafka streams for agent actions"""
        if not self.bus:
            print("❌ No bus available for monitoring")
            return
        
        print("📡 Starting Kafka stream monitoring...")
        
        while self.monitoring:
            try:
                # Check each monitoring topic
                for topic in self.monitor_topics:
                    try:
                        # Get latest message from topic
                        message = self.bus.get_latest(topic, timeout=0.1)
                        if message:
                            self.process_agent_action(topic, message)
                    except Exception as e:
                        # Topic might not exist yet, that's ok
                        pass
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                print(f"❌ Error in stream monitoring: {e}")
                time.sleep(1)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        print("🚀 Starting real-time agent monitoring...")
        
        # Set up streaming infrastructure
        if not self.setup_streaming_infrastructure():
            print("❌ Failed to setup streaming infrastructure")
            return False
        
        # Start monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_kafka_streams)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print("✅ Real-time monitoring started")
        return True
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        print("🛑 Stopping real-time monitoring...")
        
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Stop streaming threads
        for thread in self.streaming_threads:
            if hasattr(thread, 'stop'):
                thread.stop()
            thread.join(timeout=5)
        
        print("✅ Real-time monitoring stopped")
    
    def add_subscriber(self, callback: Callable[[AgentAction], None]):
        """Add a subscriber for real-time updates"""
        self.subscribers.append(callback)
    
    def remove_subscriber(self, callback: Callable[[AgentAction], None]):
        """Remove a subscriber"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "monitoring": self.monitoring,
            "total_actions": len(self.recent_actions),
            "learning_metrics": asdict(self.learning_metrics),
            "recent_actions": [asdict(action) for action in list(self.recent_actions)[-10:]]
        }
    
    def get_learning_metrics(self) -> LearningMetrics:
        """Get current learning metrics"""
        return self.learning_metrics
    
    def get_recent_actions(self, limit: int = 20) -> List[AgentAction]:
        """Get recent agent actions"""
        return list(self.recent_actions)[-limit:]

class CLIMonitor:
    """CLI-based monitor that displays agent actions in real-time"""
    
    def __init__(self, monitor: RealtimeAgentMonitor):
        self.monitor = monitor
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
            from rich.align import Align
            
            self.console = Console()
            self.layout = Layout()
            
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
    
    def create_header_panel(self):
        """Create the header panel"""
        metrics = self.monitor.get_learning_metrics()
        
        header_text = f"🤖 DSPy Agent Real-Time Monitor | Events: {metrics.total_events} | Success: {metrics.success_rate:.1%} | Trend: {metrics.learning_trend}"
        
        if self.console and self.layout:
            from rich.panel import Panel
            from rich.align import Align
            from rich.text import Text
            return Panel(
                Align.center(Text(header_text, style="bold blue")),
                style="blue"
            )
        return header_text
    
    def create_actions_panel(self):
        """Create the actions panel"""
        actions = self.monitor.get_recent_actions(10)
        
        if not actions:
            content = Text("No recent actions", style="dim")
        else:
            content = Text()
            for action in actions:
                timestamp = datetime.fromtimestamp(action.timestamp).strftime("%H:%M:%S")
                tool = action.tool
                reward = action.reward
                
                # Color code based on reward
                if reward > 0.7:
                    style = "green"
                elif reward > 0.3:
                    style = "yellow"
                else:
                    style = "red"
                
                content.append(f"{timestamp} {tool} ({reward:.2f})\n", style=style)
        
        if self.console and self.layout:
            from rich.panel import Panel
            return Panel(
                content,
                title="Recent Actions",
                border_style="green"
            )
        return str(content)
    
    def create_thoughts_panel(self) -> Panel:
        """Create the thoughts panel"""
        actions = self.monitor.get_recent_actions(5)
        
        if not actions:
            content = Text("No recent thoughts", style="dim")
        else:
            content = Text()
            for action in actions:
                reasoning = action.reasoning[:60] + "..." if len(action.reasoning) > 60 else action.reasoning
                content.append(f"💭 {reasoning}\n", style="cyan")
        
        return Panel(
            content,
            title="Agent Thoughts",
            border_style="cyan"
        )
    
    def create_stats_panel(self) -> Panel:
        """Create the stats panel"""
        metrics = self.monitor.get_learning_metrics()
        
        content = Text()
        content.append(f"Total Events: {metrics.total_events}\n", style="bold")
        content.append(f"Success Rate: {metrics.success_rate:.1%}\n", style="bold")
        content.append(f"Avg Reward: {metrics.avg_reward:.3f}\n", style="bold")
        content.append(f"Learning Trend: {metrics.learning_trend}\n", style="bold")
        
        content.append("\nTool Usage:\n", style="bold")
        for tool, count in sorted(metrics.tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]:
            content.append(f"  {tool}: {count}\n", style="white")
        
        return Panel(
            content,
            title="Learning Stats",
            border_style="yellow"
        )
    
    def create_learning_panel(self) -> Panel:
        """Create the learning panel"""
        metrics = self.monitor.get_learning_metrics()
        
        content = Text()
        
        if metrics.recent_rewards:
            content.append("Recent Rewards:\n", style="bold")
            for i, reward in enumerate(metrics.recent_rewards[-10:]):
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
        for action, count in sorted(metrics.action_distribution.items(), key=lambda x: x[1], reverse=True)[:5]:
            content.append(f"  {action}: {count}\n", style="white")
        
        return Panel(
            content,
            title="Learning Progress",
            border_style="magenta"
        )
    
    def create_footer_panel(self) -> Panel:
        """Create the footer panel"""
        footer_text = "Real-time monitoring via Kafka/Spark | Press Ctrl+C to stop"
        
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
    
    def start_cli_monitoring(self):
        """Start CLI monitoring"""
        if not self.console or not self.layout:
            print("❌ Rich library not available. Install with: pip install rich")
            return
        
        print("🚀 Starting CLI monitoring...")
        print("Press Ctrl+C to stop")
        
        with Live(self.layout, refresh_per_second=2, console=self.console) as live:
            try:
                while self.monitor.monitoring:
                    self.update_display()
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n🛑 Stopping CLI monitor...")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-Time Agent Monitor with Kafka/Spark")
    parser.add_argument("--cli", "-c", action="store_true", help="Start CLI monitoring")
    parser.add_argument("--status", "-s", action="store_true", help="Show current status")
    parser.add_argument("--frontend", "-f", action="store_true", help="Enable frontend publishing")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = RealtimeAgentMonitor(project_root)
    
    try:
        # Start monitoring
        if not monitor.start_monitoring():
            print("❌ Failed to start monitoring")
            return
        
        if args.cli:
            # Start CLI monitoring
            cli_monitor = CLIMonitor(monitor)
            cli_monitor.start_cli_monitoring()
        elif args.status:
            # Show current status
            status = monitor.get_current_status()
            print("🤖 DSPy Agent Status")
            print("=" * 50)
            print(f"Monitoring: {status['monitoring']}")
            print(f"Total Actions: {status['total_actions']}")
            print(f"Learning Metrics: {json.dumps(status['learning_metrics'], indent=2)}")
        else:
            # Default: show status and wait
            print("🤖 DSPy Agent Real-Time Monitor")
            print("=" * 50)
            print("Monitoring agent actions via Kafka/Spark streaming...")
            print("Press Ctrl+C to stop")
            
            try:
                while monitor.monitoring:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping monitor...")
    
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
