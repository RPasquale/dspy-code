#!/usr/bin/env python3
"""
Integrate Agent Actions and Thoughts into Kafka Streaming

This script modifies the agent to publish its actions, thoughts, and learning
progress to the existing Kafka streaming infrastructure for real-time monitoring.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import streaming components
from dspy_agent.streaming import get_event_bus

class AgentStreamingPublisher:
    """Publishes agent actions and thoughts to Kafka streams"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.bus = get_event_bus()
        self.action_count = 0
        self.thought_count = 0
        
        # Topics for different types of data
        self.topics = {
            "actions": "agent.actions",
            "thoughts": "agent.thoughts", 
            "learning": "agent.learning",
            "metrics": "agent.metrics",
            "frontend": "agent.monitor.frontend"
        }
    
    def publish_action(self, tool: str, action: str, reward: float, context: Dict[str, Any], reasoning: str = ""):
        """Publish an agent action to Kafka"""
        self.action_count += 1
        
        action_data = {
            "type": "agent_action",
            "timestamp": time.time(),
            "action_id": self.action_count,
            "tool": tool,
            "action": action,
            "reward": reward,
            "context": context,
            "reasoning": reasoning,
            "metrics": {
                "action_count": self.action_count,
                "timestamp": time.time()
            }
        }
        
        # Publish to actions topic
        self._publish_to_kafka(self.topics["actions"], action_data)
        
        # Also publish to frontend for real-time display
        self._publish_to_kafka(self.topics["frontend"], {
            "type": "action_update",
            "data": action_data
        })
        
        print(f"📡 Published action: {tool} -> {action} (reward: {reward:.3f})")
    
    def publish_thought(self, thought: str, context: Dict[str, Any], confidence: float = 0.0):
        """Publish an agent thought to Kafka"""
        self.thought_count += 1
        
        thought_data = {
            "type": "agent_thought",
            "timestamp": time.time(),
            "thought_id": self.thought_count,
            "thought": thought,
            "context": context,
            "confidence": confidence,
            "metrics": {
                "thought_count": self.thought_count,
                "timestamp": time.time()
            }
        }
        
        # Publish to thoughts topic
        self._publish_to_kafka(self.topics["thoughts"], thought_data)
        
        # Also publish to frontend for real-time display
        self._publish_to_kafka(self.topics["frontend"], {
            "type": "thought_update",
            "data": thought_data
        })
        
        print(f"💭 Published thought: {thought[:50]}...")
    
    def publish_learning_progress(self, learning_data: Dict[str, Any]):
        """Publish learning progress to Kafka"""
        learning_payload = {
            "type": "learning_progress",
            "timestamp": time.time(),
            "data": learning_data,
            "metrics": {
                "total_actions": self.action_count,
                "total_thoughts": self.thought_count,
                "timestamp": time.time()
            }
        }
        
        # Publish to learning topic
        self._publish_to_kafka(self.topics["learning"], learning_payload)
        
        # Also publish to frontend for real-time display
        self._publish_to_kafka(self.topics["frontend"], {
            "type": "learning_update",
            "data": learning_payload
        })
        
        print(f"📊 Published learning progress: {learning_data.get('total_events', 0)} events")
    
    def publish_metrics(self, metrics: Dict[str, Any]):
        """Publish agent metrics to Kafka"""
        metrics_payload = {
            "type": "agent_metrics",
            "timestamp": time.time(),
            "metrics": metrics,
            "action_count": self.action_count,
            "thought_count": self.thought_count
        }
        
        # Publish to metrics topic
        self._publish_to_kafka(self.topics["metrics"], metrics_payload)
        
        # Also publish to frontend for real-time display
        self._publish_to_kafka(self.topics["frontend"], {
            "type": "metrics_update",
            "data": metrics_payload
        })
        
        print(f"📈 Published metrics: {len(metrics)} metrics")
    
    def _publish_to_kafka(self, topic: str, data: Dict[str, Any]):
        """Publish data to Kafka topic"""
        try:
            self.bus.publish(topic, data)
        except Exception as e:
            print(f"⚠️  Failed to publish event: {e}")
    
    def _log_to_file(self, topic: str, data: Dict[str, Any]):
        """Fallback: log to file if Kafka not available"""
        log_file = self.project_root / "logs" / f"{topic.replace('.', '_')}.jsonl"
        log_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            print(f"⚠️  Failed to log to file {log_file}: {e}")

class AgentStreamingIntegration:
    """Integrates agent actions with streaming infrastructure"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.publisher = AgentStreamingPublisher(project_root)
        self.learning_events_file = project_root / ".dspy_rl_events.jsonl"
        self.learning_state_file = project_root / ".dspy_rl_state.json"
        
        # Track learning progress
        self.last_event_count = 0
        self.last_metrics_update = 0
    
    def monitor_learning_events(self):
        """Monitor learning events and publish updates"""
        if not self.learning_events_file.exists():
            return
        
        try:
            with open(self.learning_events_file) as f:
                events = [json.loads(line) for line in f if line.strip()]
            
            current_count = len(events)
            
            # If new events, publish learning progress
            if current_count > self.last_event_count:
                new_events = events[self.last_event_count:]
                self.last_event_count = current_count
                
                # Publish learning progress
                learning_data = self._analyze_learning_progress(events)
                self.publisher.publish_learning_progress(learning_data)
                
                # Publish individual new events as actions
                for event in new_events:
                    self.publisher.publish_action(
                        tool=event.get("tool", "unknown"),
                        action=event.get("action", "unknown"),
                        reward=event.get("reward", 0.0),
                        context=event.get("info", {}),
                        reasoning=event.get("reasoning", f"Executing {event.get('tool', 'unknown')}")
                    )
        
        except Exception as e:
            print(f"⚠️  Error monitoring learning events: {e}")
    
    def _analyze_learning_progress(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze learning progress from events"""
        if not events:
            return {"total_events": 0, "success_rate": 0.0, "avg_reward": 0.0}
        
        # Calculate metrics
        total_events = len(events)
        rewards = [e.get("reward", 0) for e in events if "reward" in e]
        
        success_rate = 0.0
        avg_reward = 0.0
        if rewards:
            positive_rewards = sum(1 for r in rewards if r > 0)
            success_rate = positive_rewards / len(rewards)
            avg_reward = sum(rewards) / len(rewards)
        
        # Tool usage
        tool_usage = {}
        for event in events:
            tool = event.get("tool", "unknown")
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        # Action distribution
        action_distribution = {}
        for event in events:
            action = event.get("action", "unknown")
            action_distribution[action] = action_distribution.get(action, 0) + 1
        
        # Learning trend
        learning_trend = "stable"
        if len(rewards) >= 10:
            recent_avg = sum(rewards[-10:]) / 10
            older_avg = sum(rewards[-20:-10]) / 10 if len(rewards) >= 20 else recent_avg
            if recent_avg > older_avg * 1.1:
                learning_trend = "improving"
            elif recent_avg < older_avg * 0.9:
                learning_trend = "declining"
        
        return {
            "total_events": total_events,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "learning_trend": learning_trend,
            "tool_usage": tool_usage,
            "action_distribution": action_distribution,
            "recent_rewards": rewards[-20:] if rewards else []
        }
    
    def publish_periodic_metrics(self):
        """Publish periodic metrics updates"""
        current_time = time.time()
        
        # Publish metrics every 30 seconds
        if current_time - self.last_metrics_update < 30:
            return
        
        self.last_metrics_update = current_time
        
        # Get current learning state
        learning_state = {}
        if self.learning_state_file.exists():
            try:
                with open(self.learning_state_file) as f:
                    learning_state = json.load(f)
            except Exception:
                pass
        
        # Get learning progress
        learning_progress = {}
        if self.learning_events_file.exists():
            try:
                with open(self.learning_events_file) as f:
                    events = [json.loads(line) for line in f if line.strip()]
                learning_progress = self._analyze_learning_progress(events)
            except Exception:
                pass
        
        # Combine metrics
        metrics = {
            "learning_state": learning_state,
            "learning_progress": learning_progress,
            "publisher_stats": {
                "total_actions": self.publisher.action_count,
                "total_thoughts": self.publisher.thought_count
            },
            "timestamp": current_time
        }
        
        self.publisher.publish_metrics(metrics)
    
    def start_monitoring(self):
        """Start monitoring and publishing agent data"""
        print("🚀 Starting agent streaming integration...")
        
        try:
            while True:
                # Monitor learning events
                self.monitor_learning_events()
                
                # Publish periodic metrics
                self.publish_periodic_metrics()
                
                # Small delay
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping agent streaming integration...")
        except Exception as e:
            print(f"❌ Error in agent streaming integration: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate Agent with Kafka Streaming")
    parser.add_argument("--monitor", "-m", action="store_true", help="Start monitoring mode")
    parser.add_argument("--test", "-t", action="store_true", help="Test publishing")
    
    args = parser.parse_args()
    
    integration = AgentStreamingIntegration(project_root)
    
    if args.test:
        # Test publishing
        print("🧪 Testing agent streaming integration...")
        
        # Test action publishing
        integration.publisher.publish_action(
            tool="grep",
            action="search_code",
            reward=0.8,
            context={"pattern": "def test_", "files": 5},
            reasoning="Searching for test functions to understand codebase structure"
        )
        
        # Test thought publishing
        integration.publisher.publish_thought(
            thought="I need to understand the codebase structure before making changes",
            context={"task": "code_analysis"},
            confidence=0.9
        )
        
        # Test learning progress
        integration.publisher.publish_learning_progress({
            "total_events": 10,
            "success_rate": 0.8,
            "avg_reward": 0.6,
            "learning_trend": "improving"
        })
        
        print("✅ Test publishing completed")
    
    elif args.monitor:
        # Start monitoring
        integration.start_monitoring()
    else:
        # Default: show help
        parser.print_help()

if __name__ == "__main__":
    main()
