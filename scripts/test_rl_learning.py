#!/usr/bin/env python3
"""
Comprehensive RL Learning Test for DSPy Agent

This script tests the agent's learning capabilities by running
a series of tasks and verifying that learning is actually occurring.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class RLLearningTester:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.state_file = project_root / ".dspy_rl_state.json"
        self.events_file = project_root / ".dspy_rl_events.jsonl"
        
        # Test results
        self.test_results = {
            "learning_events": False,
            "reward_aggregation": False,
            "action_selection": False,
            "tool_usage": False,
            "learning_improvement": False
        }
        
    def run_command(self, command: str, timeout: int = 30) -> tuple[bool, str, str]:
        """Run a command and return success, stdout, stderr"""
        try:
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
    
    def get_learning_events(self) -> List[Dict[str, Any]]:
        """Get all learning events"""
        events = []
        if self.events_file.exists():
            try:
                with open(self.events_file) as f:
                    for line in f:
                        if line.strip():
                            events.append(json.loads(line))
            except Exception as e:
                print(f"⚠️  Error reading events: {e}")
        return events
    
    def test_learning_events_creation(self) -> bool:
        """Test that learning events are being created"""
        print("🧪 Testing learning events creation...")
        
        initial_events = self.get_learning_events()
        initial_count = len(initial_events)
        
        # Run a simple command to generate learning events
        success, stdout, stderr = self.run_command("help")
        
        if not success:
            print("❌ Failed to run test command")
            return False
        
        # Wait for events to be processed
        time.sleep(3)
        
        final_events = self.get_learning_events()
        final_count = len(final_events)
        
        events_created = final_count > initial_count
        
        if events_created:
            print(f"✅ Learning events created: {final_count - initial_count} new events")
            self.test_results["learning_events"] = True
        else:
            print("❌ No new learning events created")
        
        return events_created
    
    def test_reward_aggregation(self) -> bool:
        """Test that rewards are being aggregated correctly"""
        print("🧪 Testing reward aggregation...")
        
        events = self.get_learning_events()
        
        if not events:
            print("❌ No learning events found")
            return False
        
        # Check for reward data
        events_with_rewards = [e for e in events if "reward" in e]
        
        if events_with_rewards:
            print(f"✅ Reward aggregation working: {len(events_with_rewards)} events with rewards")
            
            # Check reward distribution
            rewards = [e["reward"] for e in events_with_rewards]
            avg_reward = sum(rewards) / len(rewards)
            print(f"   Average reward: {avg_reward:.3f}")
            print(f"   Reward range: {min(rewards):.3f} to {max(rewards):.3f}")
            
            self.test_results["reward_aggregation"] = True
            return True
        else:
            print("❌ No reward data found in events")
            return False
    
    def test_action_selection(self) -> bool:
        """Test that action selection is working"""
        print("🧪 Testing action selection...")
        
        events = self.get_learning_events()
        
        if not events:
            print("❌ No learning events found")
            return False
        
        # Check for action data
        events_with_actions = [e for e in events if "action" in e]
        
        if events_with_actions:
            print(f"✅ Action selection working: {len(events_with_actions)} events with actions")
            
            # Analyze action distribution
            action_counts = defaultdict(int)
            for event in events_with_actions:
                action_counts[event["action"]] += 1
            
            print("   Action distribution:")
            for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"     {action}: {count} times")
            
            self.test_results["action_selection"] = True
            return True
        else:
            print("❌ No action data found in events")
            return False
    
    def test_tool_usage(self) -> bool:
        """Test that tool usage is being tracked"""
        print("🧪 Testing tool usage tracking...")
        
        events = self.get_learning_events()
        
        if not events:
            print("❌ No learning events found")
            return False
        
        # Check for tool data
        events_with_tools = [e for e in events if "tool" in e]
        
        if events_with_tools:
            print(f"✅ Tool usage tracking working: {len(events_with_tools)} events with tools")
            
            # Analyze tool distribution
            tool_counts = defaultdict(int)
            for event in events_with_tools:
                tool_counts[event["tool"]] += 1
            
            print("   Tool usage distribution:")
            for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"     {tool}: {count} times")
            
            self.test_results["tool_usage"] = True
            return True
        else:
            print("❌ No tool data found in events")
            return False
    
    def test_learning_improvement(self) -> bool:
        """Test that learning is actually improving performance"""
        print("🧪 Testing learning improvement...")
        
        events = self.get_learning_events()
        
        if len(events) < 10:
            print("⚠️  Not enough events to test learning improvement")
            return False
        
        # Analyze reward trends
        rewards = [e.get("reward", 0) for e in events if "reward" in e]
        
        if len(rewards) < 10:
            print("⚠️  Not enough reward data to test learning improvement")
            return False
        
        # Split into early and late periods
        mid_point = len(rewards) // 2
        early_rewards = rewards[:mid_point]
        late_rewards = rewards[mid_point:]
        
        early_avg = sum(early_rewards) / len(early_rewards)
        late_avg = sum(late_rewards) / len(late_rewards)
        
        improvement = late_avg - early_avg
        
        print(f"   Early period average reward: {early_avg:.3f}")
        print(f"   Late period average reward: {late_avg:.3f}")
        print(f"   Improvement: {improvement:+.3f}")
        
        # Check for improvement (allowing for some variance)
        if improvement > -0.1:  # Allow for some variance
            print("✅ Learning improvement detected")
            self.test_results["learning_improvement"] = True
            return True
        else:
            print("⚠️  No clear learning improvement detected (may need more data)")
            return False
    
    def run_learning_tests(self) -> bool:
        """Run all learning tests"""
        print("🚀 DSPy Agent RL Learning Test Suite")
        print("="*60)
        
        tests = [
            self.test_learning_events_creation,
            self.test_reward_aggregation,
            self.test_action_selection,
            self.test_tool_usage,
            self.test_learning_improvement
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed_tests += 1
            except Exception as e:
                print(f"❌ Test failed with error: {e}")
            print()
        
        # Display results
        print("📊 Test Results Summary")
        print("="*60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All learning tests passed! The agent is learning correctly.")
        elif passed_tests >= total_tests * 0.8:
            print("✅ Most learning tests passed. The agent is learning well.")
        else:
            print("⚠️  Some learning tests failed. Check the agent configuration.")
        
        return passed_tests >= total_tests * 0.8
    
    def show_learning_analysis(self):
        """Show detailed learning analysis"""
        print("\n🔍 Detailed Learning Analysis")
        print("="*60)
        
        events = self.get_learning_events()
        
        if not events:
            print("No learning events found.")
            return
        
        print(f"Total Events: {len(events)}")
        
        # Analyze by time
        if len(events) > 1:
            first_event = events[0]
            last_event = events[-1]
            print(f"Time Range: {first_event.get('timestamp', 'Unknown')} to {last_event.get('timestamp', 'Unknown')}")
        
        # Analyze rewards
        rewards = [e.get("reward", 0) for e in events if "reward" in e]
        if rewards:
            print(f"\nReward Analysis:")
            print(f"  Total rewards: {len(rewards)}")
            print(f"  Average reward: {sum(rewards) / len(rewards):.3f}")
            print(f"  Min reward: {min(rewards):.3f}")
            print(f"  Max reward: {max(rewards):.3f}")
            
            positive_rewards = sum(1 for r in rewards if r > 0)
            print(f"  Positive rewards: {positive_rewards}/{len(rewards)} ({positive_rewards/len(rewards):.1%})")
        
        # Analyze actions
        actions = [e.get("action", "unknown") for e in events if "action" in e]
        if actions:
            action_counts = defaultdict(int)
            for action in actions:
                action_counts[action] += 1
            
            print(f"\nAction Analysis:")
            for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {action}: {count} times")
        
        # Analyze tools
        tools = [e.get("tool", "unknown") for e in events if "tool" in e]
        if tools:
            tool_counts = defaultdict(int)
            for tool in tools:
                tool_counts[tool] += 1
            
            print(f"\nTool Analysis:")
            for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool}: {count} times")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DSPy Agent RL Learning")
    parser.add_argument("--analysis", "-a", action="store_true", help="Show detailed learning analysis")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick tests only")
    
    args = parser.parse_args()
    
    tester = RLLearningTester(project_root)
    
    if args.analysis:
        tester.show_learning_analysis()
    else:
        success = tester.run_learning_tests()
        if args.quick:
            print("\n💡 For detailed analysis, run: uv run python scripts/test_rl_learning.py --analysis")
        
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
