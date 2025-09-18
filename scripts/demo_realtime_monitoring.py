#!/usr/bin/env python3
"""
Demo Real-Time Agent Monitoring

This script demonstrates the complete real-time monitoring system
by simulating agent actions and showing how they appear in the monitoring interface.
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.integrate_agent_streaming import AgentStreamingPublisher

class AgentSimulator:
    """Simulates agent actions for demonstration"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.publisher = AgentStreamingPublisher(project_root)
        self.running = False
        
        # Demo scenarios
        self.scenarios = [
            {
                "name": "Code Exploration",
                "actions": [
                    ("grep", "search_code", 0.8, "Searching for test functions to understand codebase structure"),
                    ("esearch", "semantic_search", 0.9, "Finding related code patterns and dependencies"),
                    ("ctx", "extract_context", 0.7, "Extracting key context from the codebase")
                ]
            },
            {
                "name": "Bug Fixing",
                "actions": [
                    ("grep", "search_error", 0.6, "Looking for error patterns in the code"),
                    ("edit", "fix_bug", 0.8, "Applying a targeted fix for the identified bug"),
                    ("grep", "verify_fix", 0.9, "Verifying that the fix resolves the issue")
                ]
            },
            {
                "name": "Feature Development",
                "actions": [
                    ("plan", "design_feature", 0.7, "Planning the implementation approach for the new feature"),
                    ("edit", "implement_feature", 0.8, "Implementing the core functionality"),
                    ("edit", "add_tests", 0.9, "Adding comprehensive tests for the new feature")
                ]
            }
        ]
    
    def simulate_scenario(self, scenario: Dict[str, Any]):
        """Simulate a complete scenario"""
        print(f"\n🎯 Simulating: {scenario['name']}")
        print("-" * 50)
        
        for i, (tool, action, reward, reasoning) in enumerate(scenario['actions'], 1):
            print(f"Step {i}: {tool} -> {action}")
            
            # Publish action
            self.publisher.publish_action(
                tool=tool,
                action=action,
                reward=reward,
                context={
                    "scenario": scenario['name'],
                    "step": i,
                    "total_steps": len(scenario['actions'])
                },
                reasoning=reasoning
            )
            
            # Publish thought
            thought = f"Step {i} of {scenario['name']}: {reasoning}"
            self.publisher.publish_thought(
                thought=thought,
                context={"scenario": scenario['name'], "step": i},
                confidence=reward
            )
            
            # Simulate processing time
            time.sleep(2)
        
        # Publish learning progress after scenario
        learning_data = {
            "scenario_completed": scenario['name'],
            "total_actions": sum(len(s['actions']) for s in self.scenarios[:self.scenarios.index(scenario)+1]),
            "success_rate": 0.8,
            "avg_reward": 0.8,
            "learning_trend": "improving"
        }
        
        self.publisher.publish_learning_progress(learning_data)
        print(f"✅ Completed: {scenario['name']}")
    
    def run_demo(self):
        """Run the complete demonstration"""
        print("🚀 DSPy Agent Real-Time Monitoring Demo")
        print("=" * 60)
        print("This demo simulates agent actions and shows how they appear")
        print("in the real-time monitoring system.")
        print()
        print("💡 In a real scenario, you would:")
        print("   1. Start the monitoring system")
        print("   2. Use the agent normally")
        print("   3. Watch actions appear in real-time")
        print()
        
        input("Press Enter to start the demo...")
        
        # Run each scenario
        for scenario in self.scenarios:
            self.simulate_scenario(scenario)
            
            if scenario != self.scenarios[-1]:
                input("\nPress Enter to continue to next scenario...")
        
        # Final summary
        print("\n🎉 Demo Complete!")
        print("=" * 60)
        print("The agent has performed various actions and published them to Kafka.")
        print("You can now see this data in the real-time monitoring interface.")
        print()
        print("🔍 To see the monitoring in action:")
        print("   1. Start the monitoring system:")
        print("      uv run python scripts/realtime_agent_monitor.py --cli")
        print()
        print("   2. Or check the current status:")
        print("      uv run python scripts/realtime_agent_monitor.py --status")
        print()
        print("   3. View the data in your frontend (if available)")
        print("      open http://localhost:8080")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo Real-Time Agent Monitoring")
    parser.add_argument("--scenario", "-s", type=int, help="Run specific scenario (1-3)")
    parser.add_argument("--loop", "-l", action="store_true", help="Run scenarios in a loop")
    
    args = parser.parse_args()
    
    simulator = AgentSimulator(project_root)
    
    if args.scenario:
        # Run specific scenario
        if 1 <= args.scenario <= len(simulator.scenarios):
            scenario = simulator.scenarios[args.scenario - 1]
            simulator.simulate_scenario(scenario)
        else:
            print(f"❌ Invalid scenario number. Choose 1-{len(simulator.scenarios)}")
    elif args.loop:
        # Run scenarios in a loop
        print("🔄 Running scenarios in a loop...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                for scenario in simulator.scenarios:
                    simulator.simulate_scenario(scenario)
                    time.sleep(5)  # Pause between scenarios
        except KeyboardInterrupt:
            print("\n🛑 Stopping demo loop...")
    else:
        # Run complete demo
        simulator.run_demo()

if __name__ == "__main__":
    main()
