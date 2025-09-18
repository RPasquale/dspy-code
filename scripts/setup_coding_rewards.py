#!/usr/bin/env python3
"""
Coding Reward Shaping System

This script sets up reward shaping to train the agent to be a productive coding partner.
It defines rewards for good coding practices and penalizes poor practices.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class CodingRewardShaper:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reward_config_file = project_root / ".dspy_coding_rewards.json"
        
        # Define reward structure for coding behaviors
        self.coding_rewards = {
            "code_quality": {
                "description": "Rewards for writing high-quality code",
                "weights": {
                    "clean_code": 0.3,      # Clean, readable code
                    "error_handling": 0.25,  # Proper error handling
                    "documentation": 0.2,    # Good documentation
                    "testing": 0.15,         # Writing tests
                    "best_practices": 0.1    # Following best practices
                },
                "positive_signals": [
                    "def ", "class ",  # Function/class definitions
                    "try:", "except",  # Error handling
                    '"""', "'''", "# ",  # Documentation
                    "test_", "pytest",  # Testing
                    "import", "from"    # Proper imports
                ],
                "negative_signals": [
                    "TODO", "FIXME", "HACK",  # Technical debt
                    "pass",  # Empty implementations
                    "print(",  # Debug prints
                    "eval(", "exec("  # Dangerous code
                ]
            },
            
            "functionality": {
                "description": "Rewards for implementing working features",
                "weights": {
                    "feature_complete": 0.4,  # Complete feature implementation
                    "bug_free": 0.3,          # No obvious bugs
                    "performance": 0.2,       # Good performance
                    "integration": 0.1        # Proper integration
                },
                "positive_signals": [
                    "return ",  # Functions that return values
                    "yield ",   # Generator functions
                    "async def",  # Async functions
                    "with ",    # Context managers
                    "assert "   # Assertions
                ],
                "negative_signals": [
                    "raise NotImplementedError",
                    "pass",  # Empty implementations
                    "None",  # Returning None
                    "..."    # Ellipsis (incomplete)
                ]
            },
            
            "problem_solving": {
                "description": "Rewards for good problem-solving approach",
                "weights": {
                    "understanding": 0.3,     # Understanding the problem
                    "planning": 0.25,         # Good planning
                    "implementation": 0.25,   # Solid implementation
                    "verification": 0.2       # Testing and verification
                },
                "positive_signals": [
                    "plan",  # Planning
                    "analyze",  # Analysis
                    "design",  # Design
                    "implement",  # Implementation
                    "test", "verify"  # Testing
                ],
                "negative_signals": [
                    "guess",  # Guessing
                    "maybe",  # Uncertainty
                    "probably",  # Uncertainty
                    "not sure"  # Uncertainty
                ]
            },
            
            "collaboration": {
                "description": "Rewards for being a good coding partner",
                "weights": {
                    "communication": 0.3,     # Clear communication
                    "explanation": 0.25,      # Explaining decisions
                    "suggestions": 0.25,      # Helpful suggestions
                    "learning": 0.2           # Learning from feedback
                },
                "positive_signals": [
                    "explain",  # Explaining
                    "suggest",  # Suggesting
                    "recommend",  # Recommending
                    "learn",  # Learning
                    "improve"  # Improving
                ],
                "negative_signals": [
                    "don't know",  # Not knowing
                    "can't",  # Can't do
                    "impossible",  # Impossible
                    "too hard"  # Too hard
                ]
            }
        }
        
        # Task-specific rewards
        self.task_rewards = {
            "add_feature": {
                "base_reward": 0.5,
                "completion_bonus": 0.3,
                "quality_bonus": 0.2,
                "testing_bonus": 0.1
            },
            "fix_bug": {
                "base_reward": 0.4,
                "fix_bonus": 0.4,
                "prevention_bonus": 0.2
            },
            "refactor": {
                "base_reward": 0.3,
                "improvement_bonus": 0.4,
                "maintainability_bonus": 0.3
            },
            "add_tests": {
                "base_reward": 0.4,
                "coverage_bonus": 0.3,
                "quality_bonus": 0.3
            },
            "optimize": {
                "base_reward": 0.3,
                "performance_bonus": 0.5,
                "readability_bonus": 0.2
            }
        }
    
    def analyze_code_quality(self, code: str) -> Dict[str, float]:
        """Analyze code quality and return scores"""
        scores = {}
        
        for category, config in self.coding_rewards.items():
            score = 0.0
            total_weight = sum(config["weights"].values())
            
            for aspect, weight in config["weights"].items():
                aspect_score = 0.0
                
                # Check positive signals
                for signal in config["positive_signals"]:
                    if signal in code:
                        aspect_score += 0.1
                
                # Check negative signals
                for signal in config["negative_signals"]:
                    if signal in code:
                        aspect_score -= 0.1
                
                # Normalize and weight
                aspect_score = max(0.0, min(1.0, aspect_score))
                score += aspect_score * weight
            
            scores[category] = score / total_weight
        
        return scores
    
    def calculate_task_reward(self, task_type: str, code: str, success: bool) -> float:
        """Calculate reward for a specific task"""
        if task_type not in self.task_rewards:
            return 0.5  # Default reward
        
        task_config = self.task_rewards[task_type]
        base_reward = task_config["base_reward"]
        
        if not success:
            return base_reward * 0.1  # Minimal reward for failure
        
        # Analyze code quality
        quality_scores = self.analyze_code_quality(code)
        
        # Calculate bonuses
        total_reward = base_reward
        
        # Completion bonus
        if success:
            total_reward += task_config.get("completion_bonus", 0.0)
        
        # Quality bonus
        avg_quality = sum(quality_scores.values()) / len(quality_scores)
        total_reward += avg_quality * task_config.get("quality_bonus", 0.0)
        
        # Task-specific bonuses
        if task_type == "add_feature":
            if "def " in code and "return " in code:
                total_reward += task_config["testing_bonus"]
        
        elif task_type == "fix_bug":
            if "try:" in code or "if " in code:
                total_reward += task_config["prevention_bonus"]
        
        elif task_type == "refactor":
            if "def " in code and len(code.split('\n')) > 10:
                total_reward += task_config["maintainability_bonus"]
        
        elif task_type == "add_tests":
            if "test_" in code and "assert " in code:
                total_reward += task_config["coverage_bonus"]
        
        elif task_type == "optimize":
            if "import time" in code or "profile" in code:
                total_reward += task_config["performance_bonus"]
        
        return max(0.0, min(1.0, total_reward))
    
    def setup_reward_shaping(self):
        """Set up the reward shaping configuration"""
        config = {
            "coding_rewards": self.coding_rewards,
            "task_rewards": self.task_rewards,
            "setup_date": str(Path().cwd()),
            "description": "Reward shaping for productive coding behaviors"
        }
        
        with open(self.reward_config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Reward shaping configuration saved to: {self.reward_config_file}")
    
    def show_reward_structure(self):
        """Display the reward structure"""
        print("🎯 DSPy Agent Coding Reward Structure")
        print("="*60)
        
        for category, config in self.coding_rewards.items():
            print(f"\n📊 {category.replace('_', ' ').title()}")
            print(f"   Description: {config['description']}")
            print(f"   Weights:")
            for aspect, weight in config["weights"].items():
                print(f"     {aspect}: {weight}")
            print(f"   Positive Signals: {', '.join(config['positive_signals'][:3])}...")
            print(f"   Negative Signals: {', '.join(config['negative_signals'][:3])}...")
        
        print(f"\n🎯 Task-Specific Rewards:")
        for task, config in self.task_rewards.items():
            print(f"   {task}: {config['base_reward']} base + bonuses")
    
    def test_reward_calculation(self):
        """Test the reward calculation with sample code"""
        print("\n🧪 Testing Reward Calculation")
        print("-" * 40)
        
        sample_codes = {
            "good_code": '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def test_fibonacci():
    """Test the fibonacci function."""
    assert calculate_fibonacci(0) == 0
    assert calculate_fibonacci(1) == 1
    assert calculate_fibonacci(10) == 55
''',
            
            "bad_code": '''
def fib(n):
    # TODO: implement this
    pass

def calc():
    print("calculating...")
    return None
'''
        }
        
        for name, code in sample_codes.items():
            print(f"\n📝 {name.replace('_', ' ').title()}:")
            quality_scores = self.analyze_code_quality(code)
            for category, score in quality_scores.items():
                print(f"   {category}: {score:.2f}")
            
            avg_score = sum(quality_scores.values()) / len(quality_scores)
            print(f"   Average: {avg_score:.2f}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Coding Reward Shaping")
    parser.add_argument("--show", "-s", action="store_true", help="Show reward structure")
    parser.add_argument("--test", "-t", action="store_true", help="Test reward calculation")
    parser.add_argument("--setup", action="store_true", help="Setup reward shaping")
    
    args = parser.parse_args()
    
    shaper = CodingRewardShaper(project_root)
    
    if args.show:
        shaper.show_reward_structure()
    
    if args.test:
        shaper.test_reward_calculation()
    
    if args.setup:
        shaper.setup_reward_shaping()
    
    if not any([args.show, args.test, args.setup]):
        # Default: show and setup
        shaper.show_reward_structure()
        shaper.test_reward_calculation()
        shaper.setup_reward_shaping()

if __name__ == "__main__":
    main()
