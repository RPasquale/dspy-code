#!/usr/bin/env python3
"""
Train DSPy Agent to be a Productive Coding Partner

This script combines verifiers (technical assessment) with reward shaping
(behavioral guidance) to train the agent to be a skilled developer.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class TrainingTask:
    """A training task for the agent"""
    name: str
    description: str
    task_type: str  # "add_feature", "fix_bug", "refactor", "add_tests", "optimize"
    requirements: List[str]
    success_criteria: List[str]
    difficulty: str
    estimated_time: int

class ProductiveAgentTrainer:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.training_tasks = self._create_training_tasks()
        self.training_results = []
        
        # Load reward shaping configuration
        self.reward_config = self._load_reward_config()
        
    def _load_reward_config(self) -> Dict[str, Any]:
        """Load reward shaping configuration"""
        config_file = project_root / ".dspy_coding_rewards.json"
        if config_file.exists():
            with open(config_file) as f:
                return json.load(f)
        return {}
    
    def _create_training_tasks(self) -> List[TrainingTask]:
        """Create realistic training tasks"""
        return [
            TrainingTask(
                name="Add Error Handling to Calculator",
                description="Add comprehensive error handling to the calculator functions",
                task_type="fix_bug",
                requirements=[
                    "Find the calculator.py file",
                    "Add try-catch blocks for division by zero",
                    "Add input validation for non-numeric inputs",
                    "Add meaningful error messages",
                    "Ensure the calculator is robust"
                ],
                success_criteria=[
                    "Calculator handles division by zero gracefully",
                    "Calculator validates input types",
                    "Error messages are informative",
                    "All existing functionality still works"
                ],
                difficulty="easy",
                estimated_time=15
            ),
            
            TrainingTask(
                name="Add Unit Tests for Calculator",
                description="Add comprehensive unit tests for the calculator functions",
                task_type="add_tests",
                requirements=[
                    "Create test_calculator.py file",
                    "Write tests for all calculator functions",
                    "Test edge cases (division by zero, invalid inputs)",
                    "Ensure good test coverage",
                    "Follow testing best practices"
                ],
                success_criteria=[
                    "All calculator functions have tests",
                    "Edge cases are covered",
                    "Tests pass successfully",
                    "Good test coverage achieved"
                ],
                difficulty="medium",
                estimated_time=25
            ),
            
            TrainingTask(
                name="Refactor Calculator for Better Structure",
                description="Refactor the calculator to improve code organization",
                task_type="refactor",
                requirements=[
                    "Analyze current calculator structure",
                    "Extract functions for better modularity",
                    "Improve variable names and code clarity",
                    "Maintain all existing functionality",
                    "Follow Python best practices"
                ],
                success_criteria=[
                    "Code is more readable and organized",
                    "Functions are well-structured",
                    "No functionality is broken",
                    "Code follows PEP 8 standards"
                ],
                difficulty="medium",
                estimated_time=20
            ),
            
            TrainingTask(
                name="Add New Calculator Feature",
                description="Add a new mathematical operation to the calculator",
                task_type="add_feature",
                requirements=[
                    "Choose a new mathematical operation (e.g., power, square root)",
                    "Implement the function with proper error handling",
                    "Add it to the calculator interface",
                    "Write tests for the new feature",
                    "Update documentation"
                ],
                success_criteria=[
                    "New feature works correctly",
                    "Feature is well-tested",
                    "Feature integrates with existing code",
                    "Documentation is updated"
                ],
                difficulty="hard",
                estimated_time=35
            ),
            
            TrainingTask(
                name="Optimize Calculator Performance",
                description="Optimize the calculator for better performance",
                task_type="optimize",
                requirements=[
                    "Profile the calculator to find bottlenecks",
                    "Optimize slow operations",
                    "Improve memory usage if needed",
                    "Maintain code readability",
                    "Measure performance improvements"
                ],
                success_criteria=[
                    "Performance is measurably improved",
                    "Code is still readable",
                    "All functionality is preserved",
                    "Optimizations are justified"
                ],
                difficulty="hard",
                estimated_time=30
            )
        ]
    
    def run_agent_command(self, command: str, timeout: int = 60) -> Tuple[bool, str, str]:
        """Run a command with the agent"""
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
    
    def assess_with_verifiers(self, task: TrainingTask, output: str) -> Dict[str, float]:
        """Assess the result using built-in verifiers"""
        verifier_scores = {}
        
        # Simulate verifier assessments based on output analysis
        # In a real system, these would be actual verifier calls
        
        # Pass Rate Verifier (tests passing)
        if "test" in output.lower() and "pass" in output.lower():
            verifier_scores["pass_rate"] = 0.8
        elif "test" in output.lower():
            verifier_scores["pass_rate"] = 0.5
        else:
            verifier_scores["pass_rate"] = 0.2
        
        # Blast Radius Verifier (code change scope)
        lines_changed = output.count('\n')
        if lines_changed < 10:
            verifier_scores["blast_radius"] = 0.9  # Small changes are good
        elif lines_changed < 50:
            verifier_scores["blast_radius"] = 0.7
        else:
            verifier_scores["blast_radius"] = 0.4
        
        # Code Quality Verifier
        quality_signals = 0
        if "def " in output:
            quality_signals += 1
        if "try:" in output or "except" in output:
            quality_signals += 1
        if '"""' in output or "'''" in output:
            quality_signals += 1
        if "import" in output:
            quality_signals += 1
        
        verifier_scores["code_quality"] = min(quality_signals / 4.0, 1.0)
        
        return verifier_scores
    
    def assess_with_reward_shaping(self, task: TrainingTask, output: str, success: bool) -> Dict[str, float]:
        """Assess using reward shaping for behavioral guidance"""
        if not self.reward_config:
            return {"behavioral_score": 0.5}
        
        coding_rewards = self.reward_config.get("coding_rewards", {})
        task_rewards = self.reward_config.get("task_rewards", {})
        
        # Calculate behavioral scores
        behavioral_scores = {}
        
        for category, config in coding_rewards.items():
            score = 0.0
            total_weight = sum(config["weights"].values())
            
            for aspect, weight in config["weights"].items():
                aspect_score = 0.0
                
                # Check positive signals
                for signal in config["positive_signals"]:
                    if signal in output:
                        aspect_score += 0.1
                
                # Check negative signals
                for signal in config["negative_signals"]:
                    if signal in output:
                        aspect_score -= 0.1
                
                aspect_score = max(0.0, min(1.0, aspect_score))
                score += aspect_score * weight
            
            behavioral_scores[category] = score / total_weight
        
        # Calculate task-specific reward
        task_config = task_rewards.get(task.task_type, {"base_reward": 0.5})
        task_reward = task_config["base_reward"]
        
        if success:
            task_reward += task_config.get("completion_bonus", 0.0)
        
        # Quality bonus
        avg_quality = sum(behavioral_scores.values()) / len(behavioral_scores) if behavioral_scores else 0.5
        task_reward += avg_quality * task_config.get("quality_bonus", 0.0)
        
        behavioral_scores["task_reward"] = max(0.0, min(1.0, task_reward))
        
        return behavioral_scores
    
    def run_training_task(self, task: TrainingTask) -> Dict[str, Any]:
        """Run a single training task"""
        print(f"\n🎯 Training Task: {task.name}")
        print(f"📝 Description: {task.description}")
        print(f"🏷️  Type: {task.task_type}")
        print(f"📋 Requirements:")
        for req in task.requirements:
            print(f"   • {req}")
        print(f"🎯 Success Criteria:")
        for crit in task.success_criteria:
            print(f"   • {crit}")
        print(f"⏱️  Estimated Time: {task.estimated_time} minutes")
        print("-" * 60)
        
        start_time = time.time()
        
        # Create the task prompt
        task_prompt = f"""
        Task: {task.name}
        Description: {task.description}
        
        Requirements:
        {chr(10).join(f"- {req}" for req in task.requirements)}
        
        Success Criteria:
        {chr(10).join(f"- {crit}" for crit in task.success_criteria)}
        
        Please implement this task step by step. Start by understanding the requirements, then implement the solution, and finally verify it works.
        """
        
        # Step 1: Plan the task
        print("📋 Planning the task...")
        success, plan_output, error = self.run_agent_command(f"plan \"{task_prompt}\"")
        
        if not success:
            print(f"❌ Planning failed: {error}")
            return {
                "task": task.name,
                "success": False,
                "error": error,
                "time_taken": (time.time() - start_time) / 60
            }
        
        # Step 2: Implement the solution
        print("🔨 Implementing the solution...")
        success, impl_output, error = self.run_agent_command(f"edit \"{task.description}\" --apply")
        
        time_taken = (time.time() - start_time) / 60
        
        # Step 3: Assess with verifiers
        print("🔍 Assessing with verifiers...")
        verifier_scores = self.assess_with_verifiers(task, impl_output)
        
        # Step 4: Assess with reward shaping
        print("🎯 Assessing with reward shaping...")
        behavioral_scores = self.assess_with_reward_shaping(task, impl_output, success)
        
        # Step 5: Calculate combined reward
        verifier_reward = sum(verifier_scores.values()) / len(verifier_scores) if verifier_scores else 0.0
        behavioral_reward = behavioral_scores.get("task_reward", 0.5)
        combined_reward = 0.7 * behavioral_reward + 0.3 * verifier_reward
        
        # Determine overall success
        overall_success = (
            success and
            verifier_reward > 0.6 and
            behavioral_reward > 0.6
        )
        
        # Display results
        print(f"\n📊 Training Results:")
        print(f"   Overall Success: {'✅' if overall_success else '❌'}")
        print(f"   Time Taken: {time_taken:.1f} minutes")
        print(f"   Combined Reward: {combined_reward:.3f}")
        
        print(f"\n🔍 Verifier Scores:")
        for verifier, score in verifier_scores.items():
            print(f"   {verifier}: {score:.2f}")
        
        print(f"\n🎯 Behavioral Scores:")
        for category, score in behavioral_scores.items():
            if category != "task_reward":
                print(f"   {category}: {score:.2f}")
        print(f"   task_reward: {behavioral_scores.get('task_reward', 0.0):.2f}")
        
        return {
            "task": task.name,
            "task_type": task.task_type,
            "success": overall_success,
            "time_taken": time_taken,
            "verifier_scores": verifier_scores,
            "behavioral_scores": behavioral_scores,
            "combined_reward": combined_reward,
            "plan_output": plan_output,
            "impl_output": impl_output
        }
    
    def run_training_session(self, num_tasks: int = 3) -> List[Dict[str, Any]]:
        """Run a training session"""
        print("🚀 DSPy Agent Productive Coding Training")
        print("="*60)
        print("This session combines verifiers (technical assessment) with")
        print("reward shaping (behavioral guidance) to train the agent to")
        print("be a skilled, productive coding partner.")
        print()
        
        # Select tasks based on difficulty progression
        easy_tasks = [t for t in self.training_tasks if t.difficulty == "easy"]
        medium_tasks = [t for t in self.training_tasks if t.difficulty == "medium"]
        hard_tasks = [t for t in self.training_tasks if t.difficulty == "hard"]
        
        selected_tasks = []
        if num_tasks >= 1 and easy_tasks:
            selected_tasks.append(easy_tasks[0])
        if num_tasks >= 2 and medium_tasks:
            selected_tasks.append(medium_tasks[0])
        if num_tasks >= 3 and hard_tasks:
            selected_tasks.append(hard_tasks[0])
        
        results = []
        
        for i, task in enumerate(selected_tasks, 1):
            print(f"\n🎯 Training Task {i}/{len(selected_tasks)}")
            result = self.run_training_task(task)
            results.append(result)
            
            if i < len(selected_tasks):
                input("\nPress Enter to continue to next task...")
        
        # Training session summary
        self.display_training_summary(results)
        
        return results
    
    def display_training_summary(self, results: List[Dict[str, Any]]):
        """Display training session summary"""
        print("\n🎉 Training Session Complete!")
        print("="*60)
        
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r["success"])
        
        if results:
            avg_verifier = sum(
                sum(r["verifier_scores"].values()) / len(r["verifier_scores"])
                for r in results if r["verifier_scores"]
            ) / total_tasks
            
            avg_behavioral = sum(
                r["behavioral_scores"].get("task_reward", 0.0)
                for r in results
            ) / total_tasks
            
            avg_combined = sum(r["combined_reward"] for r in results) / total_tasks
        else:
            avg_verifier = avg_behavioral = avg_combined = 0.0
        
        print(f"📊 Training Results:")
        print(f"   Total Tasks: {total_tasks}")
        print(f"   Successful Tasks: {successful_tasks}")
        print(f"   Success Rate: {successful_tasks/total_tasks:.1%}")
        print(f"   Average Verifier Score: {avg_verifier:.2f}")
        print(f"   Average Behavioral Score: {avg_behavioral:.2f}")
        print(f"   Average Combined Reward: {avg_combined:.3f}")
        
        print(f"\n🎯 Learning Assessment:")
        if avg_combined > 0.7:
            print("   🎉 Excellent! The agent is becoming a skilled coding partner!")
        elif avg_combined > 0.5:
            print("   ✅ Good progress! The agent is developing coding skills.")
        elif avg_combined > 0.3:
            print("   📈 The agent is learning, but needs more practice.")
        else:
            print("   ⚠️  The agent needs more training to become productive.")
        
        print(f"\n💡 Next Steps:")
        print("   • Run more training sessions to improve skills")
        print("   • Use the agent on real coding tasks")
        print("   • Monitor learning progress with: uv run python scripts/monitor_rl_learning.py")
        print("   • Check learning stats with: stats (in agent session)")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DSPy Agent for Productive Coding")
    parser.add_argument("--tasks", "-t", type=int, default=3, help="Number of tasks to run (default: 3)")
    parser.add_argument("--difficulty", "-d", choices=["easy", "medium", "hard"], help="Task difficulty level")
    
    args = parser.parse_args()
    
    trainer = ProductiveAgentTrainer(project_root)
    results = trainer.run_training_session(args.tasks)
    
    # Save training results
    results_file = project_root / "productive_training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Training results saved to: {results_file}")

if __name__ == "__main__":
    main()
