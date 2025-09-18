#!/usr/bin/env python3
"""
Coding Agent Training System

This script trains the DSPy Agent to be a productive coding partner by:
1. Running realistic coding tasks
2. Measuring success with quality metrics
3. Shaping rewards based on code quality and functionality
4. Building up coding expertise over time
"""

import os
import sys
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class CodingTask:
    """A coding task for training the agent"""
    name: str
    description: str
    requirements: List[str]
    success_criteria: List[str]
    difficulty: str  # "easy", "medium", "hard"
    estimated_time: int  # minutes
    test_commands: List[str]
    quality_checks: List[str]

@dataclass
class TaskResult:
    """Result of a coding task"""
    task: CodingTask
    success: bool
    quality_score: float  # 0.0 to 1.0
    functionality_score: float  # 0.0 to 1.0
    code_quality_score: float  # 0.0 to 1.0
    time_taken: float  # minutes
    errors: List[str]
    improvements: List[str]
    reward: float

class CodingAgentTrainer:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.training_tasks = self._create_training_tasks()
        self.training_results = []
        
        # Quality metrics
        self.quality_weights = {
            "functionality": 0.4,  # Does it work?
            "code_quality": 0.3,   # Is it well-written?
            "testing": 0.2,        # Are there tests?
            "documentation": 0.1   # Is it documented?
        }
    
    def _create_training_tasks(self) -> List[CodingTask]:
        """Create realistic coding tasks for training"""
        return [
            CodingTask(
                name="Add Error Handling",
                description="Add comprehensive error handling to a function",
                requirements=[
                    "Find a function that lacks error handling",
                    "Add try-catch blocks for potential failures",
                    "Add meaningful error messages",
                    "Ensure graceful degradation"
                ],
                success_criteria=[
                    "Function handles errors gracefully",
                    "Error messages are informative",
                    "No unhandled exceptions",
                    "Code is more robust"
                ],
                difficulty="easy",
                estimated_time=15,
                test_commands=[
                    "python -m pytest tests/ -k error_handling",
                    "python -c \"import ast; ast.parse(open('target_file.py').read())\""
                ],
                quality_checks=[
                    "grep -n 'try:' target_file.py",
                    "grep -n 'except' target_file.py",
                    "grep -n 'raise' target_file.py"
                ]
            ),
            
            CodingTask(
                name="Add Unit Tests",
                description="Add comprehensive unit tests for existing code",
                requirements=[
                    "Identify functions that need testing",
                    "Write test cases covering edge cases",
                    "Ensure good test coverage",
                    "Follow testing best practices"
                ],
                success_criteria=[
                    "All functions have tests",
                    "Tests cover edge cases",
                    "Tests pass successfully",
                    "Good test coverage achieved"
                ],
                difficulty="medium",
                estimated_time=30,
                test_commands=[
                    "python -m pytest tests/ -v",
                    "python -m coverage run -m pytest tests/",
                    "python -m coverage report"
                ],
                quality_checks=[
                    "find tests/ -name 'test_*.py' | wc -l",
                    "grep -r 'def test_' tests/ | wc -l"
                ]
            ),
            
            CodingTask(
                name="Refactor Code",
                description="Refactor code to improve readability and maintainability",
                requirements=[
                    "Identify code that needs refactoring",
                    "Extract functions for better modularity",
                    "Improve variable names and structure",
                    "Maintain functionality while improving code"
                ],
                success_criteria=[
                    "Code is more readable",
                    "Functions are well-organized",
                    "No functionality is broken",
                    "Code follows best practices"
                ],
                difficulty="medium",
                estimated_time=25,
                test_commands=[
                    "python -m pytest tests/",
                    "python -m flake8 target_file.py",
                    "python -m black --check target_file.py"
                ],
                quality_checks=[
                    "grep -c 'def ' target_file.py",
                    "wc -l target_file.py"
                ]
            ),
            
            CodingTask(
                name="Add New Feature",
                description="Implement a new feature from scratch",
                requirements=[
                    "Understand the feature requirements",
                    "Design the implementation approach",
                    "Write clean, maintainable code",
                    "Add appropriate tests and documentation"
                ],
                success_criteria=[
                    "Feature works as specified",
                    "Code is well-structured",
                    "Tests are comprehensive",
                    "Documentation is clear"
                ],
                difficulty="hard",
                estimated_time=45,
                test_commands=[
                    "python -m pytest tests/ -k feature_name",
                    "python -c \"from module import new_feature; new_feature()\""
                ],
                quality_checks=[
                    "grep -n 'def new_feature' target_file.py",
                    "grep -n 'class.*Feature' target_file.py"
                ]
            ),
            
            CodingTask(
                name="Fix Bug",
                description="Identify and fix a bug in the codebase",
                requirements=[
                    "Reproduce the bug",
                    "Identify the root cause",
                    "Implement a fix",
                    "Verify the fix works"
                ],
                success_criteria=[
                    "Bug is fixed",
                    "Fix doesn't break other functionality",
                    "Root cause is addressed",
                    "Tests pass"
                ],
                difficulty="medium",
                estimated_time=20,
                test_commands=[
                    "python -m pytest tests/ -k bug_fix",
                    "python -c \"# Test that reproduces the bug\""
                ],
                quality_checks=[
                    "grep -n 'fix' target_file.py",
                    "grep -n 'bug' target_file.py"
                ]
            ),
            
            CodingTask(
                name="Optimize Performance",
                description="Optimize code for better performance",
                requirements=[
                    "Identify performance bottlenecks",
                    "Implement optimizations",
                    "Measure performance improvements",
                    "Maintain code readability"
                ],
                success_criteria=[
                    "Performance is improved",
                    "Code is still readable",
                    "Functionality is preserved",
                    "Optimizations are justified"
                ],
                difficulty="hard",
                estimated_time=35,
                test_commands=[
                    "python -m pytest tests/",
                    "python -m timeit 'import module; module.function()'"
                ],
                quality_checks=[
                    "grep -n 'import time' target_file.py",
                    "grep -n 'profile' target_file.py"
                ]
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
    
    def evaluate_code_quality(self, task: CodingTask, output: str) -> float:
        """Evaluate the quality of code produced"""
        quality_score = 0.0
        
        # Check for code structure
        if "def " in output or "class " in output:
            quality_score += 0.2
        
        # Check for error handling
        if "try:" in output or "except" in output:
            quality_score += 0.2
        
        # Check for documentation
        if '"""' in output or "'''" in output or "# " in output:
            quality_score += 0.2
        
        # Check for testing
        if "test" in output.lower() or "pytest" in output.lower():
            quality_score += 0.2
        
        # Check for best practices
        if "import" in output and "from" in output:
            quality_score += 0.1
        
        # Check for proper formatting
        if output.count('\n') > 5:  # Multi-line code
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def evaluate_functionality(self, task: CodingTask, output: str) -> float:
        """Evaluate if the code actually works"""
        functionality_score = 0.0
        
        # Check if the agent understood the task
        task_keywords = task.name.lower().split()
        for keyword in task_keywords:
            if keyword in output.lower():
                functionality_score += 0.2
        
        # Check for specific requirements
        for requirement in task.requirements:
            req_words = requirement.lower().split()[:3]  # First 3 words
            if any(word in output.lower() for word in req_words):
                functionality_score += 0.1
        
        # Check for success criteria
        for criteria in task.success_criteria:
            crit_words = criteria.lower().split()[:3]
            if any(word in output.lower() for word in crit_words):
                functionality_score += 0.1
        
        return min(functionality_score, 1.0)
    
    def run_quality_checks(self, task: CodingTask) -> List[str]:
        """Run quality checks on the code"""
        results = []
        
        for check in task.quality_checks:
            try:
                result = subprocess.run(
                    ["bash", "-c", check],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
                if result.returncode == 0:
                    results.append(f"✅ {check}: {result.stdout.strip()}")
                else:
                    results.append(f"❌ {check}: {result.stderr.strip()}")
            except Exception as e:
                results.append(f"⚠️ {check}: {e}")
        
        return results
    
    def run_test_commands(self, task: CodingTask) -> List[str]:
        """Run test commands to verify functionality"""
        results = []
        
        for test_cmd in task.test_commands:
            try:
                result = subprocess.run(
                    ["bash", "-c", test_cmd],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
                if result.returncode == 0:
                    results.append(f"✅ {test_cmd}: PASSED")
                else:
                    results.append(f"❌ {test_cmd}: FAILED - {result.stderr.strip()}")
            except Exception as e:
                results.append(f"⚠️ {test_cmd}: ERROR - {e}")
        
        return results
    
    def calculate_reward(self, result: TaskResult) -> float:
        """Calculate reward based on task performance"""
        base_reward = 0.0
        
        # Base reward for attempting the task
        base_reward += 0.1
        
        # Quality-based rewards
        base_reward += result.quality_score * 0.3
        base_reward += result.functionality_score * 0.4
        base_reward += result.code_quality_score * 0.2
        
        # Time-based bonus (faster is better, up to a point)
        if result.time_taken < task.estimated_time * 0.5:
            base_reward += 0.1  # Bonus for being fast
        elif result.time_taken > task.estimated_time * 2:
            base_reward -= 0.1  # Penalty for being slow
        
        # Success bonus
        if result.success:
            base_reward += 0.2
        
        # Error penalty
        base_reward -= len(result.errors) * 0.05
        
        return max(0.0, min(1.0, base_reward))
    
    def run_training_task(self, task: CodingTask) -> TaskResult:
        """Run a single training task"""
        print(f"\n🎯 Training Task: {task.name}")
        print(f"📝 Description: {task.description}")
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
        
        # Run the task with the agent
        success, output, error = self.run_agent_command(f"plan \"{task_prompt}\"")
        
        if not success:
            print(f"❌ Task failed to start: {error}")
            return TaskResult(
                task=task,
                success=False,
                quality_score=0.0,
                functionality_score=0.0,
                code_quality_score=0.0,
                time_taken=time.time() - start_time,
                errors=[error],
                improvements=[],
                reward=0.0
            )
        
        # Run the implementation
        success, output, error = self.run_agent_command(f"edit \"{task.description}\" --apply")
        
        time_taken = (time.time() - start_time) / 60  # Convert to minutes
        
        # Evaluate the results
        quality_score = self.evaluate_code_quality(task, output)
        functionality_score = self.evaluate_functionality(task, output)
        code_quality_score = (quality_score + functionality_score) / 2
        
        # Run quality checks
        quality_results = self.run_quality_checks(task)
        
        # Run tests
        test_results = self.run_test_commands(task)
        
        # Determine success
        task_success = (
            quality_score > 0.6 and
            functionality_score > 0.6 and
            len([r for r in test_results if "PASSED" in r]) > 0
        )
        
        # Calculate reward
        result = TaskResult(
            task=task,
            success=task_success,
            quality_score=quality_score,
            functionality_score=functionality_score,
            code_quality_score=code_quality_score,
            time_taken=time_taken,
            errors=[error] if error else [],
            improvements=quality_results + test_results,
            reward=0.0  # Will be calculated
        )
        
        result.reward = self.calculate_reward(result)
        
        # Display results
        print(f"\n📊 Task Results:")
        print(f"   Success: {'✅' if task_success else '❌'}")
        print(f"   Quality Score: {quality_score:.2f}")
        print(f"   Functionality Score: {functionality_score:.2f}")
        print(f"   Code Quality Score: {code_quality_score:.2f}")
        print(f"   Time Taken: {time_taken:.1f} minutes")
        print(f"   Reward: {result.reward:.3f}")
        
        if quality_results:
            print(f"\n🔍 Quality Checks:")
            for result_line in quality_results:
                print(f"   {result_line}")
        
        if test_results:
            print(f"\n🧪 Test Results:")
            for result_line in test_results:
                print(f"   {result_line}")
        
        return result
    
    def run_training_session(self, num_tasks: int = 3) -> List[TaskResult]:
        """Run a training session with multiple tasks"""
        print("🚀 DSPy Agent Coding Training Session")
        print("="*60)
        print("This session will train the agent to be a productive coding partner.")
        print("The agent will learn to write quality code, fix bugs, and build features.")
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
    
    def display_training_summary(self, results: List[TaskResult]):
        """Display training session summary"""
        print("\n🎉 Training Session Complete!")
        print("="*60)
        
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        avg_quality = sum(r.quality_score for r in results) / total_tasks
        avg_functionality = sum(r.functionality_score for r in results) / total_tasks
        avg_reward = sum(r.reward for r in results) / total_tasks
        
        print(f"📊 Training Results:")
        print(f"   Total Tasks: {total_tasks}")
        print(f"   Successful Tasks: {successful_tasks}")
        print(f"   Success Rate: {successful_tasks/total_tasks:.1%}")
        print(f"   Average Quality Score: {avg_quality:.2f}")
        print(f"   Average Functionality Score: {avg_functionality:.2f}")
        print(f"   Average Reward: {avg_reward:.3f}")
        
        print(f"\n🎯 Learning Progress:")
        if avg_reward > 0.7:
            print("   🎉 Excellent! The agent is learning to be a great coding partner!")
        elif avg_reward > 0.5:
            print("   ✅ Good progress! The agent is developing coding skills.")
        elif avg_reward > 0.3:
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
    
    parser = argparse.ArgumentParser(description="Train DSPy Agent for Coding")
    parser.add_argument("--tasks", "-t", type=int, default=3, help="Number of tasks to run (default: 3)")
    parser.add_argument("--difficulty", "-d", choices=["easy", "medium", "hard"], help="Task difficulty level")
    
    args = parser.parse_args()
    
    trainer = CodingAgentTrainer(project_root)
    results = trainer.run_training_session(args.tasks)
    
    # Save training results
    results_file = project_root / "training_results.json"
    with open(results_file, "w") as f:
        json.dump([{
            "task": result.task.name,
            "success": result.success,
            "quality_score": result.quality_score,
            "functionality_score": result.functionality_score,
            "reward": result.reward,
            "time_taken": result.time_taken
        } for result in results], f, indent=2)
    
    print(f"\n📄 Training results saved to: {results_file}")

if __name__ == "__main__":
    main()
