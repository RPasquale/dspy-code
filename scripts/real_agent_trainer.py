#!/usr/bin/env python3
"""
Real Agent Trainer - Uses DSPy to actually train an agent that writes code

This script creates a real AI agent that learns to write code by:
1. Analyzing project requirements
2. Writing actual code implementations
3. Testing and fixing issues
4. Learning from success/failure feedback
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import DSPy components
import dspy
from dspy_agent.db import get_enhanced_data_manager, create_action_record, ActionType, Environment
from dspy_agent.agentic import log_retrieval_event

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeTask:
    """Represents a code writing task."""
    project_name: str
    file_path: Path
    task_description: str
    current_code: str
    test_file: Path
    success_criteria: List[str]

class CodeWriter(dspy.Module):
    """DSPy module for writing code based on requirements."""
    
    def __init__(self):
        super().__init__()
        self.code_generator = dspy.ChainOfThought("task_description, current_code -> implementation")
    
    def forward(self, task_description: str, current_code: str) -> str:
        """Generate code implementation based on task description and current code."""
        try:
            result = self.code_generator(task_description=task_description, current_code=current_code)
            return result.implementation
        except Exception as e:
            logger.error(f"Error in CodeWriter: {e}")
            return f"# Error generating code: {e}"
    
    def __call__(self, task_description: str, current_code: str) -> str:
        """Allow calling the module directly."""
        return self.forward(task_description, current_code)

class CodeTester(dspy.Module):
    """DSPy module for analyzing test results and suggesting fixes."""
    
    def __init__(self):
        super().__init__()
        self.test_analyzer = dspy.ChainOfThought("test_output, code -> analysis_and_fix")
    
    def forward(self, test_output: str, code: str) -> str:
        """Analyze test output and suggest code fixes."""
        try:
            result = self.test_analyzer(test_output=test_output, code=code)
            return result.analysis_and_fix
        except Exception as e:
            logger.error(f"Error in CodeTester: {e}")
            return f"# Error analyzing test: {e}"
    
    def __call__(self, test_output: str, code: str) -> str:
        """Allow calling the module directly."""
        return self.forward(test_output, code)

class RealAgentTrainer:
    """Real agent trainer that uses DSPy to actually write code."""
    
    def __init__(self, workspace_path: Path = None):
        if workspace_path is None:
            # Use default agent workspace
            self.workspace_path = Path.home() / ".blampert_workspace"
        else:
            self.workspace_path = workspace_path
        
        self.projects_path = self.workspace_path / "projects"
        self.data_manager = get_enhanced_data_manager()
        
        # Initialize DSPy components
        self.code_writer = CodeWriter()
        self.code_tester = CodeTester()
        
        # Set up DSPy LM (using Ollama)
        try:
            # Use DSPy LM with Ollama configuration
            self.lm = dspy.LM(
                model="ollama/qwen3:1.7b",
                api_base="http://localhost:11435",
                max_tokens=4000
            )
            dspy.settings.configure(lm=self.lm)
            logger.info("Connected to Ollama service with qwen3:1.7b model")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            self.lm = None
        
        # Verify LM is configured
        if dspy.settings.get('lm') is None:
            logger.error("No LM configured in DSPy settings")
            self.lm = None
        else:
            logger.info("DSPy LM successfully configured")
        
        # Training history
        self.training_history = []
        
        logger.info("Real agent trainer initialized with DSPy")
    
    def analyze_project(self, project_path: Path) -> List[CodeTask]:
        """Analyze a project and extract code writing tasks."""
        tasks = []
        
        # Look for Python files with TODO comments
        for py_file in project_path.glob("**/*.py"):
            if py_file.name.startswith("test_"):
                continue  # Skip test files
                
            content = py_file.read_text()
            
            # Find TODO comments and extract tasks
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "TODO" in line and "implement" in line.lower():
                    # Extract task description
                    task_desc = line.split("TODO:")[-1].strip()
                    
                    # Find corresponding test file
                    test_file = project_path / f"test_{py_file.name}"
                    if not test_file.exists():
                        # Look for test files in same directory
                        test_files = list(project_path.glob(f"test_{py_file.stem}*.py"))
                        if test_files:
                            test_file = test_files[0]
                    
                    # Create task
                    task = CodeTask(
                        project_name=project_path.name,
                        file_path=py_file,
                        task_description=task_desc,
                        current_code=content,
                        test_file=test_file,
                        success_criteria=[
                            "Code compiles without syntax errors",
                            "All tests pass",
                            "Code follows best practices"
                        ]
                    )
                    tasks.append(task)
        
        logger.info(f"Found {len(tasks)} code writing tasks in {project_path.name}")
        return tasks
    
    def write_code(self, task: CodeTask) -> Tuple[str, List[str]]:
        """Use DSPy to actually write code for the task."""
        logger.info(f"Writing code for: {task.task_description}")
        
        try:
            # Create a focused prompt for the code writer
            prompt = f"""
You are an expert Python developer. Your task is to implement the following:

Task: {task.task_description}

Current code context:
```python
{task.current_code}
```

Requirements:
1. Implement the missing functionality
2. Follow Python best practices
3. Add proper error handling
4. Include docstrings
5. Make sure the code is production-ready

Return ONLY the complete, corrected Python code with your implementation.
Do not include explanations or markdown formatting.
"""
            
            # Use DSPy to generate code
            result = self.code_writer.forward(
                task_description=prompt,
                current_code=task.current_code
            )
            
            # Extract the generated code
            generated_code = result.implementation
            
            # Log the action
            self._log_code_generation(task, generated_code)
            
            return generated_code, []
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return task.current_code, [str(e)]
    
    def test_code(self, task: CodeTask, new_code: str) -> Dict[str, Any]:
        """Test the generated code and return results."""
        logger.info(f"Testing code for {task.project_name}")
        
        try:
            # Write the new code to a temporary file
            temp_file = task.file_path.parent / f"temp_{task.file_path.name}"
            temp_file.write_text(new_code)
            
            # Run tests
            if task.test_file.exists():
                result = subprocess.run([
                    "python", "-m", "pytest", str(task.test_file), "-v"
                ], capture_output=True, text=True, timeout=30)
                
                # Parse test results
                output_lines = result.stdout.split('\n')
                passed_tests = sum(1 for line in output_lines if 'PASSED' in line)
                failed_tests = sum(1 for line in output_lines if 'FAILED' in line)
                total_tests = passed_tests + failed_tests
                
                success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
                
                return {
                    "success": success_rate >= 0.9,
                    "success_rate": success_rate,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "total_tests": total_tests,
                    "output": result.stdout,
                    "errors": result.stderr.split('\n') if result.stderr else []
                }
            else:
                # No test file, just check syntax
                syntax_result = subprocess.run([
                    "python", "-m", "py_compile", str(temp_file)
                ], capture_output=True, text=True)
                
                return {
                    "success": syntax_result.returncode == 0,
                    "success_rate": 1.0 if syntax_result.returncode == 0 else 0.0,
                    "passed_tests": 1 if syntax_result.returncode == 0 else 0,
                    "failed_tests": 0 if syntax_result.returncode == 0 else 1,
                    "total_tests": 1,
                    "output": "Syntax check",
                    "errors": syntax_result.stderr.split('\n') if syntax_result.stderr else []
                }
                
        except Exception as e:
            logger.error(f"Error testing code: {e}")
            return {
                "success": False,
                "success_rate": 0.0,
                "passed_tests": 0,
                "failed_tests": 1,
                "total_tests": 1,
                "output": "",
                "errors": [str(e)]
            }
        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
    
    def fix_code(self, task: CodeTask, test_result: Dict[str, Any]) -> str:
        """Use DSPy to analyze test failures and fix the code."""
        if test_result["success"]:
            return task.current_code  # No fixes needed
        
        logger.info(f"Analyzing test failures and fixing code...")
        
        try:
            # Create prompt for code fixing
            prompt = f"""
The following code failed tests. Analyze the test output and fix the issues:

Test Output:
{test_result['output']}

Errors:
{test_result['errors']}

Current Code:
```python
{task.current_code}
```

Please provide the corrected code that will pass all tests.
Return ONLY the complete, corrected Python code.
"""
            
            # Use DSPy to analyze and fix
            result = self.code_tester.forward(
                test_output=test_result['output'],
                code=task.current_code
            )
            
            # Extract the fixed code
            fixed_code = result.analysis_and_fix
            
            # Log the fix attempt
            self._log_code_fix(task, test_result, fixed_code)
            
            return fixed_code
            
        except Exception as e:
            logger.error(f"Error fixing code: {e}")
            return task.current_code
    
    def train_on_project(self, project_path: Path, max_iterations: int = 3) -> Dict[str, Any]:
        """Train the agent on a specific project."""
        logger.info(f"Training agent on project: {project_path.name}")
        
        # Analyze project to find tasks
        tasks = self.analyze_project(project_path)
        
        if not tasks:
            logger.warning(f"No code writing tasks found in {project_path.name}")
            return {"success": False, "score": 0.0, "message": "No tasks found"}
        
        results = []
        
        for task in tasks:
            logger.info(f"Working on task: {task.task_description}")
            
            best_code = task.current_code
            best_score = 0.0
            
            for iteration in range(max_iterations):
                logger.info(f"  Iteration {iteration + 1}/{max_iterations}")
                
                # Generate code
                new_code, errors = self.write_code(task)
                
                if errors:
                    logger.warning(f"Code generation errors: {errors}")
                    continue
                
                # Test the code
                test_result = self.test_code(task, new_code)
                
                # Update best result
                if test_result["success_rate"] > best_score:
                    best_code = new_code
                    best_score = test_result["success_rate"]
                
                # If successful, we're done
                if test_result["success"]:
                    logger.info(f"  Task completed successfully in {iteration + 1} iterations!")
                    break
                
                # If not successful and we have more iterations, try to fix
                if iteration < max_iterations - 1:
                    new_code = self.fix_code(task, test_result)
                    test_result = self.test_code(task, new_code)
                    
                    if test_result["success_rate"] > best_score:
                        best_code = new_code
                        best_score = test_result["success_rate"]
            
            # Save the best code
            if best_score > 0.0:
                task.file_path.write_text(best_code)
                logger.info(f"Saved improved code with {best_score:.1%} success rate")
            
            results.append({
                "task": task.task_description,
                "success_rate": best_score,
                "iterations_used": iteration + 1
            })
        
        # Calculate overall project score
        overall_score = sum(r["success_rate"] for r in results) / len(results) if results else 0.0
        overall_success = overall_score >= 0.8
        
        return {
            "success": overall_success,
            "score": overall_score,
            "results": results,
            "tasks_completed": len(results)
        }
    
    def _log_code_generation(self, task: CodeTask, generated_code: str):
        """Log code generation action to RedDB."""
        action = create_action_record(
            action_type=ActionType.CODE_ANALYSIS,
            state_before={"file": str(task.file_path), "task": task.task_description},
            state_after={"file": str(task.file_path), "code_generated": True},
            parameters={"task_description": task.task_description},
            result={"generated_code_length": len(generated_code)},
            reward=0.5,  # Neutral reward for generation attempt
            confidence=0.7,
            execution_time=0.0,
            environment=Environment.DEVELOPMENT
        )
        
        self.data_manager.record_action(action)
    
    def _log_code_fix(self, task: CodeTask, test_result: Dict[str, Any], fixed_code: str):
        """Log code fixing action to RedDB."""
        action = create_action_record(
            action_type=ActionType.CODE_ANALYSIS,
            state_before={"file": str(task.file_path), "test_failures": test_result["failed_tests"]},
            state_after={"file": str(task.file_path), "fix_attempted": True},
            parameters={"test_output": test_result["output"][:500]},  # Truncate for storage
            result={"fixed_code_length": len(fixed_code)},
            reward=0.3,  # Lower reward for fix attempts
            confidence=0.6,
            execution_time=0.0,
            environment=Environment.DEVELOPMENT
        )
        
        self.data_manager.record_action(action)
    
    def run_training_session(self, project_name: str = None, max_iterations: int = 3) -> List[Dict[str, Any]]:
        """Run a complete training session."""
        logger.info("Starting real agent training session...")
        
        # Find projects to train on
        if project_name:
            project_paths = [self.projects_path / project_name]
        else:
            project_paths = [p for p in self.projects_path.iterdir() if p.is_dir()]
        
        results = []
        
        for project_path in project_paths:
            if not project_path.exists():
                continue
                
            logger.info(f"Training on project: {project_path.name}")
            
            # Train on this project
            result = self.train_on_project(project_path, max_iterations)
            result["project_name"] = project_path.name
            
            results.append(result)
            
            # Log training result
            self._log_training_result(result)
        
        return results
    
    def _log_training_result(self, result: Dict[str, Any]):
        """Log training result to RedDB."""
        action = create_action_record(
            action_type=ActionType.CODE_ANALYSIS,
            state_before={"project": result["project_name"], "training_start": True},
            state_after={"project": result["project_name"], "training_complete": True},
            parameters={"max_iterations": 3},
            result=result,
            reward=result["score"],  # Use score as reward
            confidence=0.8 if result["success"] else 0.4,
            execution_time=0.0,
            environment=Environment.DEVELOPMENT
        )
        
        self.data_manager.record_action(action)
        
        # Store in training history
        self.training_history.append({
            "timestamp": time.time(),
            "project": result["project_name"],
            "success": result["success"],
            "score": result["score"],
            "tasks_completed": result.get("tasks_completed", 0)
        })
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        if not self.training_history:
            return {"message": "No training history available"}
        
        total_sessions = len(self.training_history)
        successful_sessions = sum(1 for h in self.training_history if h["success"])
        avg_score = sum(h["score"] for h in self.training_history) / total_sessions
        total_tasks = sum(h["tasks_completed"] for h in self.training_history)
        
        return {
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "success_rate": successful_sessions / total_sessions,
            "average_score": avg_score,
            "total_tasks_completed": total_tasks,
            "projects_trained": len(set(h["project"] for h in self.training_history))
        }

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train real agent with DSPy")
    parser.add_argument("--project", help="Specific project to train on")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations per task")
    parser.add_argument("--workspace-path", help="Path to agent workspace (default: ~/.blampert_workspace)")
    
    args = parser.parse_args()
    
    # Set up trainer
    workspace_path = Path(args.workspace_path) if args.workspace_path else None
    trainer = RealAgentTrainer(workspace_path)
    
    if not trainer.workspace_path.exists():
        logger.error(f"Agent workspace not found: {trainer.workspace_path}")
        logger.error("Run 'python scripts/agent_workspace_manager.py' to create the workspace first")
        sys.exit(1)
    
    # Run training
    logger.info("Starting real agent training...")
    results = trainer.run_training_session(
        project_name=args.project,
        max_iterations=args.iterations
    )
    
    # Print results
    logger.info("Training completed!")
    summary = trainer.get_training_summary()
    
    print("\n" + "="*60)
    print("REAL AGENT TRAINING SUMMARY")
    print("="*60)
    print(f"Total Sessions: {summary['total_sessions']}")
    print(f"Successful Sessions: {summary['successful_sessions']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Average Score: {summary['average_score']:.2f}")
    print(f"Total Tasks Completed: {summary['total_tasks_completed']}")
    print(f"Projects Trained: {summary['projects_trained']}")
    print("="*60)
    
    # Save results
    results_file = Path("real_training_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "summary": summary,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
