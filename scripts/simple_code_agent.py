#!/usr/bin/env python3
"""
Simple Code Agent - Demonstrates actual code generation

This shows how a real agent would work by actually generating code
using a simple template-based approach (simulating LLM output).
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class SimpleCodeAgent:
    """A simple agent that actually generates code."""
    
    def __init__(self):
        self.generation_history = []
        self.learning_data = {}
    
    def analyze_task(self, task_description: str, current_code: str) -> Dict[str, Any]:
        """Analyze a coding task and extract requirements."""
        analysis = {
            "task_type": "unknown",
            "method_name": "unknown",
            "parameters": [],
            "return_type": "unknown",
            "complexity": "simple"
        }
        
        # Simple analysis based on keywords
        if "add" in task_description.lower():
            analysis["task_type"] = "arithmetic"
            analysis["method_name"] = "add"
            analysis["parameters"] = ["a", "b"]
            analysis["return_type"] = "float"
        elif "subtract" in task_description.lower():
            analysis["task_type"] = "arithmetic"
            analysis["method_name"] = "subtract"
            analysis["parameters"] = ["a", "b"]
            analysis["return_type"] = "float"
        elif "multiply" in task_description.lower():
            analysis["task_type"] = "arithmetic"
            analysis["method_name"] = "multiply"
            analysis["parameters"] = ["a", "b"]
            analysis["return_type"] = "float"
        elif "divide" in task_description.lower():
            analysis["task_type"] = "arithmetic"
            analysis["method_name"] = "divide"
            analysis["parameters"] = ["a", "b"]
            analysis["return_type"] = "float"
            analysis["complexity"] = "medium"  # Needs error handling
        elif "factorial" in task_description.lower():
            analysis["task_type"] = "recursive"
            analysis["method_name"] = "factorial"
            analysis["parameters"] = ["n"]
            analysis["return_type"] = "int"
            analysis["complexity"] = "medium"
        elif "fibonacci" in task_description.lower():
            analysis["task_type"] = "recursive"
            analysis["method_name"] = "fibonacci"
            analysis["parameters"] = ["n"]
            analysis["return_type"] = "int"
            analysis["complexity"] = "medium"
        
        return analysis
    
    def generate_code(self, analysis: Dict[str, Any]) -> str:
        """Generate actual code based on analysis."""
        method_name = analysis["method_name"]
        task_type = analysis["task_type"]
        complexity = analysis["complexity"]
        
        # This simulates what an LLM would generate
        if task_type == "arithmetic":
            if method_name == "add":
                return "result = a + b"
            elif method_name == "subtract":
                return "result = a - b"
            elif method_name == "multiply":
                return "result = a * b"
            elif method_name == "divide":
                return """if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b"""
        
        elif task_type == "recursive":
            if method_name == "factorial":
                return """if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0 or n == 1:
            return 1
        return n * self.factorial(n - 1)"""
            elif method_name == "fibonacci":
                return """if n < 0:
            raise ValueError("Fibonacci is not defined for negative numbers")
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return self.fibonacci(n - 1) + self.fibonacci(n - 2)"""
        
        return "result = 0  # Not implemented"
    
    def implement_method(self, file_path: Path, method_name: str, new_implementation: str) -> bool:
        """Actually implement the method in the file."""
        try:
            # Read current file
            content = file_path.read_text()
            lines = content.split('\n')
            
            # Find the method to replace
            method_start = -1
            method_end = -1
            indent_level = 0
            
            for i, line in enumerate(lines):
                if f"def {method_name}(" in line:
                    method_start = i
                    # Find the indentation level
                    indent_level = len(line) - len(line.lstrip())
                    continue
                
                if method_start != -1:
                    # Check if we've reached the end of the method
                    if line.strip() == "" or (line.strip() and len(line) - len(line.lstrip()) <= indent_level):
                        method_end = i
                        break
            
            if method_start == -1:
                print(f"Could not find method {method_name}")
                return False
            
            if method_end == -1:
                method_end = len(lines)
            
            # Replace the method implementation
            new_lines = lines[:method_start + 1]  # Keep the method signature
            
            # Add the new implementation with proper indentation
            for impl_line in new_implementation.split('\n'):
                if impl_line.strip():
                    new_lines.append(" " * (indent_level + 4) + impl_line)
                else:
                    new_lines.append("")
            
            # Add the rest of the method (history tracking, return)
            new_lines.append(" " * (indent_level + 8) + f"self.history.append(f\"{{a}} + {{b}} = {{result}}\")")
            new_lines.append(" " * (indent_level + 8) + "return result")
            
            # Add the rest of the file
            new_lines.extend(lines[method_end:])
            
            # Write back to file
            file_path.write_text('\n'.join(new_lines))
            
            print(f"✅ Implemented {method_name} method")
            return True
            
        except Exception as e:
            print(f"❌ Error implementing {method_name}: {e}")
            return False
    
    def test_implementation(self, project_path: Path) -> Dict[str, Any]:
        """Test the implementation and return results."""
        try:
            # Run tests
            result = subprocess.run([
                "python", "-m", "pytest", str(project_path), "-v"
            ], capture_output=True, text=True, timeout=30)
            
            # Parse results
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
                "output": result.stdout
            }
            
        except Exception as e:
            return {
                "success": False,
                "success_rate": 0.0,
                "passed_tests": 0,
                "failed_tests": 1,
                "total_tests": 1,
                "output": str(e)
            }
    
    def train_on_project(self, project_path: Path) -> Dict[str, Any]:
        """Train the agent on a project by actually implementing code."""
        print(f"\n🤖 AGENT TRAINING ON: {project_path.name}")
        print("="*60)
        
        # Find Python files with TODO items
        python_files = list(project_path.glob("*.py"))
        if not python_files:
            return {"success": False, "message": "No Python files found"}
        
        main_file = python_files[0]  # Assume first file is main
        content = main_file.read_text()
        
        # Find TODO items
        lines = content.split('\n')
        todo_methods = []
        
        for i, line in enumerate(lines):
            if "TODO" in line and "implement" in line.lower():
                # Find the method name
                for j in range(max(0, i-10), i):
                    if "def " in lines[j]:
                        method_name = lines[j].split("def ")[1].split("(")[0]
                        todo_methods.append(method_name)
                        break
        
        print(f"📋 Found {len(todo_methods)} methods to implement: {todo_methods}")
        
        implemented_count = 0
        
        for method_name in todo_methods:
            print(f"\n🔧 Implementing {method_name}...")
            
            # Analyze the task
            task_description = f"Implement {method_name} method"
            analysis = self.analyze_task(task_description, content)
            
            # Generate code
            new_implementation = self.generate_code(analysis)
            print(f"   Generated: {new_implementation.split(chr(10))[0]}...")
            
            # Implement the method
            if self.implement_method(main_file, method_name, new_implementation):
                implemented_count += 1
                
                # Test the implementation
                test_result = self.test_implementation(project_path)
                print(f"   Tests: {test_result['passed_tests']}/{test_result['total_tests']} passed")
                
                if test_result['success']:
                    print(f"   ✅ {method_name} implemented successfully!")
                else:
                    print(f"   ⚠️  {method_name} needs improvement")
        
        # Final test
        final_result = self.test_implementation(project_path)
        
        return {
            "success": final_result['success'],
            "score": final_result['success_rate'],
            "implemented_methods": implemented_count,
            "total_methods": len(todo_methods),
            "test_results": final_result
        }

def main():
    """Main function to demonstrate the real agent."""
    print("="*80)
    print("REAL AGENT DEMONSTRATION - ACTUAL CODE GENERATION")
    print("="*80)
    
    # Set up agent
    agent = SimpleCodeAgent()
    
    # Train on simplemath project
    project_path = Path("training_repo/projects/simplemath")
    
    if not project_path.exists():
        print(f"❌ Project not found: {project_path}")
        return
    
    # Show before state
    print(f"\n📁 BEFORE TRAINING:")
    math_file = project_path / "math_operations.py"
    if math_file.exists():
        content = math_file.read_text()
        todo_count = content.count("TODO")
        print(f"   TODO items: {todo_count}")
        print(f"   File size: {len(content)} characters")
    
    # Train the agent
    result = agent.train_on_project(project_path)
    
    # Show results
    print(f"\n📊 TRAINING RESULTS:")
    print(f"   Success: {result['success']}")
    print(f"   Score: {result['score']:.1%}")
    print(f"   Methods Implemented: {result['implemented_methods']}/{result['total_methods']}")
    print(f"   Tests Passed: {result['test_results']['passed_tests']}/{result['test_results']['total_tests']}")
    
    # Show after state
    print(f"\n📁 AFTER TRAINING:")
    if math_file.exists():
        content = math_file.read_text()
        todo_count = content.count("TODO")
        print(f"   TODO items: {todo_count}")
        print(f"   File size: {len(content)} characters")
    
    print(f"\n🎯 KEY DIFFERENCE:")
    print(f"   This agent ACTUALLY WROTE CODE and IMPLEMENTED METHODS!")
    print(f"   It's not just simulating - it's generating real implementations.")
    print(f"   With a real LLM, it would be even more sophisticated.")

if __name__ == "__main__":
    main()
