#!/usr/bin/env python3
"""
Agent Workspace Manager

Creates and manages a completely isolated environment for the agent to:
- Create its own projects
- Have its own virtual environment
- Run its own streaming engine
- Use its own database
- Learn and train independently
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class AgentWorkspaceManager:
    """Manages the agent's isolated workspace environment."""
    
    def __init__(self, workspace_path: Path = None):
        if workspace_path is None:
            # Create workspace in a separate location
            self.workspace_root = Path.home() / ".blampert_workspace"
        else:
            self.workspace_root = workspace_path
        
        self.venv_path = self.workspace_root / "venv"
        self.projects_path = self.workspace_root / "projects"
        self.data_path = self.workspace_root / "data"
        self.streaming_path = self.workspace_root / "streaming"
        self.logs_path = self.workspace_root / "logs"
        self.models_path = self.workspace_root / "models"
        
        logger.info(f"Agent workspace: {self.workspace_root}")
    
    def create_workspace(self) -> bool:
        """Create the complete agent workspace."""
        try:
            logger.info("Creating agent workspace...")
            
            # Create directory structure
            for path in [self.workspace_root, self.projects_path, self.data_path, 
                        self.streaming_path, self.logs_path, self.models_path]:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created: {path}")
            
            # Create virtual environment
            self._create_virtual_environment()
            
            # Set up agent configuration
            self._setup_agent_config()
            
            # Create initial projects
            self._create_initial_projects()
            
            # Set up streaming engine
            self._setup_streaming_engine()
            
            logger.info("Agent workspace created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create workspace: {e}")
            return False
    
    def _create_virtual_environment(self):
        """Create isolated virtual environment for the agent."""
        logger.info("Creating agent virtual environment...")
        
        # Create venv
        subprocess.run([
            sys.executable, "-m", "venv", str(self.venv_path)
        ], check=True)
        
        # Get pip path
        if os.name == 'nt':  # Windows
            pip_path = self.venv_path / "Scripts" / "pip"
            python_path = self.venv_path / "Scripts" / "python"
        else:  # Unix/Linux/macOS
            pip_path = self.venv_path / "bin" / "pip"
            python_path = self.venv_path / "bin" / "python"
        
        # Install agent dependencies
        dependencies = [
            "dspy-ai>=2.4.9",
            "pytest>=8.3.5",
            "rich>=13.7.1",
            "pydantic>=2.7.4",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "requests>=2.31.0",
            "websockets>=12.0",
            "redis>=5.0.0",  # For agent's own database
            "sqlalchemy>=2.0.0",
            "alembic>=1.12.0"
        ]
        
        for dep in dependencies:
            subprocess.run([
                str(pip_path), "install", dep
            ], check=True)
        
        logger.info("Virtual environment created and dependencies installed")
    
    def _setup_agent_config(self):
        """Set up agent configuration files."""
        config = {
            "workspace": {
                "root": str(self.workspace_root),
                "projects": str(self.projects_path),
                "data": str(self.data_path),
                "streaming": str(self.streaming_path),
                "logs": str(self.logs_path),
                "models": str(self.models_path)
            },
            "agent": {
                "name": "BlampertAgent",
                "version": "1.0.0",
                "learning_rate": 0.001,
                "max_iterations": 5,
                "model": "deepseek-coder:1.3b",
                "ollama_url": "http://localhost:11435"
            },
            "streaming": {
                "enabled": True,
                "kafka_bootstrap_servers": ["localhost:9092"],
                "topics": {
                    "agent_actions": "agent.actions",
                    "learning_events": "agent.learning", 
                    "code_generation": "agent.code",
                    "test_results": "agent.tests"
                }
            },
            "database": {
                "type": "reddb",
                "url": "http://localhost:8080",
                "namespace": "blampert_agent"
            }
        }
        
        config_file = self.workspace_root / "agent_config.json"
        import json
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Agent configuration saved: {config_file}")
    
    def _create_initial_projects(self):
        """Create initial training projects for the agent."""
        logger.info("Creating initial training projects...")
        
        # Project 1: Simple Calculator
        self._create_calculator_project()
        
        # Project 2: Math Operations
        self._create_math_operations_project()
        
        # Project 3: Data Processor
        self._create_data_processor_project()
        
        logger.info("Initial projects created")
    
    def _create_calculator_project(self):
        """Create calculator training project."""
        project_path = self.projects_path / "calculator"
        project_path.mkdir(exist_ok=True)
        
        # Create calculator.py with TODO items
        calculator_code = '''"""
Simple Calculator - Agent Training Project

The agent needs to implement the missing methods.
"""

class Calculator:
    """A simple calculator for agent training."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """
        Add two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b
        """
        # TODO: Implement addition
        result = 0  # Replace with actual implementation
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """
        Subtract second number from first number.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Difference of a and b
        """
        # TODO: Implement subtraction
        result = 0  # Replace with actual implementation
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """
        Multiply two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product of a and b
        """
        # TODO: Implement multiplication
        result = 0  # Replace with actual implementation
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """
        Divide first number by second number.
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            Quotient of a and b
            
        Raises:
            ValueError: If divisor is zero
        """
        # TODO: Implement division with zero check
        if b == 0:
            raise ValueError("Cannot divide by zero")
        
        result = 0  # Replace with actual implementation
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self) -> list:
        """Get calculation history."""
        return self.history.copy()
    
    def clear_history(self):
        """Clear calculation history."""
        self.history.clear()
'''
        
        (project_path / "calculator.py").write_text(calculator_code)
        
        # Create test file
        test_code = '''"""
Tests for Calculator class.
"""

import pytest
from calculator import Calculator


class TestCalculator:
    """Test cases for Calculator class."""
    
    def setup_method(self):
        """Set up a fresh calculator instance for each test."""
        self.calc = Calculator()
    
    def test_add_positive_numbers(self):
        """Test addition with positive numbers."""
        result = self.calc.add(2, 3)
        assert result == 5
    
    def test_subtract_positive_numbers(self):
        """Test subtraction with positive numbers."""
        result = self.calc.subtract(5, 3)
        assert result == 2
    
    def test_multiply_positive_numbers(self):
        """Test multiplication with positive numbers."""
        result = self.calc.multiply(3, 4)
        assert result == 12
    
    def test_divide_positive_numbers(self):
        """Test division with positive numbers."""
        result = self.calc.divide(10, 2)
        assert result == 5
    
    def test_divide_by_zero(self):
        """Test division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            self.calc.divide(5, 0)
    
    def test_history_tracking(self):
        """Test that calculations are tracked in history."""
        self.calc.add(1, 2)
        self.calc.multiply(3, 4)
        
        history = self.calc.get_history()
        assert len(history) == 2
        assert "1 + 2 = 3" in history
        assert "3 * 4 = 12" in history
'''
        
        (project_path / "test_calculator.py").write_text(test_code)
        
        # Create requirements
        (project_path / "requirements.txt").write_text("pytest>=8.3.5\n")
        
        # Create README
        readme = '''# Calculator Project

A simple calculator implementation for agent training.

## TODO Items

The agent needs to implement:
- `add(a, b)` - Addition
- `subtract(a, b)` - Subtraction  
- `multiply(a, b)` - Multiplication
- `divide(a, b)` - Division with zero check

## Running Tests

```bash
pytest test_calculator.py -v
```

## Success Criteria

- All tests pass
- Code is properly implemented
- Error handling works correctly
'''
        
        (project_path / "README.md").write_text(readme)
    
    def _create_math_operations_project(self):
        """Create math operations training project."""
        project_path = self.projects_path / "math_operations"
        project_path.mkdir(exist_ok=True)
        
        # Create math_operations.py with TODO items
        math_code = '''"""
Math Operations - Agent Training Project

The agent needs to implement advanced mathematical operations.
"""

class MathOperations:
    """Advanced mathematical operations for agent training."""
    
    def __init__(self):
        self.history = []
    
    def factorial(self, n: int) -> int:
        """
        Calculate factorial of a number.
        
        Args:
            n: Non-negative integer
            
        Returns:
            Factorial of n
            
        Raises:
            ValueError: If n is negative
        """
        # TODO: Implement factorial calculation
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        
        result = 0  # Replace with actual implementation
        self.history.append(f"{n}! = {result}")
        return result
    
    def fibonacci(self, n: int) -> int:
        """
        Calculate nth Fibonacci number.
        
        Args:
            n: Non-negative integer
            
        Returns:
            nth Fibonacci number
        """
        # TODO: Implement Fibonacci calculation
        result = 0  # Replace with actual implementation
        self.history.append(f"fib({n}) = {result}")
        return result
    
    def prime_check(self, n: int) -> bool:
        """
        Check if a number is prime.
        
        Args:
            n: Integer to check
            
        Returns:
            True if prime, False otherwise
        """
        # TODO: Implement prime number check
        result = False  # Replace with actual implementation
        self.history.append(f"prime({n}) = {result}")
        return result
    
    def gcd(self, a: int, b: int) -> int:
        """
        Calculate greatest common divisor.
        
        Args:
            a: First integer
            b: Second integer
            
        Returns:
            Greatest common divisor of a and b
        """
        # TODO: Implement GCD calculation
        result = 0  # Replace with actual implementation
        self.history.append(f"gcd({a}, {b}) = {result}")
        return result
    
    def get_history(self) -> list:
        """Get calculation history."""
        return self.history.copy()
'''
        
        (project_path / "math_operations.py").write_text(math_code)
        
        # Create test file
        test_code = '''"""
Tests for MathOperations class.
"""

import pytest
from math_operations import MathOperations


class TestMathOperations:
    """Test cases for MathOperations class."""
    
    def setup_method(self):
        """Set up a fresh MathOperations instance for each test."""
        self.math = MathOperations()
    
    def test_factorial_positive_number(self):
        """Test factorial with positive number."""
        result = self.math.factorial(5)
        assert result == 120
    
    def test_factorial_zero(self):
        """Test factorial of zero."""
        result = self.math.factorial(0)
        assert result == 1
    
    def test_factorial_negative_number(self):
        """Test factorial with negative number raises ValueError."""
        with pytest.raises(ValueError, match="Factorial is not defined for negative numbers"):
            self.math.factorial(-1)
    
    def test_fibonacci_positive_number(self):
        """Test Fibonacci with positive number."""
        result = self.math.fibonacci(6)
        assert result == 8  # 0, 1, 1, 2, 3, 5, 8
    
    def test_fibonacci_zero(self):
        """Test Fibonacci of zero."""
        result = self.math.fibonacci(0)
        assert result == 0
    
    def test_fibonacci_one(self):
        """Test Fibonacci of one."""
        result = self.math.fibonacci(1)
        assert result == 1
    
    def test_prime_check_prime_number(self):
        """Test prime check with prime number."""
        result = self.math.prime_check(17)
        assert result == True
    
    def test_prime_check_composite_number(self):
        """Test prime check with composite number."""
        result = self.math.prime_check(15)
        assert result == False
    
    def test_gcd_positive_numbers(self):
        """Test GCD with positive numbers."""
        result = self.math.gcd(48, 18)
        assert result == 6
    
    def test_gcd_same_numbers(self):
        """Test GCD with same numbers."""
        result = self.math.gcd(7, 7)
        assert result == 7
'''
        
        (project_path / "test_math_operations.py").write_text(test_code)
        
        # Create requirements
        (project_path / "requirements.txt").write_text("pytest>=8.3.5\n")
        
        # Create README
        readme = '''# Math Operations Project

Advanced mathematical operations for agent training.

## TODO Items

The agent needs to implement:
- `factorial(n)` - Factorial calculation
- `fibonacci(n)` - Fibonacci sequence
- `prime_check(n)` - Prime number check
- `gcd(a, b)` - Greatest common divisor

## Running Tests

```bash
pytest test_math_operations.py -v
```

## Success Criteria

- All tests pass
- Code is properly implemented
- Error handling works correctly
'''
        
        (project_path / "README.md").write_text(readme)
    
    def _create_data_processor_project(self):
        """Create data processor training project."""
        project_path = self.projects_path / "data_processor"
        project_path.mkdir(exist_ok=True)
        
        # Create data_processor.py with TODO items
        processor_code = '''"""
Data Processor - Agent Training Project

The agent needs to implement data processing operations.
"""

import pandas as pd
from typing import List, Dict, Any


class DataProcessor:
    """Data processing operations for agent training."""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
    
    def load_csv(self, file_path: str) -> bool:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        # TODO: Implement CSV loading
        try:
            self.data = pd.DataFrame()  # Replace with actual implementation
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def clean_data(self) -> bool:
        """
        Clean the loaded data.
        
        Returns:
            True if cleaning successful, False otherwise
        """
        # TODO: Implement data cleaning
        if self.data is None:
            return False
        
        try:
            # Remove duplicates, handle missing values, etc.
            self.processed_data = self.data.copy()  # Replace with actual implementation
            return True
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return False
    
    def calculate_statistics(self) -> Dict[str, float]:
        """
        Calculate basic statistics for numeric columns.
        
        Returns:
            Dictionary of statistics
        """
        # TODO: Implement statistics calculation
        if self.processed_data is None:
            return {}
        
        try:
            stats = {}  # Replace with actual implementation
            return stats
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {}
    
    def save_processed_data(self, output_path: str) -> bool:
        """
        Save processed data to file.
        
        Args:
            output_path: Output file path
            
        Returns:
            True if save successful, False otherwise
        """
        # TODO: Implement data saving
        if self.processed_data is None:
            return False
        
        try:
            # Save to CSV, JSON, etc.
            return True  # Replace with actual implementation
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
'''
        
        (project_path / "data_processor.py").write_text(processor_code)
        
        # Create test file
        test_code = '''"""
Tests for DataProcessor class.
"""

import pytest
import pandas as pd
import tempfile
import os
from data_processor import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def setup_method(self):
        """Set up a fresh DataProcessor instance for each test."""
        self.processor = DataProcessor()
    
    def test_load_csv_valid_file(self):
        """Test loading a valid CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,age,city\\n")
            f.write("Alice,25,New York\\n")
            f.write("Bob,30,London\\n")
            temp_file = f.name
        
        try:
            result = self.processor.load_csv(temp_file)
            assert result == True
            assert self.processor.data is not None
        finally:
            os.unlink(temp_file)
    
    def test_load_csv_invalid_file(self):
        """Test loading an invalid CSV file."""
        result = self.processor.load_csv("nonexistent.csv")
        assert result == False
    
    def test_clean_data_with_data(self):
        """Test cleaning data when data is loaded."""
        # Set up test data
        self.processor.data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Alice'],
            'age': [25, 30, None],
            'city': ['New York', 'London', 'Paris']
        })
        
        result = self.processor.clean_data()
        assert result == True
        assert self.processor.processed_data is not None
    
    def test_clean_data_without_data(self):
        """Test cleaning data when no data is loaded."""
        result = self.processor.clean_data()
        assert result == False
    
    def test_calculate_statistics_with_data(self):
        """Test calculating statistics with data."""
        # Set up test data
        self.processor.processed_data = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'salary': [50000, 60000, 70000, 80000]
        })
        
        stats = self.processor.calculate_statistics()
        assert isinstance(stats, dict)
    
    def test_calculate_statistics_without_data(self):
        """Test calculating statistics without data."""
        stats = self.processor.calculate_statistics()
        assert stats == {}
    
    def test_save_processed_data_with_data(self):
        """Test saving processed data."""
        # Set up test data
        self.processor.processed_data = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'age': [25, 30]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            result = self.processor.save_processed_data(temp_file)
            assert result == True
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_save_processed_data_without_data(self):
        """Test saving processed data without data."""
        result = self.processor.save_processed_data("output.csv")
        assert result == False
'''
        
        (project_path / "test_data_processor.py").write_text(test_code)
        
        # Create requirements
        (project_path / "requirements.txt").write_text("pytest>=8.3.5\npandas>=2.0.0\n")
        
        # Create README
        readme = '''# Data Processor Project

Data processing operations for agent training.

## TODO Items

The agent needs to implement:
- `load_csv(file_path)` - Load data from CSV
- `clean_data()` - Clean and preprocess data
- `calculate_statistics()` - Calculate basic statistics
- `save_processed_data(output_path)` - Save processed data

## Running Tests

```bash
pytest test_data_processor.py -v
```

## Success Criteria

- All tests pass
- Code is properly implemented
- Error handling works correctly
'''
        
        (project_path / "README.md").write_text(readme)
    
    def _setup_streaming_engine(self):
        """Set up the agent's own streaming engine."""
        logger.info("Setting up agent streaming engine...")
        
        # Create streaming configuration
        streaming_config = {
            "engine": {
                "type": "kafka",
                "bootstrap_servers": ["localhost:9092"],
                "topics": {
                    "agent_actions": "agent.actions",
                    "learning_events": "agent.learning",
                    "code_generation": "agent.code",
                    "test_results": "agent.tests"
                }
            },
            "producer": {
                "batch_size": 1000,
                "linger_ms": 10,
                "compression_type": "gzip"
            },
            "consumer": {
                "group_id": "blampert_agent",
                "auto_offset_reset": "latest",
                "enable_auto_commit": True
            },
            "integration": {
                "use_existing_kafka": True,
                "kafka_url": "localhost:9092",
                "spark_integration": True,
                "spark_url": "localhost:8080"
            }
        }
        
        config_file = self.streaming_path / "streaming_config.json"
        import json
        with open(config_file, "w") as f:
            json.dump(streaming_config, f, indent=2)
        
        # Create streaming engine script
        streaming_script = '''#!/usr/bin/env python3
"""
Agent Streaming Engine

Handles real-time data streaming for the agent's learning process.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AgentStreamingEngine:
    """Streaming engine for agent data."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
    
    def _load_config(self) -> Dict[str, Any]:
        """Load streaming configuration."""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def start(self):
        """Start the streaming engine."""
        logger.info("Starting agent streaming engine...")
        self.running = True
        
        # Initialize Kafka producer/consumer
        # This would connect to actual Kafka in production
        logger.info("Streaming engine started")
    
    def stop(self):
        """Stop the streaming engine."""
        logger.info("Stopping agent streaming engine...")
        self.running = False
        logger.info("Streaming engine stopped")
    
    def publish_action(self, action: Dict[str, Any]):
        """Publish agent action to stream."""
        if self.running:
            logger.info(f"Publishing action: {action.get('type', 'unknown')}")
            # This would publish to Kafka in production
    
    def publish_learning_event(self, event: Dict[str, Any]):
        """Publish learning event to stream."""
        if self.running:
            logger.info(f"Publishing learning event: {event.get('type', 'unknown')}")
            # This would publish to Kafka in production

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config_path = Path(__file__).parent / "streaming_config.json"
    engine = AgentStreamingEngine(config_path)
    
    try:
        engine.start()
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        engine.stop()
'''
        
        (self.streaming_path / "streaming_engine.py").write_text(streaming_script)
        
        logger.info("Streaming engine configured")
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get information about the workspace."""
        return {
            "workspace_root": str(self.workspace_root),
            "virtual_env": str(self.venv_path),
            "projects": str(self.projects_path),
            "data": str(self.data_path),
            "streaming": str(self.streaming_path),
            "logs": str(self.logs_path),
            "models": str(self.models_path),
            "exists": self.workspace_root.exists()
        }
    
    def activate_environment(self) -> str:
        """Get command to activate the virtual environment."""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "activate")
        else:  # Unix/Linux/macOS
            return f"source {self.venv_path / 'bin' / 'activate'}"

def main():
    """Main function to create agent workspace."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create agent workspace")
    parser.add_argument("--path", help="Custom workspace path")
    parser.add_argument("--info", action="store_true", help="Show workspace info")
    
    args = parser.parse_args()
    
    workspace_path = Path(args.path) if args.path else None
    manager = AgentWorkspaceManager(workspace_path)
    
    if args.info:
        info = manager.get_workspace_info()
        print("Agent Workspace Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        if manager.create_workspace():
            print("✅ Agent workspace created successfully!")
            print(f"📍 Location: {manager.workspace_root}")
            print(f"🔧 Activate with: {manager.activate_environment()}")
        else:
            print("❌ Failed to create agent workspace")
            sys.exit(1)

if __name__ == "__main__":
    main()
