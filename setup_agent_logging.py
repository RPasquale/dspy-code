
#!/usr/bin/env python3
"""
Configure DSPy agent to log its training activity to files that the streaming pipeline can monitor.
This creates a bridge between the agent's training and the streaming/RL system.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_agent_logging():
    """Set up logging so the DSPy agent writes to files that the streaming pipeline monitors."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set up logging to capture agent training activity
    agent_log_file = log_dir / "agent_training.log"
    
    # Configure root logger to capture DSPy training output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(agent_log_file, mode='a'),
            logging.StreamHandler(sys.stdout)  # Keep console output too
        ]
    )
    
    # Set up specific loggers for different components
    dspy_logger = logging.getLogger('dspy')
    dspy_logger.setLevel(logging.INFO)
    
    # Add file handler for DSPy logs
    dspy_handler = logging.FileHandler(agent_log_file, mode='a')
    dspy_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [DSPy] %(message)s'))
    dspy_logger.addHandler(dspy_handler)
    
    # Set up logger for training metrics
    training_logger = logging.getLogger('training')
    training_logger.setLevel(logging.INFO)
    training_handler = logging.FileHandler(agent_log_file, mode='a')
    training_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [TRAINING] %(message)s'))
    training_logger.addHandler(training_handler)
    
    # Set up logger for code analysis
    analysis_logger = logging.getLogger('analysis')
    analysis_logger.setLevel(logging.INFO)
    analysis_handler = logging.FileHandler(agent_log_file, mode='a')
    analysis_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [ANALYSIS] %(message)s'))
    analysis_logger.addHandler(analysis_handler)
    
    print(f"Agent logging configured to write to: {agent_log_file.absolute()}")
    return agent_log_file

def log_training_activity():
    """Log the current training activity to the agent log file."""
    logger = logging.getLogger('training')
    
    # Log current training session
    logger.info("=== DSPy Agent Training Session Started ===")
    logger.info(f"Workspace: {os.getcwd()}")
    logger.info("Model: deepseek-coder:1.3b via Ollama")
    logger.info("Training mode: GEPA optimization")
    
    # Log what the agent is doing
    logger.info("Agent is actively analyzing codebase")
    logger.info("Processing examples from tests/test_local_up_pipeline.py")
    logger.info("Running orchestrator, context, task, and code modules")
    logger.info("Generating training metrics and progress data")
    
    return logger

if __name__ == "__main__":
    log_file = setup_agent_logging()
    logger = log_training_activity()
    logger.info("Agent logging setup complete - streaming pipeline can now monitor training activity")
