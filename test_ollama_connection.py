#!/usr/bin/env python3
"""
Test Ollama connection and DSPy configuration
"""

import dspy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ollama_connection():
    """Test Ollama connection and DSPy configuration."""
    logger.info("Testing Ollama connection...")
    
    try:
        # Use DSPy LM with Ollama configuration
        lm = dspy.LM(
            model="ollama/qwen3:1.7b",
            api_base="http://localhost:11435",
            max_tokens=4000
        )
        logger.info("Successfully created DSPy LM with Ollama")
        
        # Configure DSPy
        dspy.settings.configure(lm=lm)
        logger.info("Successfully configured DSPy with LM")
        
        # Test the LM
        test_prompt = "What is 2 + 2?"
        logger.info(f"Testing with prompt: {test_prompt}")
        
        response = lm(test_prompt)
        logger.info(f"Response: {response}")
        
        # Test DSPy ChainOfThought
        logger.info("Testing DSPy ChainOfThought...")
        cot = dspy.ChainOfThought("question -> answer")
        result = cot(question=test_prompt)
        logger.info(f"ChainOfThought result: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Ollama connection: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama_connection()
    if success:
        print("✅ Ollama connection test passed!")
    else:
        print("❌ Ollama connection test failed!")
