#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, '.')

# Set environment variables
os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = 'dummy'

from dspy_agent.llm import configure_lm
import dspy

print("Testing LLM configuration...")

try:
    lm = configure_lm(provider="ollama", model_name="qwen3:1.7b")
    if lm:
        print("✅ LLM configured successfully")
        
        # Test a simple call
        print("Testing simple LLM call...")
        try:
            response = lm("Hello, are you working?")
            print(f"✅ LLM response: {response}")
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
    else:
        print("❌ LLM configuration returned None")
        
except Exception as e:
    print(f"❌ LLM configuration failed: {e}")
