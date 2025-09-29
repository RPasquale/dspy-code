#!/usr/bin/env python3
"""
Quick test to verify the context provider fix works.
"""

import subprocess
import sys

def test_context_fix():
    """Test that the context provider fix resolves the dimension mismatch"""
    print("🧪 Testing context provider fix...")
    
    cmd = [
        sys.executable, "-m", "dspy_agent.cli", "train-easy",
        "--workspace", ".",
        "--signature", "CodeContextSig",
        "--rl-steps", "5",
        "--episode-len", "2",
        "--multi-policy",
        "--streaming",
        "--n-envs", "1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Context provider fix successful - no dimension mismatch")
            return True
        else:
            print(f"❌ Test failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    success = test_context_fix()
    sys.exit(0 if success else 1)
