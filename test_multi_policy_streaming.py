#!/usr/bin/env python3
"""
Test script for multi-policy streaming training functionality.
This script tests the enhanced CLI commands without requiring full infrastructure.
"""

import subprocess
import sys
import time
from pathlib import Path

def test_basic_train_easy():
    """Test basic train-easy command with multi-policy features"""
    print("🧪 Testing basic train-easy with multi-policy features...")
    
    cmd = [
        sys.executable, "-m", "dspy_agent.cli", "train-easy",
        "--workspace", ".",
        "--signature", "CodeContextSig",
        "--rl-steps", "10",
        "--episode-len", "2",
        "--multi-policy",
        "--n-envs", "1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Basic train-easy test passed")
            return True
        else:
            print(f"❌ Basic train-easy test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Basic train-easy test timed out")
        return False
    except Exception as e:
        print(f"❌ Basic train-easy test error: {e}")
        return False

def test_streaming_train_easy():
    """Test train-easy with streaming integration"""
    print("🧪 Testing train-easy with streaming integration...")
    
    cmd = [
        sys.executable, "-m", "dspy_agent.cli", "train-easy",
        "--workspace", ".",
        "--signature", "CodeContextSig",
        "--rl-steps", "5",
        "--episode-len", "2",
        "--streaming",
        "--multi-policy",
        "--n-envs", "1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Streaming train-easy test passed")
            return True
        else:
            print(f"❌ Streaming train-easy test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Streaming train-easy test timed out")
        return False
    except Exception as e:
        print(f"❌ Streaming train-easy test error: {e}")
        return False

def test_train_streaming():
    """Test advanced train-streaming command"""
    print("🧪 Testing train-streaming command...")
    
    cmd = [
        sys.executable, "-m", "dspy_agent.cli", "train-streaming",
        "--workspace", ".",
        "--signature", "CodeContextSig",
        "--rl-steps", "5",
        "--grpo-steps", "10",
        "--episode-len", "2",
        "--n-envs", "1",
        "--max-iterations", "1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✅ Train-streaming test passed")
            return True
        else:
            print(f"❌ Train-streaming test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Train-streaming test timed out")
        return False
    except Exception as e:
        print(f"❌ Train-streaming test error: {e}")
        return False

def test_cli_help():
    """Test that CLI help shows new commands"""
    print("🧪 Testing CLI help for new commands...")
    
    # Test train-easy help
    cmd1 = [sys.executable, "-m", "dspy_agent.cli", "train-easy", "--help"]
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    
    # Test train-streaming help
    cmd2 = [sys.executable, "-m", "dspy_agent.cli", "train-streaming", "--help"]
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    
    if result1.returncode == 0 and result2.returncode == 0:
        print("✅ CLI help tests passed")
        return True
    else:
        print("❌ CLI help tests failed")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Multi-Policy Streaming Training System")
    print("=" * 50)
    
    tests = [
        ("CLI Help", test_cli_help),
        ("Basic Train-Easy", test_basic_train_easy),
        ("Streaming Train-Easy", test_streaming_train_easy),
        ("Train-Streaming", test_train_streaming),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Multi-policy streaming system is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
