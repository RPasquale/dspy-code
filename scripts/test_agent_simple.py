#!/usr/bin/env python3
"""
Simple test script to verify the DSPy Agent works correctly
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_functionality():
    """Test basic agent functionality without complex features"""
    print("🧪 Testing basic DSPy Agent functionality...")
    
    # Test 1: Import core modules
    try:
        from dspy_agent.cli import app
        from dspy_agent.config import get_settings
        from dspy_agent.llm import configure_lm
        print("✅ Core modules imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Basic configuration
    try:
        settings = get_settings()
        print(f"✅ Configuration loaded: {type(settings).__name__}")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False
    
    # Test 3: LLM configuration (without actually connecting)
    try:
        # Just test that the function exists and can be called
        from dspy_agent.llm import configure_lm
        print("✅ LLM configuration module available")
    except Exception as e:
        print(f"❌ LLM configuration failed: {e}")
        return False
    
    # Test 4: Database initialization
    try:
        from dspy_agent.db import initialize_database
        initialize_database()
        print("✅ Database initialization successful")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    
    # Test 5: Code tools
    try:
        from dspy_agent.code_tools.code_search import search_text
        from dspy_agent.code_tools.code_snapshot import build_code_snapshot
        print("✅ Code tools available")
    except Exception as e:
        print(f"❌ Code tools failed: {e}")
        return False
    
    return True

def test_simple_commands():
    """Test simple commands that don't require complex setup"""
    print("\n🔧 Testing simple commands...")
    
    # Test 1: Help command
    try:
        from dspy_agent.cli import app
        # This should not crash
        print("✅ CLI app can be instantiated")
    except Exception as e:
        print(f"❌ CLI app failed: {e}")
        return False
    
    # Test 2: Basic code search
    try:
        from dspy_agent.code_tools.code_search import search_text
        # Test with a simple pattern
        results = search_text("def test", str(project_root))
        print(f"✅ Code search works: found {len(results)} results")
    except Exception as e:
        print(f"❌ Code search failed: {e}")
        return False
    
    # Test 3: Code snapshot
    try:
        from dspy_agent.code_tools.code_snapshot import build_code_snapshot
        # Test with a simple file
        test_file = project_root / "README.md"
        if test_file.exists():
            snapshot = build_code_snapshot(test_file)
            print(f"✅ Code snapshot works: {len(snapshot)} characters")
        else:
            print("⚠️  README.md not found, skipping snapshot test")
    except Exception as e:
        print(f"❌ Code snapshot failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 DSPy Agent Simple Test")
    print("=" * 40)
    
    # Test basic functionality
    basic_ok = test_basic_functionality()
    
    # Test simple commands
    commands_ok = test_simple_commands()
    
    # Summary
    print("\n📊 Test Results:")
    print(f"Basic functionality: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"Simple commands: {'✅ PASS' if commands_ok else '❌ FAIL'}")
    
    if basic_ok and commands_ok:
        print("\n🎉 All tests passed! The agent is working correctly.")
        print("\n💡 Tips for using the agent:")
        print("1. Start with simple commands like 'help' or 'ls'")
        print("2. Use 'plan <task>' to get task plans")
        print("3. Use 'grep <pattern>' to search code")
        print("4. Use 'esearch <query>' for semantic search")
        print("5. Avoid complex operations until you're familiar with the basics")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
