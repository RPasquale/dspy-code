#!/usr/bin/env python3
"""
Demo: Isolated Agent Workspace

This script demonstrates how the agent works in its own isolated environment:
- Creates its own workspace
- Has its own virtual environment
- Creates its own projects
- Uses its own streaming engine
- Has its own database
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("🤖 Blampert Agent - Isolated Workspace Demo")
    print("=" * 50)
    
    # Get the project root
    project_root = Path(__file__).parent.parent
    
    print("\n1️⃣ Creating isolated agent workspace...")
    try:
        result = subprocess.run([
            sys.executable, str(project_root / "scripts" / "agent_workspace_manager.py")
        ], capture_output=True, text=True, check=True)
        print("✅ Workspace created successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create workspace: {e}")
        print(e.stderr)
        return
    
    print("\n2️⃣ Showing workspace information...")
    try:
        result = subprocess.run([
            sys.executable, str(project_root / "scripts" / "agent_workspace_manager.py"), "--info"
        ], capture_output=True, text=True, check=True)
        print("📁 Workspace Information:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get workspace info: {e}")
        return
    
    print("\n3️⃣ Training the agent on a project...")
    print("   (This will use the real DSPy agent to write code)")
    try:
        result = subprocess.run([
            sys.executable, str(project_root / "scripts" / "real_agent_trainer.py"),
            "--project", "calculator",
            "--iterations", "2"
        ], capture_output=True, text=True, check=True)
        print("✅ Agent training completed!")
        print("📊 Training Results:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        print(e.stderr)
        return
    
    print("\n🎉 Demo completed!")
    print("\nKey Benefits of Isolated Workspace:")
    print("  • Agent has its own virtual environment")
    print("  • Agent creates its own projects")
    print("  • Agent has its own streaming engine")
    print("  • Agent has its own database")
    print("  • No pollution of the main repository")
    print("  • Complete isolation and independence")
    
    print(f"\n📍 Agent workspace location: ~/.blampert_workspace")
    print("🔧 To activate agent environment:")
    print("   source ~/.blampert_workspace/venv/bin/activate")

if __name__ == "__main__":
    main()
