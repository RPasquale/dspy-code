#!/usr/bin/env python3
"""
Simple Docker Container Selector

A lightweight tool to help users select which Docker containers
the DSPy agent should monitor for logs and streaming.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

def get_containers() -> List[Dict]:
    """Get all Docker containers with basic info."""
    try:
        result = subprocess.run([
            'docker', 'ps', '-a', '--format', 'json'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print("Error: Could not connect to Docker")
            return []
            
        containers = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    container = json.loads(line)
                    containers.append(container)
                except json.JSONDecodeError:
                    continue
        return containers
    except Exception as e:
        print(f"Error: {e}")
        return []

def classify_container(container: Dict) -> str:
    """Classify container type."""
    name = container.get('Names', '').lower()
    image = container.get('Image', '').lower()
    ports = container.get('Ports', '').lower()
    
    # Frontend
    if any(word in name or word in image for word in ['frontend', 'front', 'web', 'ui', 'client', 'react', 'vue', 'angular']):
        return 'frontend'
    
    # Backend
    if any(word in name or word in image for word in ['backend', 'back', 'api', 'server', 'app', 'django', 'flask', 'express']):
        return 'backend'
    
    # Database
    if any(word in name or word in image for word in ['mysql', 'postgres', 'mongodb', 'redis']):
        return 'database'
    
    # Port-based
    if any(port in ports for port in ['3000', '5173', '8080']):
        return 'frontend'
    elif any(port in ports for port in ['5000', '8000', '9000']):
        return 'backend'
    
    return 'unknown'

def display_containers(containers: List[Dict]):
    """Display containers in a simple format."""
    print(f"\n{'#':<3} {'Name':<25} {'Image':<30} {'Status':<15} {'Type':<10}")
    print("-" * 85)
    
    for i, container in enumerate(containers, 1):
        name = container.get('Names', '')[:24]
        image = container.get('Image', '')[:29]
        status = container.get('Status', '')[:14]
        container_type = classify_container(container)
        
        status_color = "🟢" if "Up" in status else "🔴"
        print(f"{i:<3} {name:<25} {image:<30} {status_color} {status:<12} {container_type:<10}")

def select_containers(containers: List[Dict]) -> List[str]:
    """Interactive container selection."""
    if not containers:
        print("No containers found!")
        return []
    
    display_containers(containers)
    
    print(f"\nSelect containers to monitor (1-{len(containers)}, comma-separated, or 'all' for all running):")
    print("Examples: '1,3,5' or 'all' or 'frontend,backend'")
    
    while True:
        selection = input("Selection: ").strip()
        
        if selection.lower() == 'all':
            # Select all running containers
            return [c['Names'] for c in containers if "Up" in c.get('Status', '')]
        
        if selection.lower() in ['frontend', 'backend', 'database']:
            # Select by type
            return [c['Names'] for c in containers if classify_container(c) == selection.lower()]
        
        try:
            # Parse comma-separated numbers
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected = []
            for idx in indices:
                if 0 <= idx < len(containers):
                    selected.append(containers[idx]['Names'])
                else:
                    print(f"Invalid index: {idx + 1}")
            return selected
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
            continue

def save_config(selected_containers: List[str], workspace: Path):
    """Save selected containers to config file."""
    config = {
        "selected_containers": selected_containers,
        "workspace": str(workspace),
        "timestamp": str(Path.cwd())
    }
    
    config_file = workspace / ".dspy_selected_containers.json"
    with config_file.open('w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to {config_file}")

def generate_env_var(selected_containers: List[str]) -> str:
    """Generate environment variable for selected containers."""
    return f"DSPY_DOCKER_CONTAINERS={','.join(selected_containers)}"

def main():
    """Main function."""
    print("🐳 DSPy Container Selector")
    print("=" * 40)
    
    # Get workspace
    workspace = Path.cwd()
    print(f"Workspace: {workspace}")
    
    # Discover containers
    print("\nDiscovering Docker containers...")
    containers = get_containers()
    
    if not containers:
        print("❌ No Docker containers found!")
        return
    
    print(f"✅ Found {len(containers)} containers")
    
    # Select containers
    selected = select_containers(containers)
    
    if not selected:
        print("❌ No containers selected!")
        return
    
    print(f"\n✅ Selected {len(selected)} containers:")
    for container in selected:
        print(f"  • {container}")
    
    # Save configuration
    save_config(selected, workspace)
    
    # Generate environment variable
    env_var = generate_env_var(selected)
    print(f"\n🔧 Environment variable:")
    print(f"export {env_var}")
    print(f"\n🚀 Start the agent with:")
    print(f"export {env_var} && dspy-agent up")

def list_containers():
    """List all containers without selection."""
    print("🐳 Available Docker Containers")
    print("=" * 40)
    
    containers = get_containers()
    if not containers:
        print("❌ No Docker containers found!")
        return
    
    display_containers(containers)
    
    # Show classification summary
    types = {}
    for container in containers:
        container_type = classify_container(container)
        types[container_type] = types.get(container_type, 0) + 1
    
    print(f"\n📊 Container Summary:")
    for container_type, count in types.items():
        print(f"  {container_type}: {count} containers")
    
    # Show quick selection examples
    print(f"\n💡 Quick Selection Examples:")
    print(f"  • All running: 'all'")
    print(f"  • By type: 'frontend', 'backend', 'database'")
    print(f"  • By number: '1,3,5' (comma-separated)")
    print(f"  • Run: python select_containers.py")

if __name__ == "__main__":
    main()
