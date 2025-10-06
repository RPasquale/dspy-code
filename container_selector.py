#!/usr/bin/env python3
"""
Interactive Docker Container Selector

This tool allows users to discover and select which Docker containers
they want the DSPy agent to monitor for logs and streaming.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm, Prompt
from rich.panel import Panel
from rich.text import Text

console = Console()

class ContainerInfo:
    def __init__(self, name: str, image: str, status: str, ports: str, labels: str = ""):
        self.name = name
        self.image = image
        self.status = status
        self.ports = ports
        self.labels = labels
        self.selected = False
        self.classification = "unknown"

    def __str__(self):
        return f"{self.name} ({self.image})"

def discover_containers() -> List[ContainerInfo]:
    """Discover all available Docker containers (running and stopped)."""
    containers = []
    
    try:
        # Get all containers (running and stopped)
        result = subprocess.run([
            'docker', 'ps', '-a', '--format', 'json'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            console.print("[red]Error: Could not connect to Docker. Make sure Docker is running.[/red]")
            return containers
            
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            try:
                container_data = json.loads(line)
                container = ContainerInfo(
                    name=container_data.get('Names', ''),
                    image=container_data.get('Image', ''),
                    status=container_data.get('Status', ''),
                    ports=container_data.get('Ports', ''),
                    labels=container_data.get('Labels', '')
                )
                container.classification = classify_container(container)
                containers.append(container)
            except Exception:
                continue
                
    except Exception as e:
        console.print(f"[red]Error discovering containers: {e}[/red]")
    
    return containers

def classify_container(container: ContainerInfo) -> str:
    """Classify container type based on name, image, and ports."""
    name_lower = container.name.lower()
    image_lower = container.image.lower()
    ports = container.ports.lower()
    
    # Frontend indicators
    frontend_keywords = ['frontend', 'front', 'web', 'ui', 'client', 'react', 'vue', 'angular', 'nginx']
    if any(keyword in name_lower or keyword in image_lower for keyword in frontend_keywords):
        return 'frontend'
    
    # Backend indicators
    backend_keywords = ['backend', 'back', 'api', 'server', 'app', 'django', 'flask', 'express', 'fastapi', 'node']
    if any(keyword in name_lower or keyword in image_lower for keyword in backend_keywords):
        return 'backend'
    
    # Database indicators
    db_keywords = ['mysql', 'postgres', 'mongodb', 'redis', 'elasticsearch', 'influxdb']
    if any(keyword in name_lower or keyword in image_lower for keyword in db_keywords):
        return 'database'
    
    # Port-based classification
    if '3000' in ports or '5173' in ports or '8080' in ports:
        return 'frontend'
    elif '5000' in ports or '8000' in ports or '9000' in ports:
        return 'backend'
    elif '5432' in ports or '3306' in ports or '6379' in ports:
        return 'database'
    
    return 'unknown'

def display_containers(containers: List[ContainerInfo]) -> None:
    """Display containers in a nice table format."""
    table = Table(title="Available Docker Containers")
    table.add_column("Select", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Image", style="blue")
    table.add_column("Status", style="yellow")
    table.add_column("Ports", style="magenta")
    table.add_column("Type", style="red")
    
    for i, container in enumerate(containers, 1):
        status_style = "green" if "Up" in container.status else "red"
        select_mark = "✓" if container.selected else "○"
        select_style = "green" if container.selected else "white"
        
        table.add_row(
            f"[{select_style}]{select_mark}[/{select_style}]",
            container.name,
            container.image,
            f"[{status_style}]{container.status}[/{status_style}]",
            container.ports,
            container.classification
        )
    
    console.print(table)

def interactive_selection(containers: List[ContainerInfo]) -> List[ContainerInfo]:
    """Interactive container selection interface."""
    console.print(Panel.fit(
        "Select containers to monitor for logs and streaming.\n"
        "Use arrow keys to navigate, space to select/deselect, Enter to confirm.",
        title="Container Selection",
        border_style="blue"
    ))
    
    # Display containers
    display_containers(containers)
    
    # Selection options
    console.print("\n[yellow]Selection Options:[/yellow]")
    console.print("1. [cyan]Select All[/cyan] - Select all containers")
    console.print("2. [cyan]Select Running[/cyan] - Select only running containers")
    console.print("3. [cyan]Select by Type[/cyan] - Select by container type")
    console.print("4. [cyan]Manual Selection[/cyan] - Select containers individually")
    console.print("5. [cyan]Show Details[/cyan] - Show detailed container information")
    console.print("6. [cyan]Done[/cyan] - Finish selection")
    
    while True:
        choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3", "4", "5", "6"], default="6")
        
        if choice == "1":  # Select All
            for container in containers:
                container.selected = True
            console.print("[green]All containers selected![/green]")
            display_containers(containers)
            
        elif choice == "2":  # Select Running
            for container in containers:
                container.selected = "Up" in container.status
            console.print("[green]Running containers selected![/green]")
            display_containers(containers)
            
        elif choice == "3":  # Select by Type
            types = list(set(c.classification for c in containers))
            console.print(f"\nAvailable types: {', '.join(types)}")
            selected_type = Prompt.ask("Enter type to select", choices=types)
            
            for container in containers:
                container.selected = container.classification == selected_type
            console.print(f"[green]Containers of type '{selected_type}' selected![/green]")
            display_containers(containers)
            
        elif choice == "4":  # Manual Selection
            container_names = [c.name for c in containers]
            while True:
                name = Prompt.ask("Enter container name (or 'done')", default="done")
                if name == "done":
                    break
                if name in container_names:
                    container = next(c for c in containers if c.name == name)
                    container.selected = not container.selected
                    status = "selected" if container.selected else "deselected"
                    console.print(f"[green]Container '{name}' {status}![/green]")
                else:
                    console.print(f"[red]Container '{name}' not found![/red]")
            display_containers(containers)
            
        elif choice == "5":  # Show Details
            name = Prompt.ask("Enter container name for details")
            container = next((c for c in containers if c.name == name), None)
            if container:
                details = f"""
Name: {container.name}
Image: {container.image}
Status: {container.status}
Ports: {container.ports}
Labels: {container.labels}
Classification: {container.classification}
                """
                console.print(Panel(details, title=f"Container Details: {name}"))
            else:
                console.print(f"[red]Container '{name}' not found![/red]")
                
        elif choice == "6":  # Done
            break
    
    selected_containers = [c for c in containers if c.selected]
    return selected_containers

def save_selection(containers: List[ContainerInfo], workspace: Path) -> None:
    """Save selected containers to configuration."""
    config = {
        "selected_containers": [
            {
                "name": c.name,
                "image": c.image,
                "classification": c.classification,
                "ports": c.ports
            }
            for c in containers
        ],
        "timestamp": str(Path().cwd()),
        "workspace": str(workspace)
    }
    
    config_file = workspace / ".dspy_container_selection.json"
    with config_file.open('w') as f:
        json.dump(config, f, indent=2)
    
    console.print(f"[green]Container selection saved to {config_file}[/green]")

def load_selection(workspace: Path) -> Optional[Dict]:
    """Load previously saved container selection."""
    config_file = workspace / ".dspy_container_selection.json"
    if config_file.exists():
        try:
            with config_file.open() as f:
                return json.load(f)
        except Exception:
            pass
    return None

def main(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', help="Workspace directory"),
    interactive: bool = typer.Option(True, '--interactive/--no-interactive', help="Interactive mode"),
    save: bool = typer.Option(True, '--save/--no-save', help="Save selection to config file")
):
    """Interactive Docker container selector for DSPy agent monitoring."""
    
    console.print(Panel.fit(
        "DSPy Container Selector\n"
        "Discover and select Docker containers for log monitoring",
        title="🐳 Docker Container Selector",
        border_style="blue"
    ))
    
    # Discover containers
    console.print("[yellow]Discovering Docker containers...[/yellow]")
    containers = discover_containers()
    
    if not containers:
        console.print("[red]No Docker containers found![/red]")
        return
    
    console.print(f"[green]Found {len(containers)} containers[/green]")
    
    # Check for existing selection
    existing_config = load_selection(workspace)
    if existing_config and interactive:
        if Confirm.ask("Load previous container selection?"):
            # Apply previous selection
            selected_names = {c['name'] for c in existing_config.get('selected_containers', [])}
            for container in containers:
                container.selected = container.name in selected_names
            console.print("[green]Previous selection loaded![/green]")
    
    if interactive:
        # Interactive selection
        selected_containers = interactive_selection(containers)
    else:
        # Non-interactive: select all running containers
        selected_containers = [c for c in containers if "Up" in c.status]
        console.print("[yellow]Non-interactive mode: selecting all running containers[/yellow]")
    
    if not selected_containers:
        console.print("[yellow]No containers selected![/yellow]")
        return
    
    # Display final selection
    console.print(f"\n[green]Selected {len(selected_containers)} containers:[/green]")
    for container in selected_containers:
        console.print(f"  • {container.name} ({container.classification})")
    
    if save:
        save_selection(selected_containers, workspace)
    
    # Generate environment variable
    container_names = [c.name for c in selected_containers]
    env_var = f"DSPY_DOCKER_CONTAINERS={','.join(container_names)}"
    console.print(f"\n[blue]Environment variable:[/blue]")
    console.print(f"[cyan]{env_var}[/cyan]")
    console.print("\n[yellow]You can use this environment variable to start the agent:[/yellow]")
    console.print(f"[cyan]export {env_var} && dspy-agent up[/cyan]")

if __name__ == "__main__":
    typer.run(main)
