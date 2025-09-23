#!/usr/bin/env python3
"""
Start Real-Time Monitoring System

This script starts both the WebSocket server for real-time agent monitoring
and the enhanced dashboard server with the React frontend.
"""

import asyncio
import subprocess
import sys
import time
import signal
import threading
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class RealtimeMonitoringLauncher:
    """Launcher for the real-time monitoring system"""
    
    def __init__(self):
        self.processes = []
        self.running = False
        
    def start_websocket_server(self):
        """Start the WebSocket server in a separate process"""
        print("🚀 Starting WebSocket server...")
        
        websocket_script = project_root / "scripts" / "realtime_monitoring_server.py"
        
        try:
            process = subprocess.Popen([
                sys.executable, str(websocket_script)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(process)
            print(f"✅ WebSocket server started (PID: {process.pid})")
            return process
            
        except Exception as e:
            print(f"❌ Failed to start WebSocket server: {e}")
            return None
    
    def start_dashboard_server(self):
        """Start the enhanced dashboard server"""
        print("🚀 Starting enhanced dashboard server...")
        
        dashboard_script = project_root / "enhanced_dashboard_server.py"
        
        try:
            process = subprocess.Popen([
                sys.executable, str(dashboard_script), "8080"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(process)
            print(f"✅ Dashboard server started (PID: {process.pid})")
            return process
            
        except Exception as e:
            print(f"❌ Failed to start dashboard server: {e}")
            return None
    
    def check_processes(self):
        """Check if all processes are still running"""
        for i, process in enumerate(self.processes):
            if process.poll() is not None:
                print(f"⚠️  Process {i} (PID: {process.pid}) has stopped")
                return False
        return True
    
    def stop_all_processes(self):
        """Stop all running processes"""
        print("\n🛑 Stopping all processes...")
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Process {process.pid} stopped")
            except subprocess.TimeoutExpired:
                print(f"⚠️  Process {process.pid} didn't stop gracefully, killing...")
                process.kill()
            except Exception as e:
                print(f"❌ Error stopping process {process.pid}: {e}")
        
        self.processes.clear()
        self.running = False
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n📡 Received signal {signum}, shutting down...")
        self.stop_all_processes()
        sys.exit(0)
    
    def start_monitoring(self):
        """Start the complete monitoring system"""
        print("🎯 Starting Real-Time Agent Monitoring System")
        print("=" * 60)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Start WebSocket server
        websocket_process = self.start_websocket_server()
        if not websocket_process:
            print("❌ Failed to start WebSocket server. Exiting.")
            return False
        
        # Wait a moment for WebSocket server to start
        time.sleep(2)
        
        # Start dashboard server
        dashboard_process = self.start_dashboard_server()
        if not dashboard_process:
            print("❌ Failed to start dashboard server. Exiting.")
            self.stop_all_processes()
            return False
        
        # Wait a moment for dashboard server to start
        time.sleep(2)
        
        self.running = True
        
        print("\n" + "=" * 60)
        print("🎉 Real-Time Monitoring System Started Successfully!")
        print("=" * 60)
        print()
        print("📊 Dashboard URLs:")
        print("   • Main Dashboard: http://localhost:8080/")
        print("   • Real-Time Monitoring: http://localhost:8080/realtime")
        print("   • System Overview: http://localhost:8080/")
        print("   • Advanced Learning: http://localhost:8080/advanced")
        print("   • System Map: http://localhost:8080/system")
        print()
        print("🔌 WebSocket Connection:")
        print("   • Real-Time Data: ws://localhost:8081")
        print()
        print("📡 API Endpoints:")
        print("   • Agent Status: http://localhost:8080/api/status")
        print("   • Learning Metrics: http://localhost:8080/api/learning-metrics")
        print("   • RL Metrics: http://localhost:8080/api/rl-metrics")
        print("   • System Health: http://localhost:8080/api/system-topology")
        print()
        print("🔄 Monitoring Features:")
        print("   • Real-time agent actions and thoughts")
        print("   • Live learning progress and metrics")
        print("   • System performance monitoring")
        print("   • RL training visualization")
        print("   • WebSocket-based live updates")
        print()
        print("Press Ctrl+C to stop all services")
        print("=" * 60)
        
        # Monitor processes
        try:
            while self.running:
                if not self.check_processes():
                    print("❌ One or more processes have stopped. Shutting down...")
                    break
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print("\n📡 Received interrupt signal...")
        
        finally:
            self.stop_all_processes()
        
        return True


def main():
    """Main function"""
    launcher = RealtimeMonitoringLauncher()
    
    try:
        success = launcher.start_monitoring()
        if success:
            print("✅ Monitoring system stopped gracefully")
        else:
            print("❌ Monitoring system failed to start or stopped unexpectedly")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        launcher.stop_all_processes()
        sys.exit(1)


if __name__ == "__main__":
    main()
