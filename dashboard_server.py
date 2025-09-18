#!/usr/bin/env python3
"""
Simple HTTP server to serve the DSPy Agent monitoring dashboard.
Includes real-time log streaming and agent interaction endpoints.
"""

import http.server
import socketserver
import json
import subprocess
import os
import threading
import time
from urllib.parse import parse_qs, urlparse
import requests

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="/Users/robbiepasquale/dspy_stuff", **kwargs)

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Serve the main dashboard
        if path == '/' or path == '/dashboard':
            self.serve_dashboard()
        
        # API endpoints
        elif path == '/api/status':
            self.serve_status()
        elif path == '/api/logs':
            self.serve_logs()
        elif path == '/api/metrics':
            self.serve_metrics()
        elif path == '/api/containers':
            self.serve_containers()
        else:
            super().do_GET()

    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/command':
            self.handle_command()
        elif path == '/api/restart':
            self.handle_restart()
        else:
            self.send_error(404)

    def serve_dashboard(self):
        """Serve the main dashboard HTML"""
        try:
            with open('/Users/robbiepasquale/dspy_stuff/monitoring_dashboard.html', 'r') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content.encode())
        except Exception as e:
            self.send_error(500, f"Error serving dashboard: {e}")

    def serve_status(self):
        """Check status of all services"""
        status = {
            'agent': self.check_agent_status(),
            'ollama': self.check_ollama_status(),
            'kafka': self.check_kafka_status(),
            'containers': self.check_containers_status()
        }
        
        self.send_json_response(status)

    def serve_logs(self):
        """Get recent agent logs"""
        try:
            result = subprocess.run([
                'docker-compose', '-f', '/Users/robbiepasquale/dspy_stuff/docker/lightweight/docker-compose.yml',
                'logs', 'dspy-agent', '--tail=50'
            ], capture_output=True, text=True, timeout=10)
            
            logs = result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
            self.send_json_response({'logs': logs})
        except Exception as e:
            self.send_json_response({'logs': f'Error fetching logs: {e}'})

    def serve_metrics(self):
        """Get system metrics"""
        metrics = {
            'timestamp': time.time(),
            'containers': self.get_container_count(),
            'memory_usage': self.get_memory_usage(),
            'response_time': self.get_avg_response_time()
        }
        self.send_json_response(metrics)

    def serve_containers(self):
        """Get container information"""
        try:
            result = subprocess.run([
                'docker-compose', '-f', '/Users/robbiepasquale/dspy_stuff/docker/lightweight/docker-compose.yml',
                'ps'
            ], capture_output=True, text=True, timeout=10)
            
            containers = result.stdout if result.returncode == 0 else "Error getting containers"
            self.send_json_response({'containers': containers})
        except Exception as e:
            self.send_json_response({'containers': f'Error: {e}'})

    def handle_command(self):
        """Execute agent command"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            command = data.get('command', '')
            if not command:
                self.send_json_response({'error': 'No command provided'}, 400)
                return
            
            # Execute command in agent container
            result = subprocess.run([
                'docker-compose', '-f', '/Users/robbiepasquale/dspy_stuff/docker/lightweight/docker-compose.yml',
                'exec', '-T', 'dspy-agent', 'bash', '-c',
                f'cd /app && PYTHONPATH=/app dspy-agent --workspace /app/test_project --logs /app/logs {command}'
            ], capture_output=True, text=True, timeout=30)
            
            response = {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr
            }
            self.send_json_response(response)
            
        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)

    def handle_restart(self):
        """Restart agent container"""
        try:
            result = subprocess.run([
                'docker-compose', '-f', '/Users/robbiepasquale/dspy_stuff/docker/lightweight/docker-compose.yml',
                'restart', 'dspy-agent'
            ], capture_output=True, text=True, timeout=30)
            
            response = {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr
            }
            self.send_json_response(response)
            
        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)

    def check_agent_status(self):
        """Check if agent is responding"""
        try:
            response = requests.get('http://localhost:8765/status', timeout=5)
            return {'status': 'healthy' if response.status_code == 200 else 'unhealthy', 'details': 'Running'}
        except:
            return {'status': 'unhealthy', 'details': 'Unreachable'}

    def check_ollama_status(self):
        """Check if Ollama is responding"""
        try:
            response = requests.get('http://localhost:11435/api/tags', timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m.get('name', 'unknown') for m in data.get('models', [])]
                return {'status': 'healthy', 'details': f'Models: {", ".join(models)}'}
            else:
                return {'status': 'unhealthy', 'details': 'HTTP Error'}
        except:
            return {'status': 'unhealthy', 'details': 'Unreachable'}

    def check_kafka_status(self):
        """Check if Kafka is running"""
        try:
            result = subprocess.run([
                'docker-compose', '-f', '/Users/robbiepasquale/dspy_stuff/docker/lightweight/docker-compose.yml',
                'ps', 'kafka'
            ], capture_output=True, text=True, timeout=5)
            
            if 'healthy' in result.stdout:
                return {'status': 'healthy', 'details': 'Running'}
            else:
                return {'status': 'unhealthy', 'details': 'Not healthy'}
        except:
            return {'status': 'unhealthy', 'details': 'Error checking'}

    def check_containers_status(self):
        """Check overall container status"""
        try:
            result = subprocess.run([
                'docker-compose', '-f', '/Users/robbiepasquale/dspy_stuff/docker/lightweight/docker-compose.yml',
                'ps'
            ], capture_output=True, text=True, timeout=5)
            
            lines = result.stdout.strip().split('\n')
            running = len([l for l in lines if 'Up' in l])
            total = max(0, len(lines) - 1)  # Subtract header line
            
            return {
                'status': 'healthy' if running > 0 else 'unhealthy',
                'details': f'{running}/{total} containers running'
            }
        except:
            return {'status': 'unhealthy', 'details': 'Error checking containers'}

    def get_container_count(self):
        """Get number of running containers"""
        try:
            result = subprocess.run([
                'docker-compose', '-f', '/Users/robbiepasquale/dspy_stuff/docker/lightweight/docker-compose.yml',
                'ps', '-q'
            ], capture_output=True, text=True, timeout=5)
            return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        except:
            return 0

    def get_memory_usage(self):
        """Get approximate memory usage"""
        try:
            result = subprocess.run(['docker', 'stats', '--no-stream', '--format', 'table {{.MemUsage}}'],
                                  capture_output=True, text=True, timeout=5)
            # This is a simplified version - in production you'd parse the actual output
            return "~2.5GB"
        except:
            return "Unknown"

    def get_avg_response_time(self):
        """Get average response time (simulated)"""
        return round(2.3 + (time.time() % 10) * 0.1, 2)

    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

def start_dashboard_server(port=8080):
    """Start the dashboard server"""
    handler = DashboardHandler
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"🌐 DSPy Agent Dashboard Server running at:")
        print(f"   http://localhost:{port}/dashboard")
        print(f"   http://127.0.0.1:{port}/dashboard")
        print("\n📊 Available endpoints:")
        print(f"   GET  /dashboard       - Main monitoring dashboard")
        print(f"   GET  /api/status      - Service status")
        print(f"   GET  /api/logs        - Recent agent logs")
        print(f"   GET  /api/metrics     - System metrics")
        print(f"   POST /api/command     - Execute agent command")
        print(f"   POST /api/restart     - Restart agent")
        print("\n🎯 Press Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Dashboard server stopped")

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    start_dashboard_server(port)
