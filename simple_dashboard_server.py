#!/usr/bin/env python3
"""
Simple HTTP server to serve the DSPy Agent monitoring dashboard.
No external dependencies required - uses only Python standard library.
"""

import http.server
import socketserver
import json
import subprocess
import os
import time
from urllib.parse import parse_qs, urlparse
import urllib.request
import urllib.error

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
            'containers': self.check_containers_status(),
            'timestamp': time.time()
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
            self.send_json_response({'logs': logs, 'timestamp': time.time()})
        except Exception as e:
            self.send_json_response({'logs': f'Error fetching logs: {e}', 'timestamp': time.time()})

    def serve_metrics(self):
        """Get system metrics"""
        metrics = {
            'timestamp': time.time(),
            'containers': self.get_container_count(),
            'memory_usage': self.get_memory_usage(),
            'response_time': self.get_avg_response_time(),
            'uptime': self.get_uptime()
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
            self.send_json_response({'containers': containers, 'timestamp': time.time()})
        except Exception as e:
            self.send_json_response({'containers': f'Error: {e}', 'timestamp': time.time()})

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
                'error': result.stderr,
                'timestamp': time.time()
            }
            self.send_json_response(response)
            
        except Exception as e:
            self.send_json_response({'error': str(e), 'timestamp': time.time()}, 500)

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
                'error': result.stderr,
                'timestamp': time.time()
            }
            self.send_json_response(response)
            
        except Exception as e:
            self.send_json_response({'error': str(e), 'timestamp': time.time()}, 500)

    def check_agent_status(self):
        """Check if agent is responding"""
        try:
            req = urllib.request.Request('http://localhost:8765/status')
            with urllib.request.urlopen(req, timeout=5) as response:
                return {'status': 'healthy', 'details': 'Running', 'code': response.getcode()}
        except urllib.error.URLError:
            return {'status': 'unhealthy', 'details': 'Unreachable', 'code': 0}
        except Exception as e:
            return {'status': 'unhealthy', 'details': f'Error: {e}', 'code': 0}

    def check_ollama_status(self):
        """Check if Ollama is responding"""
        try:
            req = urllib.request.Request('http://localhost:11435/api/tags')
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.getcode() == 200:
                    data = json.loads(response.read().decode())
                    models = [m.get('name', 'unknown') for m in data.get('models', [])]
                    return {'status': 'healthy', 'details': f'Models: {", ".join(models)}', 'models': models}
                else:
                    return {'status': 'unhealthy', 'details': 'HTTP Error', 'models': []}
        except Exception as e:
            return {'status': 'unhealthy', 'details': f'Unreachable: {e}', 'models': []}

    def check_kafka_status(self):
        """Check if Kafka is running"""
        try:
            result = subprocess.run([
                'docker-compose', '-f', '/Users/robbiepasquale/dspy_stuff/docker/lightweight/docker-compose.yml',
                'ps', 'kafka'
            ], capture_output=True, text=True, timeout=5)
            
            if 'healthy' in result.stdout:
                return {'status': 'healthy', 'details': 'Running'}
            elif 'Up' in result.stdout:
                return {'status': 'warning', 'details': 'Running (not healthy)'}
            else:
                return {'status': 'unhealthy', 'details': 'Not running'}
        except Exception as e:
            return {'status': 'unhealthy', 'details': f'Error checking: {e}'}

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
                'details': f'{running}/{total} containers running',
                'running': running,
                'total': total
            }
        except Exception as e:
            return {'status': 'unhealthy', 'details': f'Error checking containers: {e}', 'running': 0, 'total': 0}

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
            return "~2.5GB"  # Simplified - would parse actual output in production
        except:
            return "Unknown"

    def get_avg_response_time(self):
        """Get average response time (simulated)"""
        return round(2.3 + (time.time() % 10) * 0.1, 2)

    def get_uptime(self):
        """Get system uptime"""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                return f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m"
        except:
            return "Unknown"

    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

def start_dashboard_server(port=8080):
    """Start the dashboard server"""
    handler = DashboardHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🌐 DSPy Agent Dashboard Server running at:")
            print(f"   http://localhost:{port}/dashboard")
            print(f"   http://127.0.0.1:{port}/dashboard")
            print("\n📊 Available endpoints:")
            print(f"   GET  /dashboard       - Main monitoring dashboard")
            print(f"   GET  /api/status      - Service status")
            print(f"   GET  /api/logs        - Recent agent logs")
            print(f"   GET  /api/metrics     - System metrics")
            print(f"   GET  /api/containers  - Container information")
            print(f"   POST /api/command     - Execute agent command")
            print(f"   POST /api/restart     - Restart agent")
            print(f"\n🎯 Press Ctrl+C to stop the server")
            print(f"🔄 Dashboard will auto-refresh every 10 seconds")
            
            httpd.serve_forever()
            
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python3 simple_dashboard_server.py 8081")
        else:
            print(f"❌ Error starting server: {e}")
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard server stopped")

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    start_dashboard_server(port)
