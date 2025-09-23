#!/usr/bin/env python3
"""
Simple API server for DSPy Agent frontend
Provides mock data for testing the frontend
"""

import http.server
import socketserver
import json
from datetime import datetime

class SimpleAPIHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/'):
            self.handle_api_request()
        else:
            self.send_error(404)
    
    def handle_api_request(self):
        if self.path == '/api/status':
            self.serve_json({
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "services": {
                    "agent": "active",
                    "kafka": "connected",
                    "database": "connected"
                }
            })
        elif self.path == '/api/bus-metrics':
            self.serve_json({
                "total_messages": 1234,
                "active_consumers": 3,
                "queue_depth": 42,
                "throughput": 15.6,
                "timestamp": datetime.now().isoformat()
            })
        elif self.path == '/api/metrics':
            self.serve_json({
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.1,
                "network_io": 1024,
                "timestamp": datetime.now().isoformat()
            })
        elif self.path == '/api/learning-metrics':
            self.serve_json({
                "total_episodes": 1000,
                "average_reward": 0.75,
                "success_rate": 0.68,
                "learning_rate": 0.001,
                "timestamp": datetime.now().isoformat()
            })
        elif self.path == '/api/logs':
            self.serve_json({
                "logs": [
                    {
                        "timestamp": "2025-09-21T11:00:00Z",
                        "level": "INFO",
                        "message": "Agent started successfully",
                        "status": "success"
                    },
                    {
                        "timestamp": "2025-09-21T11:05:00Z", 
                        "level": "INFO",
                        "message": "Processing task: code_generation",
                        "status": "processing"
                    },
                    {
                        "timestamp": "2025-09-21T11:10:00Z",
                        "level": "SUCCESS",
                        "message": "Task completed successfully",
                        "status": "completed"
                    },
                    {
                        "timestamp": "2025-09-21T11:15:00Z",
                        "level": "INFO", 
                        "message": "Learning metrics updated",
                        "status": "success"
                    },
                    {
                        "timestamp": "2025-09-21T11:20:00Z",
                        "level": "WARNING",
                        "message": "High memory usage detected",
                        "status": "warning"
                    }
                ],
                "total_logs": 5,
                "timestamp": datetime.now().isoformat()
            })
        elif self.path == '/api/rl-metrics':
            self.serve_json({
                "episodes": 1000,
                "avg_reward": 0.75,
                "average_reward": 0.75,  # Keep both for compatibility
                "success_rate": 0.68,
                "exploration_rate": 0.1,
                "learning_rate": 0.001,
                "recent_rewards": [0.8, 0.7, 0.9, 0.6, 0.85],
                "timestamp": datetime.now().isoformat()
            })
        elif self.path == '/api/signatures':
            self.serve_json({
                "signatures": [
                    {"name": "code_generation", "accuracy": 0.85, "calls": 150},
                    {"name": "bug_fix", "accuracy": 0.78, "calls": 89},
                    {"name": "test_generation", "accuracy": 0.92, "calls": 67}
                ],
                "timestamp": datetime.now().isoformat()
            })
        elif self.path == '/api/verifiers':
            self.serve_json({
                "verifiers": [
                    {"name": "syntax_checker", "accuracy": 0.98, "checks": 500},
                    {"name": "logic_validator", "accuracy": 0.87, "checks": 300},
                    {"name": "performance_checker", "accuracy": 0.82, "checks": 200}
                ],
                "timestamp": datetime.now().isoformat()
            })
        elif self.path == '/api/kafka-topics':
            self.serve_json({
                "topics": [
                    {"name": "agent.results", "partitions": 3, "messages": 1234},
                    {"name": "agent.tasks", "partitions": 2, "messages": 567},
                    {"name": "logs.raw", "partitions": 1, "messages": 890}
                ],
                "total_topics": 3,
                "timestamp": datetime.now().isoformat()
            })
        elif self.path == '/api/spark-workers':
            self.serve_json({
                "workers": [
                    {"id": "worker-1", "status": "active", "cores": 4, "memory": "8GB"},
                    {"id": "worker-2", "status": "active", "cores": 4, "memory": "8GB"},
                    {"id": "worker-3", "status": "idle", "cores": 2, "memory": "4GB"}
                ],
                "total_workers": 3,
                "active_workers": 2,
                "timestamp": datetime.now().isoformat()
            })
        elif self.path == '/api/stream-metrics':
            self.serve_json({
                "throughput": 15.6,
                "latency": 45.2,
                "error_rate": 0.02,
                "active_streams": 5,
                "total_events": 10000,
                "timestamp": datetime.now().isoformat()
            })
        elif self.path.startswith('/api/performance-history'):
            self.serve_json({
                "timeframe": "1h",
                "data": [
                    {"timestamp": "2025-09-21T10:00:00Z", "cpu": 45.2, "memory": 67.8, "throughput": 15.6},
                    {"timestamp": "2025-09-21T10:15:00Z", "cpu": 47.1, "memory": 68.2, "throughput": 16.1},
                    {"timestamp": "2025-09-21T10:30:00Z", "cpu": 43.8, "memory": 66.9, "throughput": 14.8},
                    {"timestamp": "2025-09-21T10:45:00Z", "cpu": 46.5, "memory": 69.1, "throughput": 17.2}
                ],
                "timestamp": datetime.now().isoformat()
            })
        else:
            self.serve_json({
                "error": "Endpoint not found",
                "path": self.path,
                "available_endpoints": [
                    "/api/status",
                    "/api/bus-metrics", 
                    "/api/metrics",
                    "/api/learning-metrics",
                    "/api/logs",
                    "/api/rl-metrics",
                    "/api/signatures",
                    "/api/verifiers",
                    "/api/kafka-topics",
                    "/api/spark-workers",
                    "/api/stream-metrics",
                    "/api/performance-history"
                ]
            }, 404)
    
    def serve_json(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def start_simple_api_server(port=8082):
    """Start the simple API server"""
    handler = SimpleAPIHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🚀 Simple API Server running at:")
            print(f"   http://localhost:{port}/")
            print(f"   http://127.0.0.1:{port}/")
            print("\n📊 Available API endpoints:")
            print(f"   GET  /api/status            - Service status")
            print(f"   GET  /api/bus-metrics       - Bus metrics")
            print(f"   GET  /api/metrics           - System metrics")
            print(f"   GET  /api/learning-metrics  - Learning metrics")
            print(f"   GET  /api/logs              - Agent logs")
            print(f"   GET  /api/rl-metrics        - RL metrics")
            print(f"   GET  /api/signatures        - Signature performance")
            print(f"   GET  /api/verifiers         - Verifier metrics")
            print(f"   GET  /api/kafka-topics      - Kafka topics")
            print(f"   GET  /api/spark-workers     - Spark workers")
            print(f"   GET  /api/stream-metrics    - Stream metrics")
            print(f"   GET  /api/performance-history - Performance history")
            print(f"\n🔄 Press Ctrl+C to stop the server")
            
            httpd.serve_forever()
            
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python3 simple_api_server.py 8083")
        else:
            print(f"❌ Error starting server: {e}")
    except KeyboardInterrupt:
        print("\n\n🛑 Simple API server stopped")

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8082
    start_simple_api_server(port)
