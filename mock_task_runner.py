#!/usr/bin/env python3
"""
Mock Task Runner Service
Mimics the Rust env_runner HTTP API on port 8083 for testing.
"""
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskRunnerHandler(BaseHTTPRequestHandler):
    """HTTP handler for task execution requests."""
    
    def log_message(self, format: str, *args) -> None:
        """Suppress default HTTP logging."""
        pass
    
    def _json_response(self, code: int, data: dict) -> None:
        """Send a JSON response."""
        body = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == '/health':
            logger.info("Health check")
            return self._json_response(200, {"status": "healthy", "version": "mock-1.0"})
        
        if self.path == '/metrics':
            return self._json_response(200, {
                "tasks_processed": 0,
                "queue_depth": 0,
                "gpu_utilization": 0.0,
                "latency_p95_ms": 0,
                "avg_duration_ms": 0,
                "total_errors": 0,
                "uptime_seconds": 0
            })
        
        if self.path == '/prometheus':
            metrics = """# HELP env_runner_tasks_processed Total tasks processed
# TYPE env_runner_tasks_processed counter
env_runner_tasks_processed 0

# HELP env_runner_queue_depth Current queue depth
# TYPE env_runner_queue_depth gauge
env_runner_queue_depth 0
"""
            body = metrics.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        
        if self.path == '/hardware':
            return self._json_response(200, {
                "cpu": {"cores": 8, "threads": 16},
                "memory": {"total_bytes": 16000000000},
                "gpus": []
            })
        
        return self._json_response(404, {"error": "not_found"})
    
    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == '/tasks/execute':
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            try:
                task_request = json.loads(body.decode('utf-8'))
                task_id = task_request.get('task_id', 'unknown')
                task_class = task_request.get('class', 'unknown')
                payload = task_request.get('payload', {})
                
                logger.info(f"Executing task: {task_id} (class: {task_class})")
                logger.info(f"  Payload: {payload}")
                
                # Simulate task execution with a short delay
                time.sleep(0.1)
                
                # Return success response
                response = {
                    "task_id": task_id,
                    "status": "completed",
                    "result": {
                        "success": True,
                        "message": f"Mock execution of {task_id} completed",
                        "execution_time_ms": 100,
                        "mock": True
                    },
                    "error": None
                }
                
                logger.info(f"  ✅ Task {task_id} completed successfully")
                return self._json_response(200, response)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse request: {e}")
                return self._json_response(400, {"error": "invalid_json", "message": str(e)})
            except Exception as e:
                logger.error(f"Task execution error: {e}")
                return self._json_response(500, {"error": "execution_failed", "message": str(e)})
        
        return self._json_response(404, {"error": "not_found"})

def run_server(port: int = 8083):
    """Start the mock task runner server."""
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, TaskRunnerHandler)
    
    logger.info(f"🚀 Mock Task Runner starting on http://127.0.0.1:{port}")
    logger.info(f"   Health: http://127.0.0.1:{port}/health")
    logger.info(f"   Execute: POST http://127.0.0.1:{port}/tasks/execute")
    logger.info("")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down mock task runner")
        httpd.shutdown()

if __name__ == '__main__':
    run_server()

