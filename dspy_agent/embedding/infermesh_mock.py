#!/usr/bin/env python3
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import time


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # type: ignore[override]
        if self.path.startswith('/health'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'ts': time.time()}).encode('utf-8'))
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):  # type: ignore[override]
        if self.path.startswith('/embed'):
            # Read payload and return simple zero vectors per input
            length = int(self.headers.get('Content-Length', '0') or '0')
            try:
                body = self.rfile.read(length) if length > 0 else b''
                obj = json.loads(body.decode('utf-8') or '{}')
                inputs = obj.get('inputs') or []
            except Exception:
                inputs = []
            dim = 384
            vectors = [[0.0] * dim for _ in inputs]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'vectors': vectors, 'model': obj.get('model')}).encode('utf-8'))
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):  # type: ignore[override]
        return


def main():
    server = HTTPServer(('0.0.0.0', 9000), Handler)
    server.serve_forever()


if __name__ == '__main__':
    main()

