from __future__ import annotations

import http.server
import json
import socketserver
import time
from typing import Any, Dict

from ..db.redb_router import RedDBRouter, IngestRequest, QueryRequest
from ..dbkit import RedDBStorage


class IntelligentBackendHandler(http.server.BaseHTTPRequestHandler):
    router = RedDBRouter(RedDBStorage(url=None, namespace="agent"))

    def _json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get('Content-Length') or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode('utf-8') or '{}')
        except Exception:
            return {}

    def _send_json(self, data: Dict[str, Any], code: int = 200):
        try:
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        except Exception:
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/db/health':
            self._send_json({'ok': True, 'ts': time.time(), 'storage': self.router.st.health_check()})
            return
        if self.path == '/api/db/stats':
            # Minimal stats snapshot
            self._send_json({'ok': True, 'ts': time.time(), 'namespace': self.router.st.ns})
            return
        self._send_json({'error': 'not found', 'path': self.path}, 404)

    def do_POST(self):
        if self.path == '/api/db/ingest':
            body = self._json()
            try:
                req = IngestRequest(**body)
            except TypeError:
                # Allow lenient input
                body.setdefault('kind', body.get('type', 'auto'))
                req = IngestRequest(**body)
            out = self.router.route_ingest(req)
            self._send_json(out, 200 if out.get('ok') else 400)
            return
        if self.path == '/api/db/query':
            body = self._json()
            try:
                req = QueryRequest(**body)
            except TypeError:
                body.setdefault('mode', 'auto')
                req = QueryRequest(**body)
            # Vector-first with document fallback when auto
            out = self.router.route_query(req)
            if out.get('mode') == 'vector' and not (out.get('hits') or []):
                # Try document collection fallback
                q2 = QueryRequest(mode='document', namespace=req.namespace, text=req.text, collection=req.collection)
                out = self.router.route_query(q2)
            self._send_json(out)
            return
        self._send_json({'error': 'not found', 'path': self.path}, 404)


def start_intelligent_backend_server(port: int = 8766):
    with socketserver.TCPServer(("", port), IntelligentBackendHandler) as httpd:
        print(f"Intelligent Backend Server listening on http://127.0.0.1:{port}")
        httpd.serve_forever()


if __name__ == '__main__':
    import sys
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    start_intelligent_backend_server(p)

