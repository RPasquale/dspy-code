#!/usr/bin/env python3
"""
Simple RedDB HTTP Server
Provides the HTTP API endpoints that your DSPy agent expects.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
from pathlib import Path
from threading import Lock
import sqlite3

try:
    from fastapi import FastAPI, HTTPException, Request, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    import uvicorn
except ImportError:
    print("FastAPI not available. Install with: pip install fastapi uvicorn")
    exit(1)


class RedDBServer:
    """Simple file-based RedDB implementation with HTTP API."""
    
    def __init__(self, data_dir: str = "/data", admin_token: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.admin_token = admin_token
        self.lock = Lock()
        
        # Initialize SQLite database
        self.db_path = self.data_dir / "reddb.sqlite"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    namespace TEXT,
                    key TEXT,
                    value TEXT,
                    created_at REAL,
                    updated_at REAL,
                    PRIMARY KEY (namespace, key)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS streams (
                    namespace TEXT,
                    stream TEXT,
                    offset INTEGER,
                    value TEXT,
                    created_at REAL,
                    PRIMARY KEY (namespace, stream, offset)
                )
            """)
            conn.commit()
    
    def _verify_token(self, token: Optional[str]) -> bool:
        """Verify admin token if configured."""
        if not self.admin_token:
            return True
        return token == self.admin_token
    
    # KV Operations
    def put_kv(self, namespace: str, key: str, value: Any, token: Optional[str] = None) -> Dict[str, Any]:
        if not self._verify_token(token):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                now = time.time()
                conn.execute(
                    "INSERT OR REPLACE INTO kv_store (namespace, key, value, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (namespace, key, json.dumps(value), now, now)
                )
                conn.commit()
        
        return {"ok": True, "namespace": namespace, "key": key}
    
    def get_kv(self, namespace: str, key: str, token: Optional[str] = None) -> Any:
        if not self._verify_token(token):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM kv_store WHERE namespace = ? AND key = ?",
                (namespace, key)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
    
    def delete_kv(self, namespace: str, key: str, token: Optional[str] = None) -> Dict[str, Any]:
        if not self._verify_token(token):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM kv_store WHERE namespace = ? AND key = ?",
                    (namespace, key)
                )
                conn.commit()
        
        return {"ok": True, "namespace": namespace, "key": key}
    
    # Stream Operations
    def append_stream(self, namespace: str, stream: str, value: Any, token: Optional[str] = None) -> Dict[str, Any]:
        if not self._verify_token(token):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # Get next offset
                cursor = conn.execute(
                    "SELECT MAX(offset) FROM streams WHERE namespace = ? AND stream = ?",
                    (namespace, stream)
                )
                row = cursor.fetchone()
                next_offset = (row[0] or -1) + 1
                
                # Insert new record
                now = time.time()
                conn.execute(
                    "INSERT INTO streams (namespace, stream, offset, value, created_at) VALUES (?, ?, ?, ?, ?)",
                    (namespace, stream, next_offset, json.dumps(value), now)
                )
                conn.commit()
        
        return {"ok": True, "namespace": namespace, "stream": stream, "offset": next_offset}
    
    def read_stream(self, namespace: str, stream: str, start: int = 0, count: int = 100, token: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._verify_token(token):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT offset, value FROM streams WHERE namespace = ? AND stream = ? AND offset >= ? ORDER BY offset LIMIT ?",
                (namespace, stream, start, count)
            )
            rows = cursor.fetchall()
            return [{"offset": row[0], "value": json.loads(row[1])} for row in rows]
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "ok": True,
            "service": "reddb",
            "version": "1.0.0",
            "timestamp": time.time(),
            "data_dir": str(self.data_dir),
            "auth_enabled": bool(self.admin_token)
        }


# FastAPI app
app = FastAPI(title="RedDB Server", version="1.0.0")
security = HTTPBearer(auto_error=False)

# Initialize RedDB server
admin_token = os.getenv("REDB_ADMIN_TOKEN") or os.getenv("REDDB_ADMIN_TOKEN")
reddb = RedDBServer(data_dir=os.getenv("REDDB_DATA_DIR", "/data"), admin_token=admin_token)


def get_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """Extract token from Authorization header."""
    return credentials.credentials if credentials else None


@app.get("/health")
def health():
    """Health check endpoint."""
    return reddb.health_check()


@app.get("/api/health")
def api_health():
    """Alternative health check endpoint."""
    return reddb.health_check()


# KV endpoints
@app.put("/api/kv/{namespace}/{key}")
def put_kv(namespace: str, key: str, request: Request, token: Optional[str] = Depends(get_token)):
    """Store a key-value pair."""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        body = loop.run_until_complete(request.json())
        loop.close()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    return reddb.put_kv(namespace, key, body, token)


@app.get("/api/kv/{namespace}/{key}")
def get_kv(namespace: str, key: str, token: Optional[str] = Depends(get_token)):
    """Get a value by key."""
    result = reddb.get_kv(namespace, key, token)
    if result is None:
        raise HTTPException(status_code=404, detail="Key not found")
    return result


@app.delete("/api/kv/{namespace}/{key}")
def delete_kv(namespace: str, key: str, token: Optional[str] = Depends(get_token)):
    """Delete a key-value pair."""
    return reddb.delete_kv(namespace, key, token)


# Stream endpoints
@app.post("/api/streams/{namespace}/{stream}/append")
def append_stream(namespace: str, stream: str, request: Request, token: Optional[str] = Depends(get_token)):
    """Append a value to a stream."""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        body = loop.run_until_complete(request.json())
        loop.close()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    return reddb.append_stream(namespace, stream, body, token)


@app.get("/api/streams/{namespace}/{stream}/read")
def read_stream(namespace: str, stream: str, start: int = 0, count: int = 100, token: Optional[str] = Depends(get_token)):
    """Read from a stream."""
    return reddb.read_stream(namespace, stream, start, count, token)


if __name__ == "__main__":
    host = os.getenv("REDDB_HOST", "0.0.0.0")
    port = int(os.getenv("REDDB_PORT", "8080"))
    
    print(f"Starting RedDB server on {host}:{port}")
    if admin_token:
        print(f"Authentication enabled with token: {admin_token[:8]}...")
    else:
        print("Authentication disabled")
    
    uvicorn.run(app, host=host, port=port, log_level="info")
