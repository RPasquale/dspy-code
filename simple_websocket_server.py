#!/usr/bin/env python3
"""
Simple WebSocket server for DSPy Agent frontend
Provides real-time data updates
"""

import asyncio
import websockets
import json
import time
from datetime import datetime

class SimpleWebSocketServer:
    def __init__(self, host="localhost", port=8081):
        self.host = host
        self.port = port
        self.clients = set()
        
    async def register_client(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
    async def unregister_client(self, websocket):
        """Unregister a client"""
        self.clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def broadcast_message(self, message):
        """Broadcast a message to all connected clients"""
        if self.clients:
            # Create a copy of the clients set to avoid modification during iteration
            clients_copy = self.clients.copy()
            # Send message to all clients
            await asyncio.gather(
                *[client.send(message) for client in clients_copy],
                return_exceptions=True
            )
    
    async def generate_mock_data(self):
        """Generate mock real-time data"""
        counter = 0
        while True:
            counter += 1
            
            # Generate mock metrics data
            data = {
                "type": "metrics_update",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "cpu_usage": 45.2 + (counter % 20),
                    "memory_usage": 67.8 + (counter % 15),
                    "messages_processed": 1234 + counter,
                    "active_connections": len(self.clients),
                    "queue_depth": 42 + (counter % 10)
                }
            }
            
            await self.broadcast_message(json.dumps(data))
            await asyncio.sleep(2)  # Send updates every 2 seconds
    
    async def handle_client(self, websocket, path):
        """Handle a client connection"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                # Echo back any messages received
                response = {
                    "type": "echo",
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Echo: {message}"
                }
                await websocket.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self):
        """Start the WebSocket server"""
        print(f"🚀 Starting WebSocket server on {self.host}:{self.port}")
        
        # Start the mock data generator
        asyncio.create_task(self.generate_mock_data())
        
        # Start the WebSocket server
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"✅ WebSocket server running at ws://{self.host}:{self.port}")
            print("🔄 Broadcasting mock data every 2 seconds")
            print("🛑 Press Ctrl+C to stop")
            await asyncio.Future()  # Run forever

async def main():
    server = SimpleWebSocketServer()
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("\n🛑 WebSocket server stopped")

if __name__ == "__main__":
    asyncio.run(main())
