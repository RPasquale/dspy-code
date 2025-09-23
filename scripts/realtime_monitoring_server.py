#!/usr/bin/env python3
"""
Real-Time Agent Monitoring WebSocket Server

This server provides real-time WebSocket connections for the React frontend
to monitor agent actions, thoughts, learning progress, and system metrics.
"""

import asyncio
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime, timedelta
import websockets
from websockets.server import WebSocketServerProtocol
import logging

# Add the project root to the path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dspy_agent.db import get_enhanced_data_manager, Environment, ActionType, AgentState
from dspy_agent.agentic import get_retrieval_statistics, query_retrieval_events
from dspy_agent.streaming import LocalBus, StreamConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeMonitoringServer:
    """WebSocket server for real-time agent monitoring"""
    
    def __init__(self, host: str = "localhost", port: int = 8081):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.data_manager = get_enhanced_data_manager()
        self.monitoring = False
        self.monitor_thread = None
        
        # Real-time data caches
        self.recent_actions = []
        self.recent_thoughts = []
        self.learning_metrics = {}
        self.system_metrics = {}
        self.agent_status = "idle"
        self.current_task = None
        
        # Data update intervals
        self.update_intervals = {
            "actions": 1.0,      # 1 second
            "thoughts": 2.0,     # 2 seconds
            "learning": 5.0,     # 5 seconds
            "system": 3.0,       # 3 seconds
            "status": 1.0        # 1 second
        }
        
        # Last update times
        self.last_updates = {
            "actions": 0,
            "thoughts": 0,
            "learning": 0,
            "system": 0,
            "status": 0
        }
    
    async def register_client(self, websocket: WebSocketServerProtocol):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial data to the new client
        await self.send_initial_data(websocket)
    
    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """Unregister a WebSocket client"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_initial_data(self, websocket: WebSocketServerProtocol):
        """Send initial data to a new client"""
        try:
            # Send current status
            status_data = await self.get_agent_status()
            await websocket.send(json.dumps({
                "type": "status_update",
                "data": status_data,
                "timestamp": time.time()
            }))
            
            # Send recent actions
            actions_data = await self.get_recent_actions()
            await websocket.send(json.dumps({
                "type": "actions_update",
                "data": actions_data,
                "timestamp": time.time()
            }))
            
            # Send learning metrics
            learning_data = await self.get_learning_metrics()
            await websocket.send(json.dumps({
                "type": "learning_update",
                "data": learning_data,
                "timestamp": time.time()
            }))
            
            # Send system metrics
            system_data = await self.get_system_metrics()
            await websocket.send(json.dumps({
                "type": "system_update",
                "data": system_data,
                "timestamp": time.time()
            }))
            
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
    
    async def handle_client_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming messages from clients"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                # Client wants to subscribe to specific data types
                subscriptions = data.get("subscriptions", [])
                await self.handle_subscription(websocket, subscriptions)
            elif message_type == "get_data":
                # Client requests specific data
                data_type = data.get("data_type")
                await self.send_requested_data(websocket, data_type)
            elif message_type == "ping":
                # Heartbeat
                await websocket.send(json.dumps({"type": "pong", "timestamp": time.time()}))
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received from client")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def handle_subscription(self, websocket: WebSocketServerProtocol, subscriptions: List[str]):
        """Handle client subscription requests"""
        # For now, we'll send all data types to all clients
        # In the future, we could implement selective subscriptions
        await websocket.send(json.dumps({
            "type": "subscription_confirmed",
            "subscriptions": subscriptions,
            "timestamp": time.time()
        }))
    
    async def send_requested_data(self, websocket: WebSocketServerProtocol, data_type: str):
        """Send requested data to a specific client"""
        try:
            if data_type == "actions":
                data = await self.get_recent_actions()
            elif data_type == "learning":
                data = await self.get_learning_metrics()
            elif data_type == "system":
                data = await self.get_system_metrics()
            elif data_type == "status":
                data = await self.get_agent_status()
            else:
                data = {"error": f"Unknown data type: {data_type}"}
            
            await websocket.send(json.dumps({
                "type": f"{data_type}_update",
                "data": data,
                "timestamp": time.time()
            }))
            
        except Exception as e:
            logger.error(f"Error sending requested data: {e}")
    
    async def broadcast_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast an update to all connected clients"""
        if not self.clients:
            return
        
        message = json.dumps({
            "type": update_type,
            "data": data,
            "timestamp": time.time()
        })
        
        # Send to all clients
        disconnected_clients = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        try:
            # Get current context
            context = self.data_manager.get_current_context()
            
            # Get system health
            health = self.data_manager.get_system_health()
            
            # Get recent actions count
            recent_actions = self.data_manager.get_recent_actions(limit=10)
            
            status = {
                "agent_state": context.agent_state.value if context else "unknown",
                "current_task": context.current_task if context else None,
                "workspace_path": context.workspace_path if context else None,
                "active_files": context.active_files if context else [],
                "system_health": health,
                "recent_actions_count": len(recent_actions),
                "is_learning": len(recent_actions) > 0,
                "last_activity": max([a.timestamp for a in recent_actions]) if recent_actions else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting agent status: {e}")
            return {"error": str(e)}
    
    async def get_recent_actions(self) -> Dict[str, Any]:
        """Get recent agent actions"""
        try:
            actions = self.data_manager.get_recent_actions(limit=50)
            
            # Convert to serializable format
            actions_data = []
            for action in actions:
                actions_data.append({
                    "action_id": action.action_id,
                    "timestamp": action.timestamp,
                    "action_type": action.action_type.value,
                    "reward": action.reward,
                    "confidence": action.confidence,
                    "execution_time": action.execution_time,
                    "result_summary": str(action.result)[:100] + "..." if len(str(action.result)) > 100 else str(action.result),
                    "environment": action.environment.value
                })
            
            # Calculate action statistics
            if actions:
                rewards = [a.reward for a in actions]
                confidences = [a.confidence for a in actions]
                execution_times = [a.execution_time for a in actions]
                
                stats = {
                    "total_actions": len(actions),
                    "avg_reward": sum(rewards) / len(rewards),
                    "avg_confidence": sum(confidences) / len(confidences),
                    "avg_execution_time": sum(execution_times) / len(execution_times),
                    "high_reward_actions": sum(1 for r in rewards if r > 0.8),
                    "action_types": {}
                }
                
                # Count action types
                for action in actions:
                    action_type = action.action_type.value
                    stats["action_types"][action_type] = stats["action_types"].get(action_type, 0) + 1
            else:
                stats = {
                    "total_actions": 0,
                    "avg_reward": 0,
                    "avg_confidence": 0,
                    "avg_execution_time": 0,
                    "high_reward_actions": 0,
                    "action_types": {}
                }
            
            return {
                "actions": actions_data,
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"Error getting recent actions: {e}")
            return {"error": str(e)}
    
    async def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning and training metrics"""
        try:
            # Get training history
            training_history = self.data_manager.get_training_history(limit=20)
            
            # Get learning progress
            learning_progress = self.data_manager.get_learning_progress(sessions=10)
            
            # Get signature metrics
            signatures = self.data_manager.get_all_signature_metrics()
            
            # Get retrieval statistics
            retrieval_stats = {}
            try:
                # This would need a workspace path - using a default for now
                workspace = Path("/tmp")  # Default workspace
                retrieval_stats = get_retrieval_statistics(workspace)
            except Exception:
                retrieval_stats = {"total_events": 0, "avg_score": 0}
            
            # Calculate learning trends
            learning_trends = {}
            if training_history:
                training_accuracies = [t.training_accuracy for t in training_history]
                validation_accuracies = [t.validation_accuracy for t in training_history]
                losses = [t.loss for t in training_history]
                
                learning_trends = {
                    "training_accuracy": {
                        "current": training_accuracies[-1] if training_accuracies else 0,
                        "trend": "improving" if len(training_accuracies) > 1 and training_accuracies[-1] > training_accuracies[0] else "stable"
                    },
                    "validation_accuracy": {
                        "current": validation_accuracies[-1] if validation_accuracies else 0,
                        "trend": "improving" if len(validation_accuracies) > 1 and validation_accuracies[-1] > validation_accuracies[0] else "stable"
                    },
                    "loss": {
                        "current": losses[-1] if losses else 0,
                        "trend": "improving" if len(losses) > 1 and losses[-1] < losses[0] else "stable"
                    }
                }
            
            # Signature performance
            signature_performance = {}
            for sig in signatures:
                signature_performance[sig.signature_name] = {
                    "performance_score": sig.performance_score,
                    "success_rate": sig.success_rate,
                    "avg_response_time": sig.avg_response_time,
                    "active": sig.active
                }
            
            return {
                "training_sessions": len(training_history),
                "learning_progress": learning_progress,
                "learning_trends": learning_trends,
                "signature_performance": signature_performance,
                "retrieval_statistics": retrieval_stats,
                "active_signatures": len([s for s in signatures if s.active])
            }
            
        except Exception as e:
            logger.error(f"Error getting learning metrics: {e}")
            return {"error": str(e)}
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            # Get cache statistics
            cache_stats = self.data_manager.get_cache_stats()
            
            # Get recent logs
            recent_logs = self.data_manager.get_recent_logs(limit=20)
            
            # Get system health
            health = self.data_manager.get_system_health()
            
            # Calculate log statistics
            log_stats = {
                "total_logs": len(recent_logs),
                "error_count": sum(1 for log in recent_logs if log.level == "ERROR"),
                "warning_count": sum(1 for log in recent_logs if log.level == "WARN"),
                "info_count": sum(1 for log in recent_logs if log.level == "INFO"),
                "recent_errors": [log.message for log in recent_logs if log.level == "ERROR"][-5:]
            }
            
            return {
                "cache_performance": cache_stats,
                "log_statistics": log_stats,
                "system_health": health,
                "uptime": time.time() - (health.get("start_time", time.time()) if health else time.time()),
                "memory_usage": "2.1GB",  # Placeholder
                "cpu_usage": "45%"        # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e)}
    
    async def monitor_agent_data(self):
        """Background task to monitor agent data and broadcast updates"""
        logger.info("Starting agent data monitoring...")
        
        while self.monitoring:
            try:
                current_time = time.time()
                
                # Check if we need to update actions
                if current_time - self.last_updates["actions"] >= self.update_intervals["actions"]:
                    actions_data = await self.get_recent_actions()
                    await self.broadcast_update("actions_update", actions_data)
                    self.last_updates["actions"] = current_time
                
                # Check if we need to update learning metrics
                if current_time - self.last_updates["learning"] >= self.update_intervals["learning"]:
                    learning_data = await self.get_learning_metrics()
                    await self.broadcast_update("learning_update", learning_data)
                    self.last_updates["learning"] = current_time
                
                # Check if we need to update system metrics
                if current_time - self.last_updates["system"] >= self.update_intervals["system"]:
                    system_data = await self.get_system_metrics()
                    await self.broadcast_update("system_update", system_data)
                    self.last_updates["system"] = current_time
                
                # Check if we need to update status
                if current_time - self.last_updates["status"] >= self.update_intervals["status"]:
                    status_data = await self.get_agent_status()
                    await self.broadcast_update("status_update", status_data)
                    self.last_updates["status"] = current_time
                
                # Sleep for a short time to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a WebSocket client connection"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            await self.unregister_client(websocket)
    
    def start_monitoring(self):
        """Start the background monitoring thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._run_monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Agent monitoring started")
    
    def stop_monitoring(self):
        """Stop the background monitoring thread"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            logger.info("Agent monitoring stopped")
    
    def _run_monitoring_loop(self):
        """Run the monitoring loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.monitor_agent_data())
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        # Start monitoring
        self.start_monitoring()
        
        # Start WebSocket server
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"WebSocket server running on ws://{self.host}:{self.port}")
            logger.info("Clients can connect to monitor agent in real-time")
            
            # Keep the server running
            try:
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                logger.info("Shutting down server...")
                self.stop_monitoring()


async def main():
    """Main function to start the monitoring server"""
    server = RealtimeMonitoringServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        server.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
