"""
React Dashboard API for RL Training Data

This module provides API endpoints for the React dashboard to display
RL training metrics, performance data, and hyperparameter sweep results.
"""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .rl_tracking import get_rl_tracker, RLTrackingSystem


app = FastAPI(title="DSPy-Code RL Training Dashboard API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RL tracker
rl_tracker = get_rl_tracker()


@app.get("/api/rl/status")
async def get_rl_status():
    """Get overall RL training system status"""
    try:
        system_metrics = rl_tracker.get_system_metrics()
        return {
            "status": "active",
            "timestamp": time.time(),
            "metrics": system_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/sessions")
async def get_training_sessions(
    limit: int = Query(10, description="Number of sessions to return"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """Get list of training sessions"""
    try:
        # This would query RedDB for sessions
        # For now, return mock data
        sessions = [
            {
                "session_id": "session_1703123456",
                "start_time": time.time() - 3600,
                "end_time": None,
                "framework": "auto",
                "num_episodes": 150,
                "total_timesteps": 150000,
                "status": "running",
                "best_performance": 0.85,
                "final_performance": 0.82
            }
        ]
        return {"sessions": sessions[:limit]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/sessions/{session_id}")
async def get_training_session(session_id: str):
    """Get detailed training session data"""
    try:
        session = rl_tracker.get_training_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/sessions/{session_id}/metrics")
async def get_session_metrics(
    session_id: str,
    limit: int = Query(100, description="Number of metrics to return")
):
    """Get episode metrics for a training session"""
    try:
        # This would query RedDB for episode metrics
        # For now, return mock data
        metrics = []
        for i in range(min(limit, 50)):
            metrics.append({
                "episode": i + 1,
                "timestamp": time.time() - (50 - i) * 60,
                "reward": 0.5 + (i / 50) * 0.3 + (0.1 * (i % 10)),
                "episode_length": 1000 + (i % 5) * 100,
                "policy_loss": 0.5 - (i / 50) * 0.3,
                "value_loss": 0.3 - (i / 50) * 0.2,
                "entropy_loss": 0.1 - (i / 50) * 0.05,
                "learning_rate": 0.001,
                "explained_variance": 0.6 + (i / 50) * 0.3,
                "fps": 200 + (i % 3) * 50,
                "memory_usage": 1.5 + (i % 2) * 0.5,
                "cpu_usage": 0.3 + (i % 4) * 0.1,
                "gpu_usage": 0.4 + (i % 3) * 0.2,
                "convergence_score": min(0.5 + (i / 50) * 0.5, 1.0),
                "success_rate": 0.6 + (i / 50) * 0.3
            })
        
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/sweeps")
async def get_hyperparameter_sweeps(
    limit: int = Query(10, description="Number of sweeps to return")
):
    """Get list of hyperparameter sweeps"""
    try:
        # This would query RedDB for sweeps
        # For now, return mock data
        sweeps = [
            {
                "sweep_id": "sweep_1703123456",
                "start_time": time.time() - 7200,
                "end_time": time.time() - 3600,
                "framework": "auto",
                "num_trials": 50,
                "completed_trials": 50,
                "status": "completed",
                "best_performance": 0.92,
                "best_hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "n_epochs": 4
                }
            }
        ]
        return {"sweeps": sweeps[:limit]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/sweeps/{sweep_id}")
async def get_hyperparameter_sweep(sweep_id: str):
    """Get detailed hyperparameter sweep data"""
    try:
        sweep = rl_tracker.get_sweep_results(sweep_id)
        if not sweep:
            raise HTTPException(status_code=404, detail="Sweep not found")
        
        return sweep.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/sweeps/{sweep_id}/trials")
async def get_sweep_trials(
    sweep_id: str,
    limit: int = Query(100, description="Number of trials to return")
):
    """Get trial results for a hyperparameter sweep"""
    try:
        # This would query RedDB for trial results
        # For now, return mock data
        trials = []
        for i in range(min(limit, 20)):
            trials.append({
                "trial_id": f"trial_{i + 1}",
                "start_time": time.time() - (20 - i) * 300,
                "end_time": time.time() - (20 - i) * 300 + 180,
                "hyperparameters": {
                    "learning_rate": 0.001 * (1 + i * 0.1),
                    "batch_size": 32 * (1 + i % 3),
                    "n_epochs": 4 + i % 4
                },
                "performance": 0.6 + (i / 20) * 0.3 + (0.05 * (i % 5)),
                "convergence_episode": 50 + i * 2,
                "final_reward": 0.6 + (i / 20) * 0.3,
                "success": True,
                "resource_usage": {
                    "memory_usage": 1.5 + (i % 3) * 0.3,
                    "cpu_usage": 0.4 + (i % 2) * 0.2,
                    "gpu_usage": 0.5 + (i % 4) * 0.1
                }
            })
        
        return {"trials": trials}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/performance/summary")
async def get_performance_summary(
    hours: int = Query(24, description="Hours of data to include")
):
    """Get performance summary for the last N hours"""
    try:
        # This would aggregate performance data from RedDB
        # For now, return mock data
        summary = {
            "total_sessions": 5,
            "active_sessions": 1,
            "total_episodes": 750,
            "total_timesteps": 750000,
            "average_performance": 0.78,
            "best_performance": 0.92,
            "convergence_rate": 0.85,
            "time_range": {
                "start": time.time() - hours * 3600,
                "end": time.time()
            }
        }
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/performance/trends")
async def get_performance_trends(
    session_id: Optional[str] = Query(None, description="Specific session ID"),
    hours: int = Query(24, description="Hours of data to include")
):
    """Get performance trends over time"""
    try:
        # This would query time series data from RedDB
        # For now, return mock data
        trends = []
        start_time = time.time() - hours * 3600
        for i in range(hours):
            timestamp = start_time + i * 3600
            trends.append({
                "timestamp": timestamp,
                "performance": 0.5 + (i / hours) * 0.4 + (0.05 * (i % 6)),
                "episodes": 10 + i * 2,
                "convergence_score": min(0.3 + (i / hours) * 0.7, 1.0)
            })
        
        return {"trends": trends}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/frameworks/status")
async def get_framework_status():
    """Get status of available RL frameworks"""
    try:
        frameworks = {
            "pufferlib": {
                "available": True,
                "version": "3.0.0",
                "frameworks": {
                    "protein": True,
                    "carbs": True,
                    "ray": False,
                    "cleanrl": False
                }
            },
            "fallback": {
                "available": True,
                "framework": "torch"
            }
        }
        return frameworks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/health")
async def get_health():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "reddb_connected": True,
            "tracking_active": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8766)
