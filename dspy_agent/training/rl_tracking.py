"""
RL Training Tracking System for DSPy-Code Agent

This module provides comprehensive tracking of RL training metrics, performance,
and hyperparameter sweeps using RedDB storage.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# RedDB imports
from ..db.data_models import RedDBDataManager
from ..db.factory import get_storage


@dataclass
class RLTrainingSession:
    """RL Training session record"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    framework: str = "auto"
    num_episodes: int = 0
    total_timesteps: int = 0
    num_workers: int = 1
    num_gpus: int = 0
    status: str = "running"  # running, completed, failed, stopped
    config: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    convergence_episode: Optional[int] = None
    best_performance: float = 0.0
    final_performance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'framework': self.framework,
            'num_episodes': self.num_episodes,
            'total_timesteps': self.total_timesteps,
            'num_workers': self.num_workers,
            'num_gpus': self.num_gpus,
            'status': self.status,
            'config': self.config,
            'performance_metrics': self.performance_metrics,
            'convergence_episode': self.convergence_episode,
            'best_performance': self.best_performance,
            'final_performance': self.final_performance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RLTrainingSession':
        return cls(**data)


@dataclass
class RLEpisodeMetrics:
    """RL Episode performance metrics"""
    session_id: str
    episode: int
    timestamp: float
    reward: float
    episode_length: int
    policy_loss: float
    value_loss: float
    entropy_loss: float
    learning_rate: float
    explained_variance: float
    fps: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: float
    convergence_score: float
    action_distribution: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'episode': self.episode,
            'timestamp': self.timestamp,
            'reward': self.reward,
            'episode_length': self.episode_length,
            'policy_loss': self.policy_loss,
            'value_loss': self.value_loss,
            'entropy_loss': self.entropy_loss,
            'learning_rate': self.learning_rate,
            'explained_variance': self.explained_variance,
            'fps': self.fps,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'gpu_usage': self.gpu_usage,
            'convergence_score': self.convergence_score,
            'action_distribution': self.action_distribution,
            'error_count': self.error_count,
            'success_rate': self.success_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RLEpisodeMetrics':
        return cls(**data)


@dataclass
class RLHyperparameterSweep:
    """RL Hyperparameter sweep record"""
    sweep_id: str
    start_time: float
    end_time: Optional[float] = None
    framework: str = "auto"
    num_trials: int = 0
    completed_trials: int = 0
    status: str = "running"  # running, completed, failed, stopped
    best_trial_id: Optional[str] = None
    best_performance: float = 0.0
    best_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    search_space: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sweep_id': self.sweep_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'framework': self.framework,
            'num_trials': self.num_trials,
            'completed_trials': self.completed_trials,
            'status': self.status,
            'best_trial_id': self.best_trial_id,
            'best_performance': self.best_performance,
            'best_hyperparameters': self.best_hyperparameters,
            'search_space': self.search_space,
            'results': self.results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RLHyperparameterSweep':
        return cls(**data)


@dataclass
class RLTrialResult:
    """RL Trial result for hyperparameter sweeps"""
    trial_id: str
    sweep_id: str
    start_time: float
    end_time: float
    hyperparameters: Dict[str, Any]
    performance: float
    convergence_episode: Optional[int]
    final_reward: float
    success: bool
    error_message: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trial_id': self.trial_id,
            'sweep_id': self.sweep_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'hyperparameters': self.hyperparameters,
            'performance': self.performance,
            'convergence_episode': self.convergence_episode,
            'final_reward': self.final_reward,
            'success': self.success,
            'error_message': self.error_message,
            'resource_usage': self.resource_usage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RLTrialResult':
        return cls(**data)


class RLTrackingSystem:
    """Comprehensive RL training tracking system using RedDB"""
    
    def __init__(self, namespace: str = "dspy_agent_rl"):
        self.data_manager = RedDBDataManager(namespace)
        self.storage = get_storage()
        self.logger = logging.getLogger(__name__)
        
        # Initialize RedDB tables for RL tracking
        self._init_rl_tables()
    
    def _init_rl_tables(self):
        """Initialize RedDB for RL tracking (using key-value storage)"""
        try:
            # RedDB uses key-value storage, not SQL tables
            # We'll use structured keys for different data types
            
            # Initialize tracking keys
            self.storage.put("rl_tracking:initialized", True)
            self.storage.put("rl_tracking:sessions:count", 0)
            self.storage.put("rl_tracking:sweeps:count", 0)
            
            self.logger.info("RL tracking initialized successfully with RedDB key-value storage")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RL tracking: {e}")
    
    def start_training_session(self, session_id: str, config: Dict[str, Any]) -> RLTrainingSession:
        """Start a new RL training session"""
        session = RLTrainingSession(
            session_id=session_id,
            start_time=time.time(),
            framework=config.get('framework', 'auto'),
            num_workers=config.get('num_workers', 1),
            num_gpus=config.get('num_gpus', 0),
            config=config
        )
        
        # Store in RedDB
        self.storage.put(f"rl_sessions:{session_id}", json.dumps(session.to_dict()))
        
        # Also store in time series
        self.storage.append("rl_training_sessions", session.to_dict())
        
        self.logger.info(f"Started RL training session: {session_id}")
        return session
    
    def update_training_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update a training session"""
        session_data = self.storage.get(f"rl_sessions:{session_id}")
        if session_data:
            session = RLTrainingSession.from_dict(json.loads(session_data))
            
            # Update fields
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            
            # Store updated session
            self.storage.put(f"rl_sessions:{session_id}", json.dumps(session.to_dict()))
            
            # Update time series
            self.storage.append("rl_training_sessions", session.to_dict())
    
    def log_episode_metrics(self, session_id: str, episode: int, metrics: Dict[str, Any]) -> None:
        """Log episode metrics"""
        episode_metrics = RLEpisodeMetrics(
            session_id=session_id,
            episode=episode,
            timestamp=time.time(),
            **metrics
        )
        
        # Store in RedDB
        self.storage.put(f"rl_episodes:{session_id}:{episode}", json.dumps(episode_metrics.to_dict()))
        
        # Also store in time series
        self.storage.append("rl_episode_metrics", episode_metrics.to_dict())
        
        # Update session performance
        self._update_session_performance(session_id, episode_metrics)
    
    def _update_session_performance(self, session_id: str, episode_metrics: RLEpisodeMetrics) -> None:
        """Update session performance metrics"""
        session_data = self.storage.get(f"rl_sessions:{session_id}")
        if session_data:
            session = RLTrainingSession.from_dict(json.loads(session_data))
            
            # Update performance metrics
            session.num_episodes = episode_metrics.episode
            session.total_timesteps += episode_metrics.episode_length
            
            # Update best performance
            if episode_metrics.reward > session.best_performance:
                session.best_performance = episode_metrics.reward
            
            # Update final performance
            session.final_performance = episode_metrics.reward
            
            # Check for convergence
            if episode_metrics.convergence_score > 0.95:
                session.convergence_episode = episode_metrics.episode
            
            # Store updated session
            self.storage.put(f"rl_sessions:{session_id}", json.dumps(session.to_dict()))
    
    def start_hyperparameter_sweep(self, sweep_id: str, config: Dict[str, Any]) -> RLHyperparameterSweep:
        """Start a new hyperparameter sweep"""
        sweep = RLHyperparameterSweep(
            sweep_id=sweep_id,
            start_time=time.time(),
            framework=config.get('framework', 'auto'),
            num_trials=config.get('num_trials', 100),
            search_space=config.get('search_space', {})
        )
        
        # Store in RedDB
        self.storage.put(f"rl_sweeps:{sweep_id}", json.dumps(sweep.to_dict()))
        
        # Also store in time series
        self.storage.append("rl_hyperparameter_sweeps", sweep.to_dict())
        
        self.logger.info(f"Started RL hyperparameter sweep: {sweep_id}")
        return sweep
    
    def log_trial_result(self, trial_id: str, sweep_id: str, result: Dict[str, Any]) -> None:
        """Log trial result"""
        trial_result = RLTrialResult(
            trial_id=trial_id,
            sweep_id=sweep_id,
            start_time=result.get('start_time', time.time()),
            end_time=result.get('end_time', time.time()),
            hyperparameters=result.get('hyperparameters', {}),
            performance=result.get('performance', 0.0),
            convergence_episode=result.get('convergence_episode'),
            final_reward=result.get('final_reward', 0.0),
            success=result.get('success', False),
            error_message=result.get('error_message'),
            resource_usage=result.get('resource_usage', {})
        )
        
        # Store in RedDB
        self.storage.put(f"rl_trials:{trial_id}", json.dumps(trial_result.to_dict()))
        
        # Also store in time series
        self.storage.append("rl_trial_results", trial_result.to_dict())
        
        # Update sweep with trial result
        self._update_sweep_with_trial(sweep_id, trial_result)
    
    def _update_sweep_with_trial(self, sweep_id: str, trial_result: RLTrialResult) -> None:
        """Update sweep with trial result"""
        sweep_data = self.storage.get(f"rl_sweeps:{sweep_id}")
        if sweep_data:
            sweep = RLHyperparameterSweep.from_dict(json.loads(sweep_data))
            
            # Update sweep metrics
            sweep.completed_trials += 1
            sweep.results.append(trial_result.to_dict())
            
            # Update best trial if this one is better
            if trial_result.performance > sweep.best_performance:
                sweep.best_performance = trial_result.performance
                sweep.best_trial_id = trial_result.trial_id
                sweep.best_hyperparameters = trial_result.hyperparameters
            
            # Check if sweep is complete
            if sweep.completed_trials >= sweep.num_trials:
                sweep.status = "completed"
                sweep.end_time = time.time()
            
            # Store updated sweep
            self.storage.put(f"rl_sweeps:{sweep_id}", json.dumps(sweep.to_dict()))
    
    def get_training_session(self, session_id: str) -> Optional[RLTrainingSession]:
        """Get training session by ID"""
        session_data = self.storage.get(f"rl_sessions:{session_id}")
        if session_data:
            return RLTrainingSession.from_dict(json.loads(session_data))
        return None
    
    def get_episode_metrics(self, session_id: str, episode: int) -> Optional[RLEpisodeMetrics]:
        """Get episode metrics by session ID and episode number"""
        episode_data = self.storage.get(f"rl_episodes:{session_id}:{episode}")
        if episode_data:
            return RLEpisodeMetrics.from_dict(json.loads(episode_data))
        return None
    
    def get_training_history(self, session_id: str, limit: int = 100) -> List[RLEpisodeMetrics]:
        """Get training history for a session"""
        # This would query the time series data
        # For now, return empty list
        return []
    
    def get_sweep_results(self, sweep_id: str) -> Optional[RLHyperparameterSweep]:
        """Get hyperparameter sweep results"""
        sweep_data = self.storage.get(f"rl_sweeps:{sweep_id}")
        if sweep_data:
            return RLHyperparameterSweep.from_dict(json.loads(sweep_data))
        return None
    
    def get_best_hyperparameters(self, sweep_id: str) -> Optional[Dict[str, Any]]:
        """Get best hyperparameters from a sweep"""
        sweep = self.get_sweep_results(sweep_id)
        if sweep:
            return sweep.best_hyperparameters
        return None
    
    def get_performance_summary(self, session_id: str) -> Dict[str, Any]:
        """Get performance summary for a session"""
        session = self.get_training_session(session_id)
        if not session:
            return {}
        
        return {
            'session_id': session.session_id,
            'framework': session.framework,
            'num_episodes': session.num_episodes,
            'total_timesteps': session.total_timesteps,
            'best_performance': session.best_performance,
            'final_performance': session.final_performance,
            'convergence_episode': session.convergence_episode,
            'status': session.status,
            'duration': session.end_time - session.start_time if session.end_time else time.time() - session.start_time
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics"""
        # This would aggregate metrics from all sessions
        return {
            'total_sessions': 0,
            'active_sessions': 0,
            'total_episodes': 0,
            'total_timesteps': 0,
            'average_performance': 0.0,
            'best_performance': 0.0,
            'convergence_rate': 0.0
        }


# Global RL tracking instance
_rl_tracker = None

def get_rl_tracker() -> RLTrackingSystem:
    """Get global RL tracking instance"""
    global _rl_tracker
    if _rl_tracker is None:
        _rl_tracker = RLTrackingSystem()
    return _rl_tracker
