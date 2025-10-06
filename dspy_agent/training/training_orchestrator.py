"""
Training Orchestrator for DSPy-Code Agent

This module orchestrates the entire training pipeline, integrating distributed training,
hyperparameter sweeps, judge models, and global objectives.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
import torch
import numpy as np
from dataclasses import dataclass, field
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue, Empty

# PufferLib imports
import pufferlib
from pufferlib.utils import setup_logging, get_logger

# Local imports
from .distributed_config import DistributedTrainingConfig, create_training_config
from .distributed_trainer import DistributedTrainer, create_distributed_trainer
from .hyperparameter_sweeps import HyperparameterSweep, create_sweep_config, run_hyperparameter_sweep
from .judge_models import create_judge_model, create_ensemble_judge
from .global_objective import GlobalObjectiveSystem, create_global_objective_system, GlobalObjectiveConfig
from ..streaming.streamkit import LocalBus
from ..agents.knowledge import KnowledgeBase


@dataclass
class TrainingOrchestratorConfig:
    """Configuration for the training orchestrator."""
    
    # Training configuration
    training_type: str = "distributed"  # distributed, sweep, hybrid
    num_episodes: int = 1000
    evaluation_interval: int = 100
    save_interval: int = 500
    
    # Distributed training
    distributed_config: Optional[DistributedTrainingConfig] = None
    num_workers: int = 4
    num_gpus: int = 1
    
    # Hyperparameter sweeps
    sweep_config: Optional[Dict[str, Any]] = None
    run_sweeps: bool = False
    sweep_framework: str = "protein"
    
    # Global objectives
    global_objective_config: Optional[GlobalObjectiveConfig] = None
    use_global_objectives: bool = True
    
    # Performance optimization
    auto_scaling: bool = True
    performance_monitoring: bool = True
    resource_optimization: bool = True
    
    # Logging and monitoring
    log_dir: str = "logs/training_orchestrator"
    tensorboard: bool = True
    wandb: bool = True
    wandb_project: str = "dspy-code-orchestrator"
    
    # Advanced features
    curriculum_learning: bool = True
    meta_learning: bool = True
    transfer_learning: bool = True
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create log directory
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup default configurations
        if self.distributed_config is None:
            self.distributed_config = create_training_config(
                framework="protein",
                num_gpus=self.num_gpus,
                distributed=True
            )
        
        if self.global_objective_config is None:
            from .global_objective import GlobalObjectiveConfig
            self.global_objective_config = GlobalObjectiveConfig()
        
        if self.sweep_config is None:
            self.sweep_config = {
                'sweep_name': 'dspy_code_sweep',
                'num_trials': 50,
                'framework': self.sweep_framework
            }


class TrainingOrchestrator:
    """Main training orchestrator for the DSPy-Code agent."""
    
    def __init__(self, config: TrainingOrchestratorConfig, bus: LocalBus):
        self.config = config
        self.bus = bus
        self.logger = get_logger(__name__)
        
        # Setup logging
        setup_logging(level=logging.INFO)
        
        # Initialize components
        self.knowledge_base = KnowledgeBase()
        self.global_objective_system = None
        self.distributed_trainer = None
        self.hyperparameter_sweep = None
        
        # Training state
        self.training_active = False
        self.current_episode = 0
        self.best_performance = 0.0
        self.performance_history = []
        
        # Threading
        self.training_thread = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Queues for communication
        self.training_queue = Queue()
        self.monitoring_queue = Queue()
        
        # Setup components
        self._setup_components()
        
        # Setup monitoring
        self._setup_monitoring()
    
    def _setup_components(self):
        """Setup all training components."""
        self.logger.info("Setting up training components")
        
        # Setup global objective system
        if self.config.use_global_objectives:
            self.global_objective_system = create_global_objective_system(
                config=self.config.global_objective_config,
                bus=self.bus
            )
        
        # Setup distributed trainer
        if self.config.training_type in ["distributed", "hybrid"]:
            self.distributed_trainer = create_distributed_trainer(
                config=self.config.distributed_config,
                bus=self.bus
            )
        
        # Setup hyperparameter sweep
        if self.config.run_sweeps or self.config.training_type == "sweep":
            self.hyperparameter_sweep = HyperparameterSweep(
                create_sweep_config(**self.config.sweep_config)
            )
    
    def _setup_monitoring(self):
        """Setup monitoring and logging."""
        if self.config.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(self.config.log_dir / "tensorboard")
        
        if self.config.wandb:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__
            )
    
    def start_training(self):
        """Start the training process."""
        self.logger.info("Starting training orchestrator")
        
        if self.training_active:
            self.logger.warning("Training is already active")
            return
        
        self.training_active = True
        self.stop_event.clear()
        
        # Start training thread
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.start()
        
        # Start monitoring thread
        if self.config.performance_monitoring:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.start()
        
        self.logger.info("Training orchestrator started")
    
    def stop_training(self):
        """Stop the training process."""
        self.logger.info("Stopping training orchestrator")
        
        self.training_active = False
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.training_thread:
            self.training_thread.join(timeout=10)
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        self.logger.info("Training orchestrator stopped")
    
    def _training_loop(self):
        """Main training loop."""
        self.logger.info("Starting training loop")
        
        try:
            while self.training_active and not self.stop_event.is_set():
                # Run training episode
                self._run_training_episode()
                
                # Check for evaluation
                if self.current_episode % self.config.evaluation_interval == 0:
                    self._run_evaluation()
                
                # Check for saving
                if self.current_episode % self.config.save_interval == 0:
                    self._save_checkpoint()
                
                # Update episode counter
                self.current_episode += 1
                
                # Check for convergence
                if self._check_convergence():
                    self.logger.info("Training converged")
                    break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Training loop error: {e}")
            self.training_active = False
        
        self.logger.info("Training loop completed")
    
    def _run_training_episode(self):
        """Run a single training episode."""
        # This would run the actual training episode
        # For now, we'll simulate the training
        
        # Simulate episode performance
        episode_performance = np.random.normal(0.8, 0.1)
        self.performance_history.append(episode_performance)
        
        # Update best performance
        if episode_performance > self.best_performance:
            self.best_performance = episode_performance
        
        # Log episode
        self.logger.info(f"Episode {self.current_episode}: Performance = {episode_performance:.3f}")
        
        # Send to monitoring queue
        self.monitoring_queue.put({
            'episode': self.current_episode,
            'performance': episode_performance,
            'best_performance': self.best_performance
        })
    
    def _run_evaluation(self):
        """Run evaluation on the current model."""
        self.logger.info(f"Running evaluation at episode {self.current_episode}")
        
        # This would run actual evaluation
        # For now, we'll simulate the evaluation
        
        evaluation_metrics = {
            'accuracy': np.random.uniform(0.7, 0.9),
            'precision': np.random.uniform(0.7, 0.9),
            'recall': np.random.uniform(0.7, 0.9),
            'f1_score': np.random.uniform(0.7, 0.9)
        }
        
        # Log evaluation results
        self.logger.info(f"Evaluation results: {evaluation_metrics}")
        
        # Send to monitoring queue
        self.monitoring_queue.put({
            'episode': self.current_episode,
            'evaluation': evaluation_metrics
        })
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = self.config.log_dir / f"checkpoint_episode_{self.current_episode}.json"
        
        checkpoint_data = {
            'episode': self.current_episode,
            'best_performance': self.best_performance,
            'performance_history': self.performance_history,
            'config': self.config.__dict__
        }
        
        with checkpoint_path.open('w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _check_convergence(self) -> bool:
        """Check if training has converged."""
        if len(self.performance_history) < 10:
            return False
        
        # Check if recent performance is stable
        recent_performance = self.performance_history[-10:]
        performance_std = np.std(recent_performance)
        
        return performance_std < 0.01  # Very stable performance
    
    def _monitoring_loop(self):
        """Monitoring loop for performance tracking."""
        self.logger.info("Starting monitoring loop")
        
        try:
            while self.training_active and not self.stop_event.is_set():
                # Process monitoring queue
                try:
                    while True:
                        data = self.monitoring_queue.get(timeout=1)
                        self._process_monitoring_data(data)
                except Empty:
                    pass
                
                # Small delay
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}")
        
        self.logger.info("Monitoring loop completed")
    
    def _process_monitoring_data(self, data: Dict[str, Any]):
        """Process monitoring data."""
        if 'episode' in data:
            # Episode data
            episode = data['episode']
            performance = data.get('performance', 0.0)
            
            # Log to tensorboard
            if self.config.tensorboard:
                self.tensorboard_writer.add_scalar('Performance/Episode', performance, episode)
                self.tensorboard_writer.add_scalar('Performance/Best', data.get('best_performance', 0.0), episode)
            
            # Log to wandb
            if self.config.wandb:
                import wandb
                wandb.log({
                    'episode': episode,
                    'performance': performance,
                    'best_performance': data.get('best_performance', 0.0)
                })
        
        elif 'evaluation' in data:
            # Evaluation data
            evaluation = data['evaluation']
            
            # Log to tensorboard
            if self.config.tensorboard:
                for metric, value in evaluation.items():
                    self.tensorboard_writer.add_scalar(f'Evaluation/{metric}', value, data['episode'])
            
            # Log to wandb
            if self.config.wandb:
                import wandb
                wandb.log({
                    'episode': data['episode'],
                    **evaluation
                })
    
    def run_hyperparameter_sweep(self):
        """Run hyperparameter sweep."""
        if not self.hyperparameter_sweep:
            self.logger.error("Hyperparameter sweep not configured")
            return
        
        self.logger.info("Starting hyperparameter sweep")
        
        try:
            results = self.hyperparameter_sweep.run_sweep()
            self.logger.info(f"Hyperparameter sweep completed with {len(results)} trials")
            
            # Find best trial
            best_trial = max(results, key=lambda t: t.metrics.episode_reward)
            self.logger.info(f"Best trial: {best_trial.trial_id} with reward: {best_trial.metrics.episode_reward:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Hyperparameter sweep failed: {e}")
            return []
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'training_active': self.training_active,
            'current_episode': self.current_episode,
            'best_performance': self.best_performance,
            'performance_history': self.performance_history[-10:] if self.performance_history else [],
            'convergence_status': self._check_convergence()
        }
    
    def save_state(self, path: Union[str, Path]):
        """Save the current state."""
        state = {
            'config': self.config.__dict__,
            'training_active': self.training_active,
            'current_episode': self.current_episode,
            'best_performance': self.best_performance,
            'performance_history': self.performance_history
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: Union[str, Path]):
        """Load the state."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.training_active = state['training_active']
        self.current_episode = state['current_episode']
        self.best_performance = state['best_performance']
        self.performance_history = state['performance_history']


def create_training_orchestrator(
    config: Optional[TrainingOrchestratorConfig] = None,
    bus: Optional[LocalBus] = None
) -> TrainingOrchestrator:
    """Create a training orchestrator."""
    
    if config is None:
        config = TrainingOrchestratorConfig()
    
    if bus is None:
        from ..streaming.streamkit import LocalBus
        bus = LocalBus()
    
    return TrainingOrchestrator(config, bus)


def run_training_orchestrator(
    config: Optional[TrainingOrchestratorConfig] = None,
    bus: Optional[LocalBus] = None
):
    """Run the training orchestrator."""
    
    orchestrator = create_training_orchestrator(config, bus)
    
    try:
        # Start training
        orchestrator.start_training()
        
        # Wait for training to complete
        while orchestrator.training_active:
            time.sleep(1)
        
        # Stop training
        orchestrator.stop_training()
        
    except KeyboardInterrupt:
        orchestrator.stop_training()
        print("Training interrupted by user")
    
    except Exception as e:
        orchestrator.stop_training()
        print(f"Training failed: {e}")


if __name__ == "__main__":
    # Example usage
    config = TrainingOrchestratorConfig(
        training_type="distributed",
        num_episodes=1000,
        num_workers=4,
        num_gpus=1,
        run_sweeps=True,
        use_global_objectives=True
    )
    
    orchestrator = create_training_orchestrator(config)
    
    # Start training
    orchestrator.start_training()
    
    # Wait for training to complete
    while orchestrator.training_active:
        time.sleep(1)
    
    # Stop training
    orchestrator.stop_training()
    
    print("Training completed")
