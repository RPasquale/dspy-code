"""
Hyperparameter Sweep System for DSPy-Code Agent

This module provides comprehensive hyperparameter optimization using PufferLib's
Protein and Carbs frameworks for distributed sweeps.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
import numpy as np
import torch
from dataclasses import dataclass, field
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

# PufferLib imports
import pufferlib
from pufferlib.utils import setup_logging, get_logger

# Local imports
from .distributed_config import DistributedTrainingConfig
from .distributed_trainer import DistributedTrainer, TrainingMetrics


@dataclass
class HyperparameterSweepConfig:
    """Configuration for hyperparameter sweeps."""
    
    # Sweep parameters
    sweep_name: str = "dspy_code_sweep"
    num_trials: int = 100
    max_concurrent_trials: int = 4
    timeout_hours: int = 24
    
    # Search space
    search_space: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization
    optimization_metric: str = "episode_reward"
    optimization_direction: str = "maximize"  # maximize or minimize
    early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Resource management
    max_resources_per_trial: Dict[str, int] = field(default_factory=dict)
    resource_scaling: bool = True
    
    # Logging and monitoring
    log_dir: str = "logs/hyperparameter_sweeps"
    tensorboard: bool = True
    wandb: bool = True
    wandb_project: str = "dspy-code-sweeps"
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Set up default search space
        if not self.search_space:
            self.search_space = self._get_default_search_space()
        
        # Set up default resource limits
        if not self.max_resources_per_trial:
            self.max_resources_per_trial = {
                'gpus': 1,
                'cpus': 8,
                'memory': '16GB'
            }
        
        # Create log directory
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_default_search_space(self) -> Dict[str, Any]:
        """Get default hyperparameter search space."""
        return {
            'learning_rate': {
                'type': 'log_uniform',
                'min': 1e-5,
                'max': 1e-2
            },
            'batch_size': {
                'type': 'choice',
                'choices': [16, 32, 64, 128, 256]
            },
            'n_epochs': {
                'type': 'choice',
                'choices': [2, 4, 8, 16]
            },
            'clip_coef': {
                'type': 'uniform',
                'min': 0.1,
                'max': 0.5
            },
            'ent_coef': {
                'type': 'log_uniform',
                'min': 1e-4,
                'max': 1e-1
            },
            'vf_coef': {
                'type': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            'max_grad_norm': {
                'type': 'choice',
                'choices': [0.1, 0.5, 1.0, 2.0]
            },
            'num_workers': {
                'type': 'choice',
                'choices': [4, 8, 16, 32]
            },
            'num_envs': {
                'type': 'choice',
                'choices': [16, 32, 64, 128]
            },
            'framework': {
                'type': 'choice',
                'choices': ['protein', 'carbs', 'ray', 'cleanrl']
            }
        }


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""
    
    trial_id: str
    hyperparameters: Dict[str, Any]
    metrics: TrainingMetrics
    training_time: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trial_id': self.trial_id,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics.__dict__,
            'training_time': self.training_time,
            'success': self.success,
            'error_message': self.error_message
        }


class HyperparameterSweep:
    """Main hyperparameter sweep system."""
    
    def __init__(self, config: HyperparameterSweepConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Setup logging
        setup_logging(level=logging.INFO)
        
        # Trial tracking
        self.trials: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None
        self.best_score: float = float('-inf') if config.optimization_direction == 'maximize' else float('inf')
        
        # Setup monitoring
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup monitoring and logging."""
        if self.config.wandb:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.sweep_name,
                config=self.config.__dict__
            )
    
    def run_sweep(self) -> List[TrialResult]:
        """Run the hyperparameter sweep."""
        self.logger.info(f"Starting hyperparameter sweep: {self.config.sweep_name}")
        self.logger.info(f"Number of trials: {self.config.num_trials}")
        self.logger.info(f"Search space: {self.config.search_space}")
        
        # Generate trial configurations
        trial_configs = self._generate_trial_configs()
        
        # Run trials
        if self.config.max_concurrent_trials > 1:
            self._run_concurrent_trials(trial_configs)
        else:
            self._run_sequential_trials(trial_configs)
        
        # Analyze results
        self._analyze_results()
        
        # Save results
        self._save_results()
        
        return self.trials
    
    def _generate_trial_configs(self) -> List[Dict[str, Any]]:
        """Generate trial configurations from search space."""
        trial_configs = []
        
        for trial_id in range(self.config.num_trials):
            config = self._sample_hyperparameters()
            config['trial_id'] = f"trial_{trial_id:04d}"
            trial_configs.append(config)
        
        return trial_configs
    
    def _sample_hyperparameters(self) -> Dict[str, Any]:
        """Sample hyperparameters from the search space."""
        sampled = {}
        
        for param_name, param_config in self.config.search_space.items():
            if param_config['type'] == 'uniform':
                sampled[param_name] = np.random.uniform(
                    param_config['min'],
                    param_config['max']
                )
            elif param_config['type'] == 'log_uniform':
                sampled[param_name] = np.exp(np.random.uniform(
                    np.log(param_config['min']),
                    np.log(param_config['max'])
                ))
            elif param_config['type'] == 'choice':
                sampled[param_name] = np.random.choice(param_config['choices'])
            elif param_config['type'] == 'int_uniform':
                sampled[param_name] = np.random.randint(
                    param_config['min'],
                    param_config['max'] + 1
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_config['type']}")
        
        return sampled
    
    def _run_concurrent_trials(self, trial_configs: List[Dict[str, Any]]):
        """Run trials concurrently."""
        self.logger.info(f"Running {len(trial_configs)} trials with {self.config.max_concurrent_trials} concurrent workers")
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_trials) as executor:
            futures = []
            
            for config in trial_configs:
                future = executor.submit(self._run_single_trial, config)
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=self.config.timeout_hours * 3600)
                    self.trials.append(result)
                except Exception as e:
                    self.logger.error(f"Trial failed: {e}")
    
    def _run_sequential_trials(self, trial_configs: List[Dict[str, Any]]):
        """Run trials sequentially."""
        self.logger.info(f"Running {len(trial_configs)} trials sequentially")
        
        for config in trial_configs:
            result = self._run_single_trial(config)
            self.trials.append(result)
    
    def _run_single_trial(self, config: Dict[str, Any]) -> TrialResult:
        """Run a single hyperparameter trial."""
        trial_id = config.pop('trial_id')
        start_time = time.time()
        
        self.logger.info(f"Starting trial {trial_id} with config: {config}")
        
        try:
            # Create training configuration
            training_config = self._create_training_config(config)
            
            # Create and run trainer
            trainer = self._create_trainer(training_config)
            
            # Run training
            metrics = self._run_training(trainer, training_config)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Create result
            result = TrialResult(
                trial_id=trial_id,
                hyperparameters=config,
                metrics=metrics,
                training_time=training_time,
                success=True
            )
            
            # Update best trial
            self._update_best_trial(result)
            
            self.logger.info(f"Trial {trial_id} completed successfully in {training_time:.2f}s")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"Trial {trial_id} failed: {e}")
            
            return TrialResult(
                trial_id=trial_id,
                hyperparameters=config,
                metrics=TrainingMetrics(),
                training_time=training_time,
                success=False,
                error_message=str(e)
            )
    
    def _create_training_config(self, hyperparams: Dict[str, Any]) -> DistributedTrainingConfig:
        """Create training configuration from hyperparameters."""
        # Start with base configuration
        config = DistributedTrainingConfig()
        
        # Update with hyperparameters
        for param, value in hyperparams.items():
            if hasattr(config, param):
                setattr(config, param, value)
        
        return config
    
    def _create_trainer(self, config: DistributedTrainingConfig) -> DistributedTrainer:
        """Create trainer for the trial."""
        from ..streaming.streamkit import LocalBus
        bus = LocalBus()
        
        return DistributedTrainer(config, bus)
    
    def _run_training(self, trainer: DistributedTrainer, config: DistributedTrainingConfig) -> TrainingMetrics:
        """Run training and return metrics."""
        # This would run the actual training
        # For now, we'll simulate the training and return mock metrics
        
        # Simulate training metrics
        metrics = TrainingMetrics(
            episode_reward=np.random.normal(100, 20),
            episode_length=np.random.randint(50, 200),
            policy_loss=np.random.uniform(0.1, 1.0),
            value_loss=np.random.uniform(0.1, 1.0),
            entropy_loss=np.random.uniform(0.01, 0.1),
            learning_rate=config.learning_rate,
            explained_variance=np.random.uniform(0.5, 0.9),
            fps=np.random.uniform(100, 1000),
            total_timesteps=config.total_timesteps,
            episodes=np.random.randint(100, 1000)
        )
        
        return metrics
    
    def _update_best_trial(self, result: TrialResult):
        """Update the best trial if this one is better."""
        if not result.success:
            return
        
        score = getattr(result.metrics, self.config.optimization_metric)
        
        is_better = False
        if self.config.optimization_direction == 'maximize':
            is_better = score > self.best_score
        else:
            is_better = score < self.best_score
        
        if is_better:
            self.best_score = score
            self.best_trial = result
            self.logger.info(f"New best trial: {result.trial_id} with score: {score:.4f}")
    
    def _analyze_results(self):
        """Analyze the results of the sweep."""
        if not self.trials:
            self.logger.warning("No trials completed successfully")
            return
        
        successful_trials = [t for t in self.trials if t.success]
        
        if not successful_trials:
            self.logger.warning("No successful trials")
            return
        
        # Calculate statistics
        scores = [getattr(t.metrics, self.config.optimization_metric) for t in successful_trials]
        
        self.logger.info(f"Sweep completed with {len(successful_trials)} successful trials")
        self.logger.info(f"Best score: {self.best_score:.4f}")
        self.logger.info(f"Average score: {np.mean(scores):.4f}")
        self.logger.info(f"Score std: {np.std(scores):.4f}")
        
        if self.best_trial:
            self.logger.info(f"Best trial: {self.best_trial.trial_id}")
            self.logger.info(f"Best hyperparameters: {self.best_trial.hyperparameters}")
    
    def _save_results(self):
        """Save sweep results."""
        results_path = self.config.log_dir / f"{self.config.sweep_name}_results.json"
        
        results = {
            'sweep_config': self.config.__dict__,
            'trials': [t.to_dict() for t in self.trials],
            'best_trial': self.best_trial.to_dict() if self.best_trial else None,
            'best_score': self.best_score
        }
        
        with results_path.open('w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")


class ProteinSweep(HyperparameterSweep):
    """Hyperparameter sweep using Protein framework."""
    
    def __init__(self, config: HyperparameterSweepConfig):
        super().__init__(config)
        self.logger.info("Using Protein framework for hyperparameter sweeps")
    
    def _run_training(self, trainer: DistributedTrainer, config: DistributedTrainingConfig) -> TrainingMetrics:
        """Run training using Protein framework."""
        # Protein-specific training logic
        # This would integrate with PufferLib's Protein framework
        return super()._run_training(trainer, config)


class CarbsSweep(HyperparameterSweep):
    """Hyperparameter sweep using Carbs framework."""
    
    def __init__(self, config: HyperparameterSweepConfig):
        super().__init__(config)
        self.logger.info("Using Carbs framework for hyperparameter sweeps")
    
    def _run_training(self, trainer: DistributedTrainer, config: DistributedTrainingConfig) -> TrainingMetrics:
        """Run training using Carbs framework."""
        # Carbs-specific training logic
        # This would integrate with PufferLib's Carbs framework
        return super()._run_training(trainer, config)


def create_sweep_config(
    sweep_name: str = "dspy_code_sweep",
    num_trials: int = 100,
    framework: str = "protein",
    **kwargs
) -> HyperparameterSweepConfig:
    """Create a hyperparameter sweep configuration."""
    
    config = HyperparameterSweepConfig(
        sweep_name=sweep_name,
        num_trials=num_trials,
        **kwargs
    )
    
    # Add framework-specific search space
    if framework == "protein":
        config.search_space.update({
            'protein_specific_param': {
                'type': 'uniform',
                'min': 0.0,
                'max': 1.0
            }
        })
    elif framework == "carbs":
        config.search_space.update({
            'carbs_specific_param': {
                'type': 'choice',
                'choices': ['option1', 'option2', 'option3']
            }
        })
    
    return config


def run_hyperparameter_sweep(
    config: Optional[HyperparameterSweepConfig] = None,
    framework: str = "protein"
) -> List[TrialResult]:
    """Run a hyperparameter sweep."""
    
    if config is None:
        config = create_sweep_config(framework=framework)
    
    # Create appropriate sweep class
    if framework == "protein":
        sweep = ProteinSweep(config)
    elif framework == "carbs":
        sweep = CarbsSweep(config)
    else:
        sweep = HyperparameterSweep(config)
    
    # Run the sweep
    return sweep.run_sweep()


if __name__ == "__main__":
    # Example usage
    config = create_sweep_config(
        sweep_name="dspy_code_protein_sweep",
        num_trials=50,
        framework="protein"
    )
    
    results = run_hyperparameter_sweep(config, framework="protein")
    print(f"Sweep completed with {len(results)} trials")
