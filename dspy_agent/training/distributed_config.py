"""
Distributed Training Configuration for DSPy-Code Agent

This module provides comprehensive distributed training capabilities using PufferLib,
supporting scaling from single GPU to massive distributed clusters.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import torch
import torch.distributed as dist
from pufferlib import policy, environment, training
from pufferlib.frameworks import cleanrl, ray, protein, carbs
from pufferlib.utils import setup_logging


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training of DSPy-Code agent."""
    
    # Training Framework
    framework: str = "protein"  # protein, carbs, ray, cleanrl
    num_envs: int = 64
    num_workers: int = 8
    num_gpus: int = 1
    num_cpus: int = 16
    
    # PufferLib Features
    use_protein: bool = True
    use_carbs: bool = True
    use_ray: bool = False
    use_cleanrl: bool = False
    
    # Training Parameters
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Agent-Specific Parameters
    agent_type: str = "dspy_code"
    task_type: str = "software_engineering"
    reward_shaping: bool = True
    curriculum_learning: bool = True
    
    # Distributed Training
    distributed: bool = True
    backend: str = "nccl"  # nccl, gloo
    master_addr: str = "localhost"
    master_port: int = 29500
    world_size: int = 1
    rank: int = 0
    
    # Scaling Configuration
    auto_scaling: bool = True
    min_workers: int = 1
    max_workers: int = 100
    scaling_threshold: float = 0.8
    
    # Performance Optimization
    mixed_precision: bool = True
    compile_model: bool = True
    gradient_checkpointing: bool = True
    dataloader_workers: int = 4
    pin_memory: bool = True
    
    # Monitoring and Logging
    log_dir: str = "logs/distributed_training"
    tensorboard: bool = True
    wandb: bool = True
    wandb_project: str = "dspy-code-distributed"
    log_interval: int = 100
    save_interval: int = 1000
    
    # Environment Configuration
    env_config: Dict[str, Any] = field(default_factory=dict)
    policy_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Set up logging directory
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure environment based on available resources
        self._configure_resources()
        
        # Set up distributed training if enabled
        if self.distributed:
            self._setup_distributed()
    
    def _configure_resources(self):
        """Configure resources based on available hardware."""
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.num_cpus = os.cpu_count() or 16
        else:
            self.num_gpus = 0
            self.num_cpus = os.cpu_count() or 8
        
        # Adjust workers based on available resources
        if self.num_gpus > 0:
            self.num_workers = min(self.num_workers, self.num_gpus * 8)
        else:
            self.num_workers = min(self.num_workers, self.num_cpus // 2)
    
    def _setup_distributed(self):
        """Set up distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                init_method=f"tcp://{self.master_addr}:{self.master_port}",
                world_size=self.world_size,
                rank=self.rank
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        path = Path(path)
        with path.open('w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DistributedTrainingConfig':
        """Load configuration from file."""
        path = Path(path)
        with path.open('r') as f:
            data = json.load(f)
        return cls(**data)


class ScalingManager:
    """Manages automatic scaling of distributed training."""
    
    def __init__(self, config: DistributedTrainingConfig):
        self.config = config
        self.current_workers = config.num_workers
        self.performance_history = []
    
    def should_scale_up(self, performance_metric: float) -> bool:
        """Determine if we should scale up based on performance."""
        if self.current_workers >= self.config.max_workers:
            return False
        
        if performance_metric > self.config.scaling_threshold:
            return True
        
        return False
    
    def should_scale_down(self, performance_metric: float) -> bool:
        """Determine if we should scale down based on performance."""
        if self.current_workers <= self.config.min_workers:
            return False
        
        if performance_metric < 0.3:  # Low utilization
            return True
        
        return False
    
    def scale_workers(self, new_worker_count: int):
        """Scale the number of workers."""
        if self.config.framework == "protein":
            self._scale_protein_workers(new_worker_count)
        elif self.config.framework == "carbs":
            self._scale_carbs_workers(new_worker_count)
        elif self.config.framework == "ray":
            self._scale_ray_workers(new_worker_count)
    
    def _scale_protein_workers(self, new_count: int):
        """Scale Protein workers."""
        # Protein-specific scaling logic
        pass
    
    def _scale_carbs_workers(self, new_count: int):
        """Scale Carbs workers."""
        # Carbs-specific scaling logic
        pass
    
    def _scale_ray_workers(self, new_count: int):
        """Scale Ray workers."""
        # Ray-specific scaling logic
        pass


def create_training_config(
    framework: str = "protein",
    num_gpus: int = 1,
    distributed: bool = True,
    **kwargs
) -> DistributedTrainingConfig:
    """Create a training configuration optimized for the given setup."""
    
    # Base configuration
    config = DistributedTrainingConfig(
        framework=framework,
        distributed=distributed,
        **kwargs
    )
    
    # Optimize for single GPU (4090)
    if num_gpus == 1:
        config.num_workers = 8
        config.num_envs = 32
        config.batch_size = 32
        config.learning_rate = 3e-4
        config.mixed_precision = True
        config.compile_model = True
    
    # Optimize for multi-GPU
    elif num_gpus > 1:
        config.num_workers = num_gpus * 8
        config.num_envs = num_gpus * 32
        config.batch_size = 64
        config.learning_rate = 3e-4
        config.distributed = True
    
    # Optimize for large clusters
    elif num_gpus > 8:
        config.num_workers = num_gpus * 16
        config.num_envs = num_gpus * 64
        config.batch_size = 128
        config.learning_rate = 1e-4
        config.auto_scaling = True
        config.max_workers = 100
    
    return config


def get_optimal_config_for_hardware() -> DistributedTrainingConfig:
    """Get optimal configuration based on available hardware."""
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        
        # Optimize for specific GPU types
        if "4090" in gpu_name:
            return create_training_config(
                framework="protein",
                num_gpus=1,
                num_workers=8,
                num_envs=32,
                batch_size=32,
                mixed_precision=True,
                compile_model=True
            )
        elif "A100" in gpu_name:
            return create_training_config(
                framework="protein",
                num_gpus=num_gpus,
                num_workers=num_gpus * 16,
                num_envs=num_gpus * 64,
                batch_size=128,
                mixed_precision=True
            )
        else:
            return create_training_config(
                framework="protein",
                num_gpus=num_gpus,
                distributed=True
            )
    else:
        # CPU-only configuration
        return create_training_config(
            framework="cleanrl",
            num_gpus=0,
            distributed=False,
            num_workers=4,
            num_envs=16
        )
