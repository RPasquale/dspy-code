#!/usr/bin/env python3
"""
Cloud GPU Integration System for DSPy Agent Training
Supports multiple cloud GPU platforms including Prime Intellect, RunPod, Nebius, CoreWeave, etc.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudProvider(Enum):
    """Supported cloud GPU providers."""
    PRIME_INTELLECT = "prime_intellect"
    RUNPOD = "runpod"
    NEBIUS = "nebius"
    COREWEAVE = "coreweave"
    LAMBDA_LABS = "lambda_labs"
    VESSL = "vessl"
    WEIGHTS_BIASES = "weights_biases"
    HUGGING_FACE = "hugging_face"

@dataclass
class GPUConfig:
    """GPU configuration for cloud training."""
    provider: CloudProvider
    instance_type: str
    gpu_count: int
    gpu_type: str
    memory_gb: int
    storage_gb: int
    region: str
    price_per_hour: float

@dataclass
class TrainingConfig:
    """Training configuration for cloud execution."""
    training_method: str
    module_name: str
    model_name: str
    dataset_path: str
    output_dir: str
    workspace_dir: str
    batch_size: int
    learning_rate: float
    max_steps: int
    epochs: int
    gpu_config: GPUConfig

class CloudGPUManager:
    """Manages cloud GPU resources for agent training."""
    
    def __init__(self, provider: CloudProvider, api_key: str, region: str = "us-east-1"):
        self.provider = provider
        self.api_key = api_key
        self.region = region
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        # Provider-specific configurations
        self.provider_configs = {
            CloudProvider.PRIME_INTELLECT: {
                'base_url': 'https://api.primeintellect.com/v1',
                'endpoints': {
                    'instances': '/instances',
                    'jobs': '/jobs',
                    'templates': '/templates'
                }
            },
            CloudProvider.RUNPOD: {
                'base_url': 'https://api.runpod.io/v2',
                'endpoints': {
                    'instances': '/pods',
                    'jobs': '/pods',
                    'templates': '/templates'
                }
            },
            CloudProvider.NEBIUS: {
                'base_url': 'https://api.nebius.com/v1',
                'endpoints': {
                    'instances': '/instances',
                    'jobs': '/jobs',
                    'templates': '/templates'
                }
            },
            CloudProvider.COREWEAVE: {
                'base_url': 'https://api.coreweave.com/v1',
                'endpoints': {
                    'instances': '/instances',
                    'jobs': '/jobs',
                    'templates': '/templates'
                }
            }
        }
    
    def get_available_gpus(self) -> List[GPUConfig]:
        """Get available GPU configurations from the provider."""
        config = self.provider_configs.get(self.provider)
        if not config:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        try:
            response = self.session.get(f"{config['base_url']}{config['endpoints']['instances']}")
            response.raise_for_status()
            
            gpu_data = response.json()
            return self._parse_gpu_configs(gpu_data)
            
        except Exception as e:
            logger.error(f"Failed to get available GPUs: {e}")
            return self._get_default_gpu_configs()
    
    def _parse_gpu_configs(self, data: Dict[str, Any]) -> List[GPUConfig]:
        """Parse GPU configurations from provider response."""
        configs = []
        
        if self.provider == CloudProvider.PRIME_INTELLECT:
            # Parse Prime Intellect response
            for instance in data.get('instances', []):
                configs.append(GPUConfig(
                    provider=self.provider,
                    instance_type=instance.get('type', 'gpu'),
                    gpu_count=instance.get('gpu_count', 1),
                    gpu_type=instance.get('gpu_type', 'A100'),
                    memory_gb=instance.get('memory_gb', 80),
                    storage_gb=instance.get('storage_gb', 500),
                    region=instance.get('region', self.region),
                    price_per_hour=instance.get('price_per_hour', 2.0)
                ))
        
        elif self.provider == CloudProvider.RUNPOD:
            # Parse RunPod response
            for pod in data.get('pods', []):
                configs.append(GPUConfig(
                    provider=self.provider,
                    instance_type=pod.get('name', 'gpu'),
                    gpu_count=pod.get('gpu_count', 1),
                    gpu_type=pod.get('gpu_type', 'RTX 4090'),
                    memory_gb=pod.get('memory_gb', 24),
                    storage_gb=pod.get('storage_gb', 100),
                    region=pod.get('region', self.region),
                    price_per_hour=pod.get('price_per_hour', 0.5)
                ))
        
        return configs
    
    def _get_default_gpu_configs(self) -> List[GPUConfig]:
        """Get default GPU configurations if API fails."""
        return [
            GPUConfig(
                provider=self.provider,
                instance_type="gpu-standard",
                gpu_count=1,
                gpu_type="A100",
                memory_gb=80,
                storage_gb=500,
                region=self.region,
                price_per_hour=2.0
            ),
            GPUConfig(
                provider=self.provider,
                instance_type="gpu-high-memory",
                gpu_count=2,
                gpu_type="A100",
                memory_gb=160,
                storage_gb=1000,
                region=self.region,
                price_per_hour=4.0
            )
        ]
    
    def create_training_instance(self, config: TrainingConfig) -> str:
        """Create a training instance with the specified configuration."""
        logger.info(f"Creating training instance with {config.gpu_config.provider.value}")
        
        # Prepare instance creation request
        instance_request = self._prepare_instance_request(config)
        
        try:
            response = self.session.post(
                f"{self.provider_configs[self.provider]['base_url']}"
                f"{self.provider_configs[self.provider]['endpoints']['instances']}",
                json=instance_request
            )
            response.raise_for_status()
            
            instance_data = response.json()
            instance_id = instance_data.get('id')
            
            logger.info(f"Training instance created: {instance_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to create training instance: {e}")
            raise
    
    def _prepare_instance_request(self, config: TrainingConfig) -> Dict[str, Any]:
        """Prepare instance creation request based on provider."""
        base_request = {
            'instance_type': config.gpu_config.instance_type,
            'gpu_count': config.gpu_config.gpu_count,
            'gpu_type': config.gpu_config.gpu_type,
            'memory_gb': config.gpu_config.memory_gb,
            'storage_gb': config.gpu_config.storage_gb,
            'region': config.gpu_config.region,
            'training_config': {
                'method': config.training_method,
                'module': config.module_name,
                'model': config.model_name,
                'dataset': config.dataset_path,
                'output': config.output_dir,
                'workspace': config.workspace_dir,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'max_steps': config.max_steps,
                'epochs': config.epochs
            }
        }
        
        if self.provider == CloudProvider.PRIME_INTELLECT:
            base_request.update({
                'image': 'dspy-agent:latest',
                'environment': {
                    'TRAINING_METHOD': config.training_method,
                    'MODULE_NAME': config.module_name,
                    'MODEL_NAME': config.model_name
                }
            })
        
        elif self.provider == CloudProvider.RUNPOD:
            base_request.update({
                'docker_image': 'dspy-agent:latest',
                'env': {
                    'TRAINING_METHOD': config.training_method,
                    'MODULE_NAME': config.module_name,
                    'MODEL_NAME': config.model_name
                }
            })
        
        return base_request
    
    def submit_training_job(self, instance_id: str, config: TrainingConfig) -> str:
        """Submit a training job to the instance."""
        logger.info(f"Submitting training job to instance {instance_id}")
        
        job_request = {
            'instance_id': instance_id,
            'command': self._get_training_command(config),
            'environment': self._get_training_environment(config),
            'timeout': 3600 * 24,  # 24 hours
            'restart_policy': 'on_failure'
        }
        
        try:
            response = self.session.post(
                f"{self.provider_configs[self.provider]['base_url']}"
                f"{self.provider_configs[self.provider]['endpoints']['jobs']}",
                json=job_request
            )
            response.raise_for_status()
            
            job_data = response.json()
            job_id = job_data.get('id')
            
            logger.info(f"Training job submitted: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to submit training job: {e}")
            raise
    
    def _get_training_command(self, config: TrainingConfig) -> str:
        """Get the training command based on configuration."""
        if config.training_method == 'grpo':
            return f"dspy-agent grpo train --dataset {config.dataset_path} --model {config.model_name} --out-dir {config.output_dir}"
        elif config.training_method == 'gepa':
            return f"python -m dspy_agent.training.train_gepa --module {config.module_name} --dataset {config.dataset_path}"
        elif config.training_method == 'teleprompt':
            return f"python -m dspy_agent.training.train_teleprompt --module {config.module_name} --dataset {config.dataset_path}"
        elif config.training_method == 'rl':
            return f"python -m dspy_agent.training.entrypoint --workspace {config.workspace_dir} --signature CodeContextSig"
        else:
            return f"python -m dspy_agent.training.train_{config.training_method} --dataset {config.dataset_path}"
    
    def _get_training_environment(self, config: TrainingConfig) -> Dict[str, str]:
        """Get environment variables for training."""
        return {
            'TRAINING_METHOD': config.training_method,
            'MODULE_NAME': config.module_name,
            'MODEL_NAME': config.model_name,
            'DATASET_PATH': config.dataset_path,
            'OUTPUT_DIR': config.output_dir,
            'WORKSPACE_DIR': config.workspace_dir,
            'BATCH_SIZE': str(config.batch_size),
            'LEARNING_RATE': str(config.learning_rate),
            'MAX_STEPS': str(config.max_steps),
            'EPOCHS': str(config.epochs),
            'CUDA_VISIBLE_DEVICES': '0'
        }
    
    def monitor_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor training job progress."""
        try:
            response = self.session.get(
                f"{self.provider_configs[self.provider]['base_url']}"
                f"{self.provider_configs[self.provider]['endpoints']['jobs']}/{job_id}"
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to monitor job {job_id}: {e}")
            return {'status': 'unknown', 'error': str(e)}
    
    def get_job_logs(self, job_id: str) -> str:
        """Get training job logs."""
        try:
            response = self.session.get(
                f"{self.provider_configs[self.provider]['base_url']}"
                f"{self.provider_configs[self.provider]['endpoints']['jobs']}/{job_id}/logs"
            )
            response.raise_for_status()
            
            return response.text
            
        except Exception as e:
            logger.error(f"Failed to get job logs: {e}")
            return f"Error retrieving logs: {e}"
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate the training instance."""
        try:
            response = self.session.delete(
                f"{self.provider_configs[self.provider]['base_url']}"
                f"{self.provider_configs[self.provider]['endpoints']['instances']}/{instance_id}"
            )
            response.raise_for_status()
            
            logger.info(f"Instance {instance_id} terminated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate instance {instance_id}: {e}")
            return False

class CloudTrainingOrchestrator:
    """Orchestrates cloud training across multiple providers."""
    
    def __init__(self):
        self.managers = {}
        self.active_jobs = {}
    
    def add_provider(self, provider: CloudProvider, api_key: str, region: str = "us-east-1"):
        """Add a cloud provider."""
        self.managers[provider] = CloudGPUManager(provider, api_key, region)
        logger.info(f"Added provider: {provider.value}")
    
    def find_best_gpu(self, requirements: Dict[str, Any]) -> Optional[GPUConfig]:
        """Find the best GPU configuration based on requirements."""
        best_config = None
        best_price = float('inf')
        
        for provider, manager in self.managers.items():
            try:
                gpu_configs = manager.get_available_gpus()
                
                for config in gpu_configs:
                    if self._meets_requirements(config, requirements):
                        if config.price_per_hour < best_price:
                            best_price = config.price_per_hour
                            best_config = config
                            
            except Exception as e:
                logger.warning(f"Failed to get GPUs from {provider.value}: {e}")
                continue
        
        return best_config
    
    def _meets_requirements(self, config: GPUConfig, requirements: Dict[str, Any]) -> bool:
        """Check if GPU configuration meets requirements."""
        if requirements.get('min_gpu_count', 1) > config.gpu_count:
            return False
        
        if requirements.get('min_memory_gb', 8) > config.memory_gb:
            return False
        
        if requirements.get('min_storage_gb', 100) > config.storage_gb:
            return False
        
        if requirements.get('max_price_per_hour', float('inf')) < config.price_per_hour:
            return False
        
        return True
    
    def start_training(self, config: TrainingConfig, requirements: Dict[str, Any]) -> str:
        """Start training on the best available GPU."""
        # Find best GPU
        best_gpu = self.find_best_gpu(requirements)
        if not best_gpu:
            raise RuntimeError("No suitable GPU found for training")
        
        logger.info(f"Using GPU: {best_gpu.provider.value} - {best_gpu.gpu_type} x{best_gpu.gpu_count}")
        
        # Create training instance
        manager = self.managers[best_gpu.provider]
        instance_id = manager.create_training_instance(config)
        
        # Submit training job
        job_id = manager.submit_training_job(instance_id, config)
        
        # Store job info
        self.active_jobs[job_id] = {
            'provider': best_gpu.provider,
            'instance_id': instance_id,
            'manager': manager,
            'config': config
        }
        
        return job_id
    
    def monitor_training(self, job_id: str) -> Dict[str, Any]:
        """Monitor training progress."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Unknown job ID: {job_id}")
        
        job_info = self.active_jobs[job_id]
        manager = job_info['manager']
        
        return manager.monitor_job(job_id)
    
    def get_training_logs(self, job_id: str) -> str:
        """Get training logs."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Unknown job ID: {job_id}")
        
        job_info = self.active_jobs[job_id]
        manager = job_info['manager']
        
        return manager.get_job_logs(job_id)
    
    def stop_training(self, job_id: str) -> bool:
        """Stop training and clean up resources."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Unknown job ID: {job_id}")
        
        job_info = self.active_jobs[job_id]
        manager = job_info['manager']
        instance_id = job_info['instance_id']
        
        # Terminate instance
        success = manager.terminate_instance(instance_id)
        
        # Remove from active jobs
        del self.active_jobs[job_id]
        
        return success

def main():
    parser = argparse.ArgumentParser(description="Cloud GPU Training for DSPy Agent")
    parser.add_argument('--provider', required=True, choices=[p.value for p in CloudProvider])
    parser.add_argument('--api-key', required=True, help='API key for the cloud provider')
    parser.add_argument('--region', default='us-east-1', help='Region for the cloud provider')
    parser.add_argument('--training-method', required=True, 
                       choices=['grpo', 'gepa', 'teleprompt', 'rl', 'codegen', 'orchestrator', 'prefs'])
    parser.add_argument('--module-name', default='orchestrator')
    parser.add_argument('--model-name', default='gpt2')
    parser.add_argument('--dataset-path', default='/workspace/datasets/training.jsonl')
    parser.add_argument('--output-dir', default='/workspace/models/agent')
    parser.add_argument('--workspace-dir', default='/workspace')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--min-gpu-count', type=int, default=1)
    parser.add_argument('--min-memory-gb', type=int, default=8)
    parser.add_argument('--min-storage-gb', type=int, default=100)
    parser.add_argument('--max-price-per-hour', type=float, default=10.0)
    parser.add_argument('--monitor', action='store_true', help='Monitor training progress')
    parser.add_argument('--logs', action='store_true', help='Show training logs')
    parser.add_argument('--stop', help='Stop training job by ID')
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = CloudTrainingOrchestrator()
    
    # Add provider
    provider = CloudProvider(args.provider)
    orchestrator.add_provider(provider, args.api_key, args.region)
    
    if args.stop:
        # Stop training job
        success = orchestrator.stop_training(args.stop)
        if success:
            print(f"Training job {args.stop} stopped successfully")
        else:
            print(f"Failed to stop training job {args.stop}")
        return
    
    # Create training configuration
    gpu_config = GPUConfig(
        provider=provider,
        instance_type="gpu-standard",
        gpu_count=1,
        gpu_type="A100",
        memory_gb=80,
        storage_gb=500,
        region=args.region,
        price_per_hour=2.0
    )
    
    training_config = TrainingConfig(
        training_method=args.training_method,
        module_name=args.module_name,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        workspace_dir=args.workspace_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        epochs=args.epochs,
        gpu_config=gpu_config
    )
    
    # Define requirements
    requirements = {
        'min_gpu_count': args.min_gpu_count,
        'min_memory_gb': args.min_memory_gb,
        'min_storage_gb': args.min_storage_gb,
        'max_price_per_hour': args.max_price_per_hour
    }
    
    try:
        # Start training
        job_id = orchestrator.start_training(training_config, requirements)
        print(f"Training started with job ID: {job_id}")
        
        if args.monitor:
            # Monitor training
            while True:
                status = orchestrator.monitor_training(job_id)
                print(f"Job status: {status.get('status', 'unknown')}")
                
                if status.get('status') in ['completed', 'failed', 'stopped']:
                    break
                
                time.sleep(30)  # Check every 30 seconds
        
        if args.logs:
            # Show logs
            logs = orchestrator.get_training_logs(job_id)
            print("Training logs:")
            print(logs)
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
