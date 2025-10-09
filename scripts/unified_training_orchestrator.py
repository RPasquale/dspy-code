#!/usr/bin/env python3
"""
Unified Training Orchestrator for DSPy Agent
Supports both local Slurm clusters and cloud GPU platforms (Prime Intellect, RunPod, etc.)
Automatically detects environment and uses appropriate backend.
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

class TrainingBackend(Enum):
    """Available training backends."""
    LOCAL_SLURM = "local_slurm"
    PRIME_INTELLECT = "prime_intellect"
    RUNPOD = "runpod"
    NEBIUS = "nebius"
    COREWEAVE = "coreweave"
    AUTO_DETECT = "auto_detect"

@dataclass
class TrainingRequest:
    """Training request configuration."""
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
    gpu_requirements: Dict[str, Any]
    backend: TrainingBackend = TrainingBackend.AUTO_DETECT

@dataclass
class TrainingResult:
    """Training result information."""
    job_id: str
    backend: TrainingBackend
    status: str
    instance_id: Optional[str] = None
    logs_url: Optional[str] = None
    output_url: Optional[str] = None
    cost: Optional[float] = None

class EnvironmentDetector:
    """Detects available training environments."""
    
    @staticmethod
    def detect_available_backends() -> List[TrainingBackend]:
        """Detect available training backends."""
        backends = []
        
        # Check for local Slurm
        if EnvironmentDetector._check_slurm():
            backends.append(TrainingBackend.LOCAL_SLURM)
        
        # Check for cloud providers
        if EnvironmentDetector._check_prime_intellect():
            backends.append(TrainingBackend.PRIME_INTELLECT)
        
        if EnvironmentDetector._check_runpod():
            backends.append(TrainingBackend.RUNPOD)
        
        if EnvironmentDetector._check_nebius():
            backends.append(TrainingBackend.NEBIUS)
        
        if EnvironmentDetector._check_coreweave():
            backends.append(TrainingBackend.COREWEAVE)
        
        return backends
    
    @staticmethod
    def _check_slurm() -> bool:
        """Check if Slurm is available locally."""
        try:
            result = subprocess.run(['sinfo', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @staticmethod
    def _check_prime_intellect() -> bool:
        """Check if Prime Intellect API is available."""
        api_key = os.getenv('PRIME_INTELLECT_API_KEY')
        return api_key is not None
    
    @staticmethod
    def _check_runpod() -> bool:
        """Check if RunPod API is available."""
        api_key = os.getenv('RUNPOD_API_KEY')
        return api_key is not None
    
    @staticmethod
    def _check_nebius() -> bool:
        """Check if Nebius API is available."""
        api_key = os.getenv('NEBIUS_API_KEY')
        return api_key is not None
    
    @staticmethod
    def _check_coreweave() -> bool:
        """Check if CoreWeave API is available."""
        api_key = os.getenv('COREWEAVE_API_KEY')
        return api_key is not None

class LocalSlurmBackend:
    """Local Slurm backend for training."""
    
    def __init__(self):
        self.slurm_scripts_dir = Path("deploy/slurm")
        self.ensure_slurm_scripts()
    
    def ensure_slurm_scripts(self):
        """Ensure Slurm scripts are available."""
        if not self.slurm_scripts_dir.exists():
            logger.warning("Slurm scripts directory not found, creating basic scripts...")
            self.slurm_scripts_dir.mkdir(parents=True, exist_ok=True)
            self._create_basic_slurm_script()
    
    def _create_basic_slurm_script(self):
        """Create basic Slurm script if not available."""
        script_path = self.slurm_scripts_dir / "train_agent_basic.sbatch"
        
        script_content = '''#!/bin/bash
#SBATCH --job-name=agent_training
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=/tmp/agent_training_%j.out
#SBATCH --error=/tmp/agent_training_%j.err

# Set environment variables
export TRAINING_METHOD=${TRAINING_METHOD:-grpo}
export MODULE_NAME=${MODULE_NAME:-orchestrator}
export MODEL_NAME=${MODEL_NAME:-gpt2}
export DATASET_PATH=${DATASET_PATH:-/tmp/datasets/training.jsonl}
export OUTPUT_DIR=${OUTPUT_DIR:-/tmp/models/agent}
export WORKSPACE_DIR=${WORKSPACE_DIR:-/tmp}

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $(dirname $DATASET_PATH)

# Run training
python -m dspy_agent.training.entrypoint \\
    --workspace $WORKSPACE_DIR \\
    --signature CodeContextSig \\
    --steps 200 \\
    --env production
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
    
    def submit_job(self, request: TrainingRequest) -> TrainingResult:
        """Submit job to local Slurm."""
        logger.info("Submitting job to local Slurm...")
        
        # Prepare environment variables
        env = os.environ.copy()
        env.update({
            'TRAINING_METHOD': request.training_method,
            'MODULE_NAME': request.module_name,
            'MODEL_NAME': request.model_name,
            'DATASET_PATH': request.dataset_path,
            'OUTPUT_DIR': request.output_dir,
            'WORKSPACE_DIR': request.workspace_dir,
            'BATCH_SIZE': str(request.batch_size),
            'LEARNING_RATE': str(request.learning_rate),
            'MAX_STEPS': str(request.max_steps),
            'EPOCHS': str(request.epochs),
            'LOG_DIR': os.path.join(request.workspace_dir, 'logs')
        })

        if request.training_method == 'rl':
            env.setdefault('RL_RESULTS_TOPIC', f"rl.results.{request.module_name}")
            env.setdefault('RL_BUFFER_DIR', os.path.join(request.workspace_dir, 'logs', 'rl'))
            env.setdefault('RL_WORKSPACE_DIR', request.workspace_dir)

        if request.training_method == 'rl':
            env.setdefault('RL_STEPS', str(request.max_steps or 1000))
            env.setdefault('RL_N_ENVS', str(request.gpu_requirements.get('gpu_count', 1)))
            env.setdefault('RL_LR', str(request.learning_rate))
            env.setdefault('RL_ENTROPY', '0.01')
            env.setdefault('RL_REPLAY_CAPACITY', '4096')
            env.setdefault('RL_REPLAY_BATCH', '256')
            env.setdefault('RL_GRAD_CLIP', '1.0')
            env.setdefault('RL_LOG_INTERVAL', '10')
            env.setdefault('RL_SKIP_GEPA', '0')
            env.setdefault('RL_GEPA_MODULES', '')
            env.setdefault('RL_LOG_JSONL', os.path.join(request.output_dir, f"rl_{int(time.time())}.jsonl"))

        script_name = 'train_agent_basic.sbatch'
        if request.training_method == 'rl':
            script_name = 'train_puffer_rl.sbatch'

        # Submit job
        try:
            result = subprocess.run(
                ['sbatch', str(self.slurm_scripts_dir / script_name)],
                env=env,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Slurm submission failed: {result.stderr}")
            
            # Parse job ID from output
            job_id = result.stdout.strip().split()[-1]
            
            return TrainingResult(
                job_id=job_id,
                backend=TrainingBackend.LOCAL_SLURM,
                status="submitted"
            )
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Slurm submission timed out")
        except Exception as e:
            raise RuntimeError(f"Failed to submit Slurm job: {e}")
    
    def monitor_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor Slurm job."""
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '--format=%i,%T,%R', '--noheader'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return {'status': 'unknown', 'error': result.stderr}
            
            lines = result.stdout.strip().split('\n')
            if not lines or lines[0] == '':
                return {'status': 'completed'}
            
            parts = lines[0].split(',')
            if len(parts) >= 2:
                return {
                    'status': parts[1].strip(),
                    'reason': parts[2].strip() if len(parts) > 2 else None
                }
            
            return {'status': 'unknown'}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def get_job_logs(self, job_id: str) -> str:
        """Get Slurm job logs."""
        try:
            # Try to find log files
            log_patterns = [
                f"/tmp/agent_training_{job_id}.out",
                f"/tmp/agent_training_{job_id}.err",
                f"/workspace/logs/agent_training_{job_id}.out"
            ]
            
            logs = []
            for pattern in log_patterns:
                if os.path.exists(pattern):
                    with open(pattern, 'r') as f:
                        logs.append(f"=== {pattern} ===\n{f.read()}")
            
            if logs:
                return '\n'.join(logs)
            else:
                return f"No log files found for job {job_id}"
                
        except Exception as e:
            return f"Error retrieving logs: {e}"

class PrimeIntellectBackend:
    """Prime Intellect cloud backend for training."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.primeintellect.com/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def submit_job(self, request: TrainingRequest) -> TrainingResult:
        """Submit job to Prime Intellect."""
        logger.info("Submitting job to Prime Intellect...")
        
        # Create cluster request
        cluster_request = self._prepare_cluster_request(request)
        
        try:
            response = self.session.post(f"{self.base_url}/clusters", json=cluster_request)
            response.raise_for_status()
            
            cluster_data = response.json()
            cluster_id = cluster_data.get('id')
            
            # Submit training job
            job_request = self._prepare_job_request(request, cluster_id)
            
            response = self.session.post(f"{self.base_url}/jobs", json=job_request)
            response.raise_for_status()
            
            job_data = response.json()
            job_id = job_data.get('id')
            
            return TrainingResult(
                job_id=job_id,
                backend=TrainingBackend.PRIME_INTELLECT,
                status="submitted",
                instance_id=cluster_id
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to submit Prime Intellect job: {e}")
    
    def _prepare_cluster_request(self, request: TrainingRequest) -> Dict[str, Any]:
        """Prepare Prime Intellect cluster request."""
        # Map GPU requirements to Prime Intellect GPU types
        gpu_type = self._map_gpu_requirements(request.gpu_requirements)
        
        return {
            'name': f'dspy-agent-training-{int(time.time())}',
            'image': 'ubuntu_22_cuda_12',
            'gpu_type': gpu_type,
            'gpu_count': request.gpu_requirements.get('gpu_count', 1),
            'location': 'Cheapest',
            'security': 'Cheapest',
            'spot': True,
            'environment': {
                'TRAINING_METHOD': request.training_method,
                'MODULE_NAME': request.module_name,
                'MODEL_NAME': request.model_name,
                'DATASET_PATH': request.dataset_path,
                'OUTPUT_DIR': request.output_dir,
                'WORKSPACE_DIR': request.workspace_dir,
                'BATCH_SIZE': str(request.batch_size),
                'LEARNING_RATE': str(request.learning_rate),
                'MAX_STEPS': str(request.max_steps),
                'EPOCHS': str(request.epochs)
            }
        }
    
    def _map_gpu_requirements(self, requirements: Dict[str, Any]) -> str:
        """Map GPU requirements to Prime Intellect GPU types."""
        min_memory = requirements.get('min_memory_gb', 8)
        gpu_count = requirements.get('gpu_count', 1)
        
        # Map based on memory requirements
        if min_memory >= 80:
            return 'H100' if gpu_count > 1 else 'A100'
        elif min_memory >= 48:
            return 'RTX6000'
        elif min_memory >= 24:
            return 'RTX4090'
        elif min_memory >= 16:
            return 'RTX4080'
        else:
            return 'RTX4070'
    
    def _prepare_job_request(self, request: TrainingRequest, cluster_id: str) -> Dict[str, Any]:
        """Prepare Prime Intellect job request."""
        return {
            'cluster_id': cluster_id,
            'command': self._get_training_command(request),
            'timeout': 3600 * 24,  # 24 hours
            'restart_policy': 'on_failure'
        }
    
    def _get_training_command(self, request: TrainingRequest) -> str:
        """Get training command based on method."""
        if request.training_method == 'grpo':
            return f"dspy-agent grpo train --dataset {request.dataset_path} --model {request.model_name} --out-dir {request.output_dir}"
        elif request.training_method == 'gepa':
            return f"python -m dspy_agent.training.train_gepa --module {request.module_name} --dataset {request.dataset_path}"
        elif request.training_method == 'teleprompt':
            return f"python -m dspy_agent.training.train_teleprompt --module {request.module_name} --dataset {request.dataset_path}"
        elif request.training_method == 'rl':
            return f"python -m dspy_agent.training.entrypoint --workspace {request.workspace_dir} --signature CodeContextSig"
        else:
            return f"python -m dspy_agent.training.train_{request.training_method} --dataset {request.dataset_path}"
    
    def monitor_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor Prime Intellect job."""
        try:
            response = self.session.get(f"{self.base_url}/jobs/{job_id}")
            response.raise_for_status()
            
            job_data = response.json()
            return {
                'status': job_data.get('status', 'unknown'),
                'progress': job_data.get('progress', 0),
                'cost': job_data.get('cost', 0)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def get_job_logs(self, job_id: str) -> str:
        """Get Prime Intellect job logs."""
        try:
            response = self.session.get(f"{self.base_url}/jobs/{job_id}/logs")
            response.raise_for_status()
            
            return response.text
            
        except Exception as e:
            return f"Error retrieving logs: {e}"

class UnifiedTrainingOrchestrator:
    """Unified orchestrator that can use any available backend."""
    
    def __init__(self):
        self.backends = {}
        self.active_jobs = {}
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize available backends."""
        available_backends = EnvironmentDetector.detect_available_backends()
        
        for backend in available_backends:
            if backend == TrainingBackend.LOCAL_SLURM:
                self.backends[backend] = LocalSlurmBackend()
            elif backend == TrainingBackend.PRIME_INTELLECT:
                api_key = os.getenv('PRIME_INTELLECT_API_KEY')
                if api_key:
                    self.backends[backend] = PrimeIntellectBackend(api_key)
        
        logger.info(f"Initialized backends: {list(self.backends.keys())}")
    
    def get_available_backends(self) -> List[TrainingBackend]:
        """Get list of available backends."""
        return list(self.backends.keys())
    
    def get_best_backend(self, request: TrainingRequest) -> TrainingBackend:
        """Get the best backend for the training request."""
        if request.backend != TrainingBackend.AUTO_DETECT:
            if request.backend in self.backends:
                return request.backend
            else:
                raise ValueError(f"Backend {request.backend.value} not available")
        
        # Auto-detect best backend
        available = self.get_available_backends()
        
        if not available:
            raise RuntimeError("No training backends available")
        
        # Prefer local Slurm if available
        if TrainingBackend.LOCAL_SLURM in available:
            return TrainingBackend.LOCAL_SLURM
        
        # Otherwise use first available cloud backend
        return available[0]
    
    def start_training(self, request: TrainingRequest) -> TrainingResult:
        """Start training using the best available backend."""
        backend = self.get_best_backend(request)
        logger.info(f"Using backend: {backend.value}")
        
        # Submit job
        result = self.backends[backend].submit_job(request)
        
        # Store job info
        self.active_jobs[result.job_id] = {
            'backend': backend,
            'request': request,
            'result': result
        }
        
        return result
    
    def monitor_training(self, job_id: str) -> Dict[str, Any]:
        """Monitor training progress."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Unknown job ID: {job_id}")
        
        job_info = self.active_jobs[job_id]
        backend = job_info['backend']
        
        return self.backends[backend].monitor_job(job_id)
    
    def get_training_logs(self, job_id: str) -> str:
        """Get training logs."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Unknown job ID: {job_id}")
        
        job_info = self.active_jobs[job_id]
        backend = job_info['backend']
        
        return self.backends[backend].get_job_logs(job_id)
    
    def stop_training(self, job_id: str) -> bool:
        """Stop training job."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Unknown job ID: {job_id}")
        
        job_info = self.active_jobs[job_id]
        backend = job_info['backend']
        
        # For now, just remove from active jobs
        # In a real implementation, you'd cancel the actual job
        del self.active_jobs[job_id]
        
        logger.info(f"Stopped training job {job_id}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Unified Training Orchestrator for DSPy Agent")
    parser.add_argument('--training-method', required=True,
                       choices=['grpo', 'gepa', 'teleprompt', 'rl', 'codegen', 'orchestrator', 'prefs'])
    parser.add_argument('--module-name', default='orchestrator')
    parser.add_argument('--model-name', default='gpt2')
    parser.add_argument('--dataset-path', default='/tmp/datasets/training.jsonl')
    parser.add_argument('--output-dir', default='/tmp/models/agent')
    parser.add_argument('--workspace-dir', default='/tmp')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--backend', default='auto_detect',
                       choices=['auto_detect', 'local_slurm', 'prime_intellect', 'runpod', 'nebius', 'coreweave'])
    parser.add_argument('--gpu-count', type=int, default=1)
    parser.add_argument('--min-memory-gb', type=int, default=8)
    parser.add_argument('--min-storage-gb', type=int, default=100)
    parser.add_argument('--max-price-per-hour', type=float, default=10.0)
    parser.add_argument('--monitor', action='store_true', help='Monitor training progress')
    parser.add_argument('--logs', action='store_true', help='Show training logs')
    parser.add_argument('--stop', help='Stop training job by ID')
    parser.add_argument('--list-backends', action='store_true', help='List available backends')
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = UnifiedTrainingOrchestrator()
    
    if args.list_backends:
        backends = orchestrator.get_available_backends()
        print("Available backends:")
        for backend in backends:
            print(f"  - {backend.value}")
        return
    
    if args.stop:
        success = orchestrator.stop_training(args.stop)
        if success:
            print(f"Training job {args.stop} stopped successfully")
        else:
            print(f"Failed to stop training job {args.stop}")
        return
    
    # Create training request
    request = TrainingRequest(
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
        gpu_requirements={
            'gpu_count': args.gpu_count,
            'min_memory_gb': args.min_memory_gb,
            'min_storage_gb': args.min_storage_gb,
            'max_price_per_hour': args.max_price_per_hour
        },
        backend=TrainingBackend(args.backend)
    )
    
    try:
        # Start training
        result = orchestrator.start_training(request)
        print(f"Training started:")
        print(f"  Job ID: {result.job_id}")
        print(f"  Backend: {result.backend.value}")
        print(f"  Status: {result.status}")
        
        if args.monitor:
            # Monitor training
            while True:
                status = orchestrator.monitor_training(result.job_id)
                print(f"Job status: {status.get('status', 'unknown')}")
                
                if status.get('status') in ['completed', 'failed', 'stopped']:
                    break
                
                time.sleep(30)  # Check every 30 seconds
        
        if args.logs:
            # Show logs
            logs = orchestrator.get_training_logs(result.job_id)
            print("Training logs:")
            print(logs)
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
