"""
Go Orchestrator Integration for RL Training

This module integrates the Go orchestrator with the RL training system
for optimal coordination, resource management, and performance monitoring.
"""

import asyncio
import json
import time
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import aiohttp
import numpy as np

# Local imports
from .universal_pufferlib import UniversalPufferConfig
from .rust_rl_runner import RustRLConfig, OptimizedRLTrainer
from .rl_tracking import get_rl_tracker
from ..streaming.streamkit import LocalBus


@dataclass
class GoOrchestratorConfig:
    """Configuration for Go orchestrator integration"""
    
    # Go orchestrator settings
    orchestrator_path: str = "orchestrator"
    orchestrator_port: int = 8080
    orchestrator_host: str = "localhost"
    
    # Resource management
    base_limit: int = 4
    min_limit: int = 1
    max_limit: int = 32
    increase_step: int = 2
    decrease_step: int = 1
    
    # Performance thresholds
    queue_high_watermark: float = 0.8
    gpu_wait_high: float = 5.0
    error_rate_high: float = 0.1
    adaptation_interval: float = 30.0  # seconds
    
    # RL training coordination
    rl_task_priority: int = 10  # High priority for RL tasks
    batch_size: int = 64
    max_concurrent_tasks: int = 16
    
    # Monitoring
    metrics_enabled: bool = True
    health_check_interval: float = 10.0
    performance_log_interval: float = 60.0


class GoOrchestratorClient:
    """Client for Go orchestrator coordination"""
    
    def __init__(self, config: GoOrchestratorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Orchestrator process
        self.orchestrator_process: Optional[subprocess.Popen] = None
        self.orchestrator_thread: Optional[threading.Thread] = None
        
        # Performance metrics
        self.metrics = {
            'queue_depth': 0,
            'gpu_wait_time': 0.0,
            'error_rate': 0.0,
            'concurrency_limit': self.config.base_limit,
            'active_tasks': 0
        }
        
        # Start orchestrator
        self._start_orchestrator()
    
    def _start_orchestrator(self):
        """Start the Go orchestrator process"""
        try:
            # Check if Go orchestrator exists
            if not os.path.exists(self.config.orchestrator_path):
                self.logger.warning(f"Go orchestrator not found at {self.config.orchestrator_path}, falling back to Python implementation")
                self.orchestrator_process = None
                return
            
            # Start orchestrator process
            self.orchestrator_process = subprocess.Popen(
                [self.config.orchestrator_path],
                env={
                    **os.environ,
                    "ORCHESTRATOR_PORT": str(self.config.orchestrator_port),
                    "BASE_LIMIT": str(self.config.base_limit),
                    "MIN_LIMIT": str(self.config.min_limit),
                    "MAX_LIMIT": str(self.config.max_limit)
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.logger.info(f"Started Go orchestrator (PID: {self.orchestrator_process.pid})")
            
            # Start metrics collection thread
            self.orchestrator_thread = threading.Thread(target=self._collect_metrics)
            self.orchestrator_thread.daemon = True
            self.orchestrator_thread.start()
            
        except Exception as e:
            self.logger.warning(f"Failed to start Go orchestrator: {e}, falling back to Python implementation")
            self.orchestrator_process = None
    
    def _collect_metrics(self):
        """Collect metrics from Go orchestrator"""
        while self.orchestrator_process and self.orchestrator_process.poll() is None:
            try:
                # Query orchestrator metrics
                import requests
                response = requests.get(f"http://{self.config.orchestrator_host}:{self.config.orchestrator_port}/metrics")
                if response.status_code == 200:
                    metrics_data = response.json()
                    self.metrics.update(metrics_data)
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Failed to collect metrics: {e}")
                time.sleep(self.config.health_check_interval)
    
    async def submit_rl_task(self, task: Dict[str, Any]) -> str:
        """Submit an RL task to the orchestrator"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{self.config.orchestrator_host}:{self.config.orchestrator_port}/tasks",
                    json=task
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        task_id = result.get('task_id')
                        self.logger.info(f"Submitted RL task: {task_id}")
                        return task_id
                    else:
                        raise Exception(f"Failed to submit task: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to submit RL task: {e}")
            raise
    
    async def wait_for_task_completion(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for task completion"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{self.config.orchestrator_host}:{self.config.orchestrator_port}/tasks/{task_id}"
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            if result.get('status') == 'completed':
                                return result
                            elif result.get('status') == 'failed':
                                raise Exception(f"Task failed: {result.get('error')}")
                
                await asyncio.sleep(1.0)  # 1 second polling
                
            except Exception as e:
                self.logger.error(f"Error waiting for task completion: {e}")
                await asyncio.sleep(1.0)
        
        raise Exception(f"Timeout waiting for task {task_id}")
    
    async def submit_batch_tasks(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Submit a batch of RL tasks"""
        task_ids = []
        
        for task in tasks:
            try:
                task_id = await self.submit_rl_task(task)
                task_ids.append(task_id)
            except Exception as e:
                self.logger.error(f"Failed to submit batch task: {e}")
                continue
        
        self.logger.info(f"Submitted {len(task_ids)} batch tasks")
        return task_ids
    
    async def wait_for_batch_completion(self, task_ids: List[str], timeout: int = 600) -> List[Dict[str, Any]]:
        """Wait for batch task completion"""
        results = []
        
        # Wait for all tasks to complete
        for task_id in task_ids:
            try:
                result = await self.wait_for_task_completion(task_id, timeout)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {e}")
                results.append({'task_id': task_id, 'status': 'failed', 'error': str(e)})
        
        return results
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics from orchestrator"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{self.config.orchestrator_host}:{self.config.orchestrator_port}/metrics"
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return self.metrics
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return self.metrics
    
    async def adjust_concurrency(self, new_limit: int):
        """Adjust orchestrator concurrency limit"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{self.config.orchestrator_host}:{self.config.orchestrator_port}/concurrency",
                    json={'limit': new_limit}
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Adjusted concurrency limit to {new_limit}")
                    else:
                        self.logger.error(f"Failed to adjust concurrency: {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to adjust concurrency: {e}")
    
    def close(self):
        """Close the orchestrator client"""
        if self.orchestrator_process:
            self.orchestrator_process.terminate()
            self.orchestrator_process.wait()
        
        if self.orchestrator_thread:
            self.orchestrator_thread.join(timeout=5.0)
        
        self.logger.info("Go orchestrator client closed")


class CoordinatedRLTrainer:
    """RL trainer coordinated by Go orchestrator"""
    
    def __init__(
        self, 
        config: UniversalPufferConfig,
        rust_config: RustRLConfig,
        orchestrator_config: GoOrchestratorConfig,
        bus: LocalBus
    ):
        self.config = config
        self.rust_config = rust_config
        self.orchestrator_config = orchestrator_config
        self.bus = bus
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.rl_tracker = get_rl_tracker()
        self.orchestrator_client = GoOrchestratorClient(orchestrator_config)
        self.session_id = f"coordinated_session_{int(time.time())}"
        
        # Training state
        self.training_active = False
        self.episode_count = 0
        self.total_timesteps = 0
        self.performance_history = []
        
        # Start session tracking
        self._start_session_tracking()
    
    def _start_session_tracking(self):
        """Start tracking this coordinated training session"""
        session_config = {
            'framework': self.config.framework,
            'num_workers': self.config.num_workers,
            'num_gpus': self.config.num_gpus,
            'num_envs': self.config.num_envs,
            'rust_optimized': True,
            'go_orchestrated': True,
            'coordinated': True
        }
        
        self.rl_tracker.start_training_session(self.session_id, session_config)
        self.logger.info(f"Started coordinated RL training session: {self.session_id}")
    
    async def train(self):
        """Main training loop with Go orchestrator coordination"""
        self.logger.info("Starting coordinated RL training with Go orchestrator")
        self.training_active = True
        
        try:
            # Training loop
            while self.training_active and self.total_timesteps < self.config.total_timesteps:
                # Create RL training tasks
                tasks = self._create_rl_tasks()
                
                # Submit tasks to orchestrator
                task_ids = await self.orchestrator_client.submit_batch_tasks(tasks)
                
                # Wait for task completion
                results = await self.orchestrator_client.wait_for_batch_completion(task_ids)
                
                # Process results
                self._process_training_results(results)
                
                # Log metrics
                await self._log_training_metrics()
                
                # Check for convergence
                if self._check_convergence():
                    self.logger.info("Training converged")
                    break
                
                # Adaptive concurrency adjustment
                await self._adjust_concurrency()
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.training_active = False
        finally:
            self.orchestrator_client.close()
        
        self.logger.info("Coordinated RL training completed")
    
    def _create_rl_tasks(self) -> List[Dict[str, Any]]:
        """Create RL training tasks for orchestrator"""
        tasks = []
        
        # Create episode tasks
        for episode in range(self.config.num_envs):
            task = {
                'task_type': 'rl_episode',
                'episode_id': f"episode_{self.episode_count}_{episode}",
                'session_id': self.session_id,
                'config': {
                    'framework': self.config.framework,
                    'episode_length': self.config.episode_length,
                    'batch_size': self.config.batch_size,
                    'memory_limit': self.rust_config.env_memory_limit,
                    'cpu_limit': self.rust_config.env_cpu_limit
                },
                'priority': self.orchestrator_config.rl_task_priority,
                'timestamp': time.time()
            }
            tasks.append(task)
        
        return tasks
    
    def _process_training_results(self, results: List[Dict[str, Any]]):
        """Process training results from orchestrator"""
        for result in results:
            if result.get('status') == 'completed':
                episode_data = result.get('data', {})
                reward = episode_data.get('total_reward', 0.0)
                steps = episode_data.get('steps', 0)
                
                self.performance_history.append(reward)
                self.total_timesteps += steps
                self.episode_count += 1
                
                # Log episode metrics to RedDB
                if self.config.reddb_tracking:
                    episode_metrics = {
                        'reward': reward,
                        'episode_length': steps,
                        'fps': episode_data.get('fps', 0.0),
                        'memory_usage': episode_data.get('memory_usage', 0.0),
                        'cpu_usage': episode_data.get('cpu_usage', 0.0),
                        'gpu_usage': episode_data.get('gpu_usage', 0.0),
                        'convergence_score': min(reward / 100.0, 1.0),
                        'success_rate': 1.0 if reward > 0 else 0.0,
                        'orchestrator_metrics': result.get('orchestrator_metrics', {})
                    }
                    
                    self.rl_tracker.log_episode_metrics(self.session_id, self.episode_count, episode_metrics)
    
    async def _log_training_metrics(self):
        """Log training metrics to RedDB"""
        if self.config.reddb_tracking:
            # Get orchestrator metrics
            orchestrator_metrics = await self.orchestrator_client.get_system_metrics()
            
            # Update session with current metrics
            self.rl_tracker.update_training_session(self.session_id, {
                'num_episodes': self.episode_count,
                'total_timesteps': self.total_timesteps,
                'best_performance': max(self.performance_history) if self.performance_history else 0.0,
                'final_performance': self.performance_history[-1] if self.performance_history else 0.0,
                'orchestrator_metrics': orchestrator_metrics
            })
    
    async def _adjust_concurrency(self):
        """Adjust orchestrator concurrency based on performance"""
        try:
            metrics = await self.orchestrator_client.get_system_metrics()
            
            # Get current metrics
            queue_depth = metrics.get('queue_depth', 0)
            gpu_wait_time = metrics.get('gpu_wait_time', 0.0)
            error_rate = metrics.get('error_rate', 0.0)
            current_limit = metrics.get('concurrency_limit', self.orchestrator_config.base_limit)
            
            # Adaptive concurrency adjustment
            new_limit = current_limit
            
            # Increase concurrency if queue is high and GPU wait is low
            if (queue_depth > self.orchestrator_config.queue_high_watermark * 100 and 
                gpu_wait_time < self.orchestrator_config.gpu_wait_high and
                error_rate < self.orchestrator_config.error_rate_high):
                new_limit = min(current_limit + self.orchestrator_config.increase_step, 
                               self.orchestrator_config.max_limit)
            
            # Decrease concurrency if error rate is high or GPU wait is high
            elif (error_rate > self.orchestrator_config.error_rate_high or 
                  gpu_wait_time > self.orchestrator_config.gpu_wait_high):
                new_limit = max(current_limit - self.orchestrator_config.decrease_step, 
                               self.orchestrator_config.min_limit)
            
            # Apply new limit if changed
            if new_limit != current_limit:
                await self.orchestrator_client.adjust_concurrency(new_limit)
                self.logger.info(f"Adjusted concurrency from {current_limit} to {new_limit}")
                
        except Exception as e:
            self.logger.error(f"Failed to adjust concurrency: {e}")
    
    def _check_convergence(self) -> bool:
        """Check if training has converged"""
        if len(self.performance_history) < 10:
            return False
        
        # Check if recent performance is stable
        recent_performance = self.performance_history[-10:]
        performance_std = np.std(recent_performance)
        
        return performance_std < 0.01  # Very stable performance
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'training_active': self.training_active,
            'episode_count': self.episode_count,
            'total_timesteps': self.total_timesteps,
            'performance_history': self.performance_history[-10:] if self.performance_history else [],
            'framework': self.config.framework,
            'rust_optimized': True,
            'go_orchestrated': True,
            'coordinated': True,
            'session_id': self.session_id
        }


def create_coordinated_trainer(
    config: UniversalPufferConfig,
    rust_config: Optional[RustRLConfig] = None,
    orchestrator_config: Optional[GoOrchestratorConfig] = None,
    bus: Optional[LocalBus] = None
) -> CoordinatedRLTrainer:
    """Create a coordinated RL trainer using Go orchestrator"""
    
    if rust_config is None:
        rust_config = RustRLConfig()
    
    if orchestrator_config is None:
        orchestrator_config = GoOrchestratorConfig()
    
    if bus is None:
        bus = LocalBus()
    
    return CoordinatedRLTrainer(config, rust_config, orchestrator_config, bus)


async def run_coordinated_training(
    config: UniversalPufferConfig,
    rust_config: Optional[RustRLConfig] = None,
    orchestrator_config: Optional[GoOrchestratorConfig] = None,
    bus: Optional[LocalBus] = None
):
    """Run coordinated RL training with Go orchestrator"""
    
    trainer = create_coordinated_trainer(config, rust_config, orchestrator_config, bus)
    
    try:
        await trainer.train()
    except KeyboardInterrupt:
        trainer.training_active = False
        print("Training interrupted by user")
    except Exception as e:
        trainer.training_active = False
        print(f"Training failed: {e}")


if __name__ == "__main__":
    # Example usage
    config = UniversalPufferConfig(
        framework="auto",
        num_envs=32,
        num_workers=8,
        num_gpus=1,
        total_timesteps=1_000_000,
        reddb_tracking=True,
        react_dashboard=True
    )
    
    rust_config = RustRLConfig(
        num_envs=32,
        max_parallel_envs=64,
        prefetch_enabled=True,
        batch_processing=True
    )
    
    orchestrator_config = GoOrchestratorConfig(
        base_limit=8,
        max_limit=32,
        rl_task_priority=10,
        batch_size=64
    )
    
    asyncio.run(run_coordinated_training(config, rust_config, orchestrator_config))
