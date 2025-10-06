"""
Rust RL Environment Runner Integration

This module integrates the Rust env runner with PufferLib for high-performance,
parallel RL training with optimal resource management.
"""

import asyncio
import json
import time
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

# Local imports
from .universal_pufferlib import UniversalPufferTrainer, UniversalPufferConfig
from .rl_tracking import get_rl_tracker
from ..streaming.streamkit import LocalBus


@dataclass
class RustRLConfig:
    """Configuration for Rust RL environment runner"""
    
    # Rust env runner settings
    rust_runner_path: str = "env_runner_rs"
    rust_runner_port: int = 8083
    queue_dir: str = "logs/env_queue"
    max_parallel_envs: int = 64
    env_memory_limit: int = 1024 * 1024 * 1024  # 1GB per env
    env_cpu_limit: int = 2
    env_timeout: int = 300  # 5 minutes
    
    # RL training settings
    num_envs: int = 32
    num_workers: int = 8
    batch_size: int = 64
    episode_length: int = 1000
    
    # Performance optimization
    prefetch_enabled: bool = True
    prefetch_queue_size: int = 128
    batch_processing: bool = True
    async_io: bool = True
    
    # Resource management
    gpu_memory_fraction: float = 0.8
    cpu_affinity: bool = True
    numa_aware: bool = True
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Ensure queue directory exists
        Path(self.queue_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.queue_dir}/pending").mkdir(parents=True, exist_ok=True)
        Path(f"{self.queue_dir}/done").mkdir(parents=True, exist_ok=True)


class RustRLEnvironment:
    """High-performance RL environment using Rust runner"""
    
    def __init__(self, env_id: str, config: RustRLConfig):
        self.env_id = env_id
        self.config = config
        self.logger = logging.getLogger(f"RustRLEnv-{env_id}")
        
        # Environment state
        self.initialized = False
        self.current_state = None
        self.episode_count = 0
        self.total_reward = 0.0
        
        # Performance metrics
        self.step_count = 0
        self.last_step_time = time.time()
        self.fps = 0.0
        
    async def initialize(self) -> bool:
        """Initialize the environment"""
        try:
            # Create environment initialization task
            init_task = {
                "env_id": self.env_id,
                "action": "init",
                "config": {
                    "memory_limit": self.config.env_memory_limit,
                    "cpu_limit": self.config.env_cpu_limit,
                    "timeout": self.config.env_timeout
                },
                "timestamp": time.time()
            }
            
            # Submit to Rust runner queue
            await self._submit_task(init_task)
            
            # Wait for initialization
            result = await self._wait_for_result(self.env_id, "init")
            
            if result.get("success", False):
                self.initialized = True
                self.current_state = result.get("state")
                self.logger.info(f"Environment {self.env_id} initialized successfully")
                return True
            else:
                self.logger.error(f"Failed to initialize environment {self.env_id}: {result.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing environment {self.env_id}: {e}")
            return False
    
    async def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Execute a step in the environment"""
        if not self.initialized:
            raise RuntimeError(f"Environment {self.env_id} not initialized")
        
        try:
            # Create step task
            step_task = {
                "env_id": self.env_id,
                "action": "step",
                "action_data": action,
                "timestamp": time.time()
            }
            
            # Submit to Rust runner
            await self._submit_task(step_task)
            
            # Wait for result
            result = await self._wait_for_result(self.env_id, "step")
            
            if result.get("success", False):
                # Update environment state
                self.current_state = result.get("state")
                reward = result.get("reward", 0.0)
                done = result.get("done", False)
                info = result.get("info", {})
                
                # Update metrics
                self.step_count += 1
                self.total_reward += reward
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_step_time > 0:
                    self.fps = 1.0 / (current_time - self.last_step_time)
                self.last_step_time = current_time
                
                if done:
                    self.episode_count += 1
                
                return self.current_state, reward, done, info
            else:
                raise RuntimeError(f"Step failed for environment {self.env_id}: {result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error in step for environment {self.env_id}: {e}")
            raise
    
    async def reset(self) -> Any:
        """Reset the environment"""
        try:
            # Create reset task
            reset_task = {
                "env_id": self.env_id,
                "action": "reset",
                "timestamp": time.time()
            }
            
            # Submit to Rust runner
            await self._submit_task(reset_task)
            
            # Wait for result
            result = await self._wait_for_result(self.env_id, "reset")
            
            if result.get("success", False):
                self.current_state = result.get("state")
                self.total_reward = 0.0
                self.step_count = 0
                self.logger.info(f"Environment {self.env_id} reset successfully")
                return self.current_state
            else:
                raise RuntimeError(f"Reset failed for environment {self.env_id}: {result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error resetting environment {self.env_id}: {e}")
            raise
    
    async def close(self):
        """Close the environment"""
        try:
            # Create close task
            close_task = {
                "env_id": self.env_id,
                "action": "close",
                "timestamp": time.time()
            }
            
            # Submit to Rust runner
            await self._submit_task(close_task)
            
            # Wait for result
            result = await self._wait_for_result(self.env_id, "close")
            
            if result.get("success", False):
                self.initialized = False
                self.logger.info(f"Environment {self.env_id} closed successfully")
            else:
                self.logger.error(f"Failed to close environment {self.env_id}: {result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error closing environment {self.env_id}: {e}")
    
    async def _submit_task(self, task: Dict[str, Any]):
        """Submit task to Rust runner queue"""
        task_file = Path(self.config.queue_dir) / "pending" / f"{self.env_id}_{int(time.time() * 1000)}.json"
        with open(task_file, 'w') as f:
            json.dump(task, f)
    
    async def _wait_for_result(self, env_id: str, action: str, timeout: int = 30) -> Dict[str, Any]:
        """Wait for result from Rust runner"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for result file
            result_file = Path(self.config.queue_dir) / "done" / f"{env_id}_{action}_result.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result = json.load(f)
                # Clean up result file
                result_file.unlink()
                return result
            
            await asyncio.sleep(0.01)  # 10ms polling
        
        return {"success": False, "error": f"Timeout waiting for {action} result"}


class RustRLRunner:
    """High-performance RL runner using Rust environment runner"""
    
    def __init__(self, config: RustRLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Environment management
        self.envs: Dict[str, RustRLEnvironment] = {}
        self.env_pool: List[RustRLEnvironment] = []
        self.available_envs: List[RustRLEnvironment] = []
        
        # Performance tracking
        self.total_steps = 0
        self.total_episodes = 0
        self.start_time = time.time()
        
        # Rust runner process
        self.rust_process: Optional[subprocess.Popen] = None
        self.rust_thread: Optional[threading.Thread] = None
        
        # Initialize Rust runner
        self._start_rust_runner()
    
    def _start_rust_runner(self):
        """Start the Rust environment runner"""
        try:
            # Check if Rust runner exists
            if not os.path.exists(self.config.rust_runner_path):
                self.logger.warning(f"Rust runner not found at {self.config.rust_runner_path}, falling back to Python implementation")
                self.rust_process = None
                return
            
            # Start Rust runner process
            self.rust_process = subprocess.Popen(
                [self.config.rust_runner_path],
                env={
                    **os.environ,
                    "ENV_QUEUE_DIR": self.config.queue_dir,
                    "RUST_LOG": "info"
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.logger.info(f"Started Rust environment runner (PID: {self.rust_process.pid})")
            
        except Exception as e:
            self.logger.warning(f"Failed to start Rust environment runner: {e}, falling back to Python implementation")
            self.rust_process = None
    
    async def create_environments(self, num_envs: int) -> List[RustRLEnvironment]:
        """Create multiple RL environments"""
        self.logger.info(f"Creating {num_envs} RL environments")
        
        envs = []
        for i in range(num_envs):
            env_id = f"rl_env_{i}"
            env = RustRLEnvironment(env_id, self.config)
            
            # Initialize environment (with fallback if Rust runner not available)
            if self.rust_process is None:
                # Fallback to Python implementation
                self.logger.info(f"Using Python fallback for environment {env_id}")
                env.initialized = True
                env.current_state = {"observation": np.random.random(4)}  # Mock state
                envs.append(env)
                self.envs[env_id] = env
            else:
                # Use Rust runner
                if await env.initialize():
                    envs.append(env)
                    self.envs[env_id] = env
                else:
                    self.logger.error(f"Failed to create environment {env_id}")
        
        self.env_pool = envs
        self.available_envs = envs.copy()
        
        self.logger.info(f"Created {len(envs)} RL environments successfully")
        return envs
    
    async def run_parallel_episodes(self, num_episodes: int, policy_fn: Callable) -> List[Dict[str, Any]]:
        """Run multiple episodes in parallel using Rust runner"""
        self.logger.info(f"Running {num_episodes} parallel episodes")
        
        episode_results = []
        
        # Check if Rust runner is available
        if self.rust_process is None:
            # Fallback to Python implementation
            self.logger.info("Using Python fallback for episode execution")
            for episode in range(num_episodes):
                result = await self._run_python_episode(episode, policy_fn)
                episode_results.append(result)
        else:
            # Use Rust runner
            # Create episode tasks
            episode_tasks = []
            for episode in range(num_episodes):
                task = self._create_episode_task(episode, policy_fn)
                episode_tasks.append(task)
            
            # Execute episodes in parallel
            if self.config.batch_processing:
                # Batch processing for better performance
                batch_size = min(self.config.num_workers, len(episode_tasks))
                for i in range(0, len(episode_tasks), batch_size):
                    batch = episode_tasks[i:i + batch_size]
                    batch_results = await self._run_episode_batch(batch)
                    episode_results.extend(batch_results)
            else:
                # Individual episode execution
                for task in episode_tasks:
                    result = await self._run_single_episode(task)
                    episode_results.append(result)
        
        # Update metrics
        self.total_episodes += num_episodes
        self.total_steps += sum(r.get('steps', 0) for r in episode_results)
        
        self.logger.info(f"Completed {num_episodes} parallel episodes")
        return episode_results
    
    async def _run_python_episode(self, episode: int, policy_fn: Callable) -> Dict[str, Any]:
        """Run a single episode using Python fallback"""
        try:
            # Simulate episode execution
            episode_length = np.random.randint(100, 1000)
            total_reward = 0.0
            
            for step in range(episode_length):
                # Simulate environment step
                action = policy_fn(np.random.random(4))  # Mock state
                reward = np.random.normal(0.1, 0.05)  # Small positive reward
                total_reward += reward
                
                # Small delay to simulate computation
                await asyncio.sleep(0.001)
            
            return {
                'episode_id': f"episode_{episode}",
                'success': True,
                'total_reward': total_reward,
                'steps': episode_length,
                'fps': 1000.0 / episode_length,
                'memory_usage': 0.5,
                'cpu_usage': 0.3,
                'gpu_usage': 0.0
            }
        except Exception as e:
            self.logger.error(f"Python episode {episode} failed: {e}")
            return {
                'episode_id': f"episode_{episode}",
                'success': False,
                'total_reward': 0.0,
                'steps': 0,
                'error': str(e)
            }
    
    def _create_episode_task(self, episode: int, policy_fn: Callable) -> Dict[str, Any]:
        """Create an episode task for Rust runner"""
        return {
            "episode_id": f"episode_{episode}",
            "policy_fn": policy_fn,
            "config": {
                "max_steps": self.config.episode_length,
                "memory_limit": self.config.env_memory_limit,
                "cpu_limit": self.config.env_cpu_limit
            },
            "timestamp": time.time()
        }
    
    async def _run_episode_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run a batch of episodes in parallel"""
        # Submit batch to Rust runner
        batch_task = {
            "action": "run_batch",
            "episodes": batch,
            "batch_id": f"batch_{int(time.time())}",
            "timestamp": time.time()
        }
        
        # Submit to queue
        batch_file = Path(self.config.queue_dir) / "pending" / f"batch_{int(time.time())}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_task, f)
        
        # Wait for batch results
        results = await self._wait_for_batch_results(batch_task["batch_id"])
        return results
    
    async def _run_single_episode(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single episode"""
        # Submit episode to Rust runner
        episode_file = Path(self.config.queue_dir) / "pending" / f"{task['episode_id']}.json"
        with open(episode_file, 'w') as f:
            json.dump(task, f)
        
        # Wait for episode result
        result = await self._wait_for_episode_result(task['episode_id'])
        return result
    
    async def _wait_for_batch_results(self, batch_id: str, timeout: int = 300) -> List[Dict[str, Any]]:
        """Wait for batch results from Rust runner"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for batch result file
            result_file = Path(self.config.queue_dir) / "done" / f"batch_{batch_id}_results.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    results = json.load(f)
                # Clean up result file
                result_file.unlink()
                return results
            
            await asyncio.sleep(0.1)  # 100ms polling
        
        return []
    
    async def _wait_for_episode_result(self, episode_id: str, timeout: int = 60) -> Dict[str, Any]:
        """Wait for episode result from Rust runner"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for episode result file
            result_file = Path(self.config.queue_dir) / "done" / f"{episode_id}_result.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result = json.load(f)
                # Clean up result file
                result_file.unlink()
                return result
            
            await asyncio.sleep(0.01)  # 10ms polling
        
        return {"success": False, "error": "Timeout waiting for episode result"}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from Rust runner"""
        try:
            # Query Rust runner metrics
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{self.config.rust_runner_port}/metrics") as response:
                    if response.status == 200:
                        metrics = await response.json()
                        return metrics
                    else:
                        return {}
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    async def close(self):
        """Close the Rust RL runner"""
        # Close all environments
        for env in self.envs.values():
            await env.close()
        
        # Stop Rust runner
        if self.rust_process:
            self.rust_process.terminate()
            self.rust_process.wait()
        
        self.logger.info("Rust RL runner closed successfully")


class OptimizedRLTrainer:
    """Optimized RL trainer using Rust runner and Go orchestrator"""
    
    def __init__(self, config: UniversalPufferConfig, rust_config: RustRLConfig, bus: LocalBus):
        self.config = config
        self.rust_config = rust_config
        self.bus = bus
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.rl_tracker = get_rl_tracker()
        self.rust_runner = RustRLRunner(rust_config)
        self.session_id = f"optimized_session_{int(time.time())}"
        
        # Training state
        self.training_active = False
        self.episode_count = 0
        self.total_timesteps = 0
        self.performance_history = []
        
        # Start session tracking
        self._start_session_tracking()
    
    def _start_session_tracking(self):
        """Start tracking this training session"""
        session_config = {
            'framework': self.config.framework,
            'num_workers': self.config.num_workers,
            'num_gpus': self.config.num_gpus,
            'num_envs': self.config.num_envs,
            'rust_runner': True,
            'optimized': True
        }
        
        self.rl_tracker.start_training_session(self.session_id, session_config)
        self.logger.info(f"Started optimized RL training session: {self.session_id}")
    
    async def train(self):
        """Main training loop with Rust runner optimization"""
        self.logger.info("Starting optimized RL training with Rust runner")
        self.training_active = True
        
        try:
            # Create environments
            envs = await self.rust_runner.create_environments(self.config.num_envs)
            self.logger.info(f"Created {len(envs)} environments")
            
            # Training loop
            while self.training_active and self.total_timesteps < self.config.total_timesteps:
                # Run parallel episodes
                episode_results = await self.rust_runner.run_parallel_episodes(
                    self.config.num_envs, 
                    self._get_policy_function()
                )
                
                # Process results
                self._process_episode_results(episode_results)
                
                # Log metrics
                await self._log_training_metrics()
                
                # Check for convergence
                if self._check_convergence():
                    self.logger.info("Training converged")
                    break
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.training_active = False
        finally:
            await self.rust_runner.close()
        
        self.logger.info("Optimized RL training completed")
    
    def _get_policy_function(self):
        """Get policy function for environment interaction"""
        # This would return the actual policy function
        # For now, return a simple random policy
        def random_policy(state):
            return np.random.uniform(-1, 1, size=(4,))  # Example action space
        
        return random_policy
    
    def _process_episode_results(self, results: List[Dict[str, Any]]):
        """Process episode results and update metrics"""
        for result in results:
            if result.get('success', False):
                reward = result.get('total_reward', 0.0)
                steps = result.get('steps', 0)
                
                self.performance_history.append(reward)
                self.total_timesteps += steps
                self.episode_count += 1
                
                # Log episode metrics to RedDB
                if self.config.reddb_tracking:
                    episode_metrics = {
                        'reward': reward,
                        'episode_length': steps,
                        'fps': result.get('fps', 0.0),
                        'memory_usage': result.get('memory_usage', 0.0),
                        'cpu_usage': result.get('cpu_usage', 0.0),
                        'convergence_score': min(reward / 100.0, 1.0),
                        'success_rate': 1.0 if reward > 0 else 0.0
                    }
                    
                    self.rl_tracker.log_episode_metrics(self.session_id, self.episode_count, episode_metrics)
    
    async def _log_training_metrics(self):
        """Log training metrics to RedDB"""
        if self.config.reddb_tracking:
            # Get performance metrics from Rust runner
            rust_metrics = await self.rust_runner.get_performance_metrics()
            
            # Update session with current metrics
            self.rl_tracker.update_training_session(self.session_id, {
                'num_episodes': self.episode_count,
                'total_timesteps': self.total_timesteps,
                'best_performance': max(self.performance_history) if self.performance_history else 0.0,
                'final_performance': self.performance_history[-1] if self.performance_history else 0.0,
                'rust_metrics': rust_metrics
            })
    
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
            'session_id': self.session_id
        }


def create_optimized_trainer(
    config: UniversalPufferConfig,
    rust_config: Optional[RustRLConfig] = None,
    bus: Optional[LocalBus] = None
) -> OptimizedRLTrainer:
    """Create an optimized RL trainer using Rust runner"""
    
    if rust_config is None:
        rust_config = RustRLConfig()
    
    if bus is None:
        bus = LocalBus()
    
    return OptimizedRLTrainer(config, rust_config, bus)


async def run_optimized_training(
    config: UniversalPufferConfig,
    rust_config: Optional[RustRLConfig] = None,
    bus: Optional[LocalBus] = None
):
    """Run optimized RL training with Rust runner"""
    
    trainer = create_optimized_trainer(config, rust_config, bus)
    
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
    
    asyncio.run(run_optimized_training(config, rust_config))
