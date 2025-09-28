"""
Fallback RL system for when PufferLib is not available.
Provides basic RL functionality without the full PufferLib stack.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for fallback RL system."""
    learning_rate: float = 0.001
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    gamma: float = 0.95
    batch_size: int = 32
    memory_size: int = 10000
    update_frequency: int = 100
    save_frequency: int = 1000


@dataclass
class Experience:
    """Experience tuple for RL."""
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    done: bool
    timestamp: float


class ExperienceReplay:
    """Simple experience replay buffer."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0
    
    def push(self, experience: Experience):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class SimplePolicy:
    """Simple policy for action selection."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.epsilon = config.epsilon
        self.action_values: Dict[str, float] = {}
        self.action_counts: Dict[str, int] = {}
    
    def select_action(self, state: Dict[str, Any], available_actions: List[str]) -> str:
        """Select action using epsilon-greedy policy."""
        if not available_actions:
            return "plan"  # Default action

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            preferred = [a for a in available_actions if a in {
                'patch',
                'run_tests',
                'lint',
                'build',
                'shell_run',
            }]
            pool = preferred or available_actions
            return random.choice(pool)

        # Greedy selection based on action values
        best_action = available_actions[0]
        best_value = self.action_values.get(best_action, 0.0)

        for action in available_actions:
            value = self.action_values.get(action, 0.0)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def update_action_value(self, action: str, reward: float):
        """Update action value using simple averaging."""
        if action not in self.action_values:
            self.action_values[action] = 0.0
            self.action_counts[action] = 0
        
        self.action_counts[action] += 1
        count = self.action_counts[action]
        current_value = self.action_values[action]
        
        # Update using running average
        self.action_values[action] = current_value + (reward - current_value) / count
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)


class FallbackRLTrainer:
    """Fallback RL trainer that works without PufferLib."""
    
    def __init__(self, config: Optional[RLConfig] = None, save_dir: Optional[Path] = None):
        self.config = config or RLConfig()
        self.save_dir = save_dir or Path(".dspy_rl_fallback")
        self.save_dir.mkdir(exist_ok=True)
        
        self.policy = SimplePolicy(self.config)
        self.memory = ExperienceReplay(self.config.memory_size)
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        
        # Load existing model if available
        self.load_model()
    
    def train_step(self, state: Dict[str, Any], action: str, reward: float, 
                   next_state: Dict[str, Any], done: bool = False) -> Dict[str, Any]:
        """Perform one training step."""
        # Create experience
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            timestamp=time.time()
        )
        
        # Store experience
        self.memory.push(experience)
        
        # Update policy
        self.policy.update_action_value(action, reward)
        
        # Update counters
        self.step_count += 1
        self.total_reward += reward
        
        # Decay epsilon periodically
        if self.step_count % self.config.update_frequency == 0:
            self.policy.decay_epsilon()
        
        # Save model periodically
        if self.step_count % self.config.save_frequency == 0:
            self.save_model()
        
        # End episode
        if done:
            self.episode_count += 1
            episode_reward = self.total_reward
            self.total_reward = 0.0
            
            return {
                "episode": self.episode_count,
                "step": self.step_count,
                "reward": episode_reward,
                "epsilon": self.policy.epsilon,
                "action_values": dict(self.policy.action_values),
                "memory_size": len(self.memory)
            }
        
        return {
            "step": self.step_count,
            "reward": reward,
            "epsilon": self.policy.epsilon,
            "total_reward": self.total_reward
        }
    
    def select_action(self, state: Dict[str, Any], available_actions: List[str]) -> str:
        """Select action using current policy."""
        return self.policy.select_action(state, available_actions)
    
    def save_model(self):
        """Save model to disk."""
        try:
            model_data = {
                "config": {
                    "learning_rate": self.config.learning_rate,
                    "epsilon": self.policy.epsilon,
                    "epsilon_decay": self.config.epsilon_decay,
                    "min_epsilon": self.config.min_epsilon,
                    "gamma": self.config.gamma,
                    "batch_size": self.config.batch_size,
                    "memory_size": self.config.memory_size,
                },
                "policy": {
                    "action_values": self.policy.action_values,
                    "action_counts": self.policy.action_counts,
                    "epsilon": self.policy.epsilon,
                },
                "training": {
                    "episode_count": self.episode_count,
                    "step_count": self.step_count,
                    "total_reward": self.total_reward,
                }
            }
            
            model_file = self.save_dir / "model.json"
            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Model saved to {model_file}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self):
        """Load model from disk."""
        try:
            model_file = self.save_dir / "model.json"
            if not model_file.exists():
                return
            
            with open(model_file, 'r') as f:
                model_data = json.load(f)
            
            # Load policy
            if "policy" in model_data:
                policy_data = model_data["policy"]
                self.policy.action_values = policy_data.get("action_values", {})
                self.policy.action_counts = policy_data.get("action_counts", {})
                self.policy.epsilon = policy_data.get("epsilon", self.config.epsilon)
            
            # Load training state
            if "training" in model_data:
                training_data = model_data["training"]
                self.episode_count = training_data.get("episode_count", 0)
                self.step_count = training_data.get("step_count", 0)
                self.total_reward = training_data.get("total_reward", 0.0)
            
            logger.info(f"Model loaded from {model_file}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "epsilon": self.policy.epsilon,
            "action_values": dict(self.policy.action_values),
            "action_counts": dict(self.policy.action_counts),
            "memory_size": len(self.memory),
            "total_reward": self.total_reward,
        }


# Global trainer instance
_trainer: Optional[FallbackRLTrainer] = None


def get_fallback_trainer() -> FallbackRLTrainer:
    """Get the global fallback trainer instance."""
    global _trainer
    if _trainer is None:
        _trainer = FallbackRLTrainer()
    return _trainer


def train_fallback_rl(state: Dict[str, Any], action: str, reward: float, 
                     next_state: Dict[str, Any], done: bool = False) -> Dict[str, Any]:
    """Train the fallback RL system."""
    trainer = get_fallback_trainer()
    return trainer.train_step(state, action, reward, next_state, done)


def select_fallback_action(state: Dict[str, Any], available_actions: List[str]) -> str:
    """Select action using fallback RL system."""
    trainer = get_fallback_trainer()
    return trainer.select_action(state, available_actions)


def get_fallback_stats() -> Dict[str, Any]:
    """Get fallback RL statistics."""
    trainer = get_fallback_trainer()
    return trainer.get_stats()
