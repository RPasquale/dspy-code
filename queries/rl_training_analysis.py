#!/usr/bin/env python3
"""
RedDB RL Training Analysis Script

This script analyzes the reinforcement learning training data in RedDB,
focusing on action quality, reward patterns, and learning effectiveness.

Usage:
    python queries/rl_training_analysis.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dspy_agent.db import get_enhanced_data_manager, Environment, ActionType, AgentState
from dspy_agent.db.enhanced_storage import get_recent_high_reward_actions, get_top_performing_signatures


class RLTrainingAnalyzer:
    """Analyzer for RL training data in RedDB"""
    
    def __init__(self):
        self.dm = get_enhanced_data_manager()
        self.storage = self.dm.storage
        self.namespace = "dspy_agent"
        
    def analyze_rl_training(self) -> Dict[str, Any]:
        """Analyze RL training data comprehensively"""
        print("🤖 Analyzing RedDB RL Training Data...")
        print("=" * 60)
        
        analysis = {
            "timestamp": time.time(),
            "action_analysis": {},
            "reward_analysis": {},
            "learning_progress": {},
            "training_effectiveness": {},
            "policy_quality": {},
            "exploration_vs_exploitation": {},
            "training_recommendations": {}
        }
        
        # 1. Analyze actions
        analysis["action_analysis"] = self._analyze_actions()
        
        # 2. Analyze rewards
        analysis["reward_analysis"] = self._analyze_rewards()
        
        # 3. Analyze learning progress
        analysis["learning_progress"] = self._analyze_learning_progress()
        
        # 4. Analyze training effectiveness
        analysis["training_effectiveness"] = self._analyze_training_effectiveness()
        
        # 5. Analyze policy quality
        analysis["policy_quality"] = self._analyze_policy_quality()
        
        # 6. Analyze exploration vs exploitation
        analysis["exploration_vs_exploitation"] = self._analyze_exploration_exploitation()
        
        # 7. Generate training recommendations
        analysis["training_recommendations"] = self._generate_training_recommendations(analysis)
        
        return analysis
    
    def _analyze_actions(self) -> Dict[str, Any]:
        """Analyze action patterns and quality"""
        print("\n🎯 ACTION ANALYSIS")
        print("-" * 40)
        
        action_analysis = {
            "action_distribution": {},
            "action_quality_metrics": {},
            "action_timing": {},
            "action_effectiveness": {}
        }
        
        try:
            # Get recent actions
            actions = self.dm.get_recent_actions(limit=2000)
            
            if not actions:
                print("ℹ️  No actions found for analysis")
                return action_analysis
            
            # Action type distribution
            action_types = Counter(a.action_type.value for a in actions)
            action_analysis["action_distribution"] = dict(action_types)
            
            # Action quality metrics
            rewards = [a.reward for a in actions]
            confidences = [a.confidence for a in actions]
            execution_times = [a.execution_time for a in actions]
            
            action_analysis["action_quality_metrics"] = {
                "total_actions": len(actions),
                "avg_reward": statistics.mean(rewards),
                "median_reward": statistics.median(rewards),
                "reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0,
                "avg_confidence": statistics.mean(confidences),
                "median_confidence": statistics.median(confidences),
                "confidence_std": statistics.stdev(confidences) if len(confidences) > 1 else 0,
                "avg_execution_time": statistics.mean(execution_times),
                "median_execution_time": statistics.median(execution_times),
                "execution_time_std": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            }
            
            # Action timing analysis
            timestamps = [a.timestamp for a in actions]
            timestamps.sort()
            
            if len(timestamps) > 1:
                time_spans = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                action_analysis["action_timing"] = {
                    "time_span_hours": (timestamps[-1] - timestamps[0]) / 3600,
                    "avg_time_between_actions": statistics.mean(time_spans),
                    "median_time_between_actions": statistics.median(time_spans),
                    "actions_per_hour": len(actions) / ((timestamps[-1] - timestamps[0]) / 3600)
                }
            
            # Action effectiveness by type
            effectiveness_by_type = {}
            for action_type in ActionType:
                type_actions = [a for a in actions if a.action_type == action_type]
                if type_actions:
                    type_rewards = [a.reward for a in type_actions]
                    effectiveness_by_type[action_type.value] = {
                        "count": len(type_actions),
                        "avg_reward": statistics.mean(type_rewards),
                        "avg_confidence": statistics.mean([a.confidence for a in type_actions]),
                        "success_rate": sum(1 for r in type_rewards if r > 0.5) / len(type_rewards)
                    }
            
            action_analysis["action_effectiveness"] = effectiveness_by_type
            
            print(f"✅ Action Analysis:")
            print(f"   Total actions: {len(actions)}")
            print(f"   Avg reward: {action_analysis['action_quality_metrics']['avg_reward']:.3f}")
            print(f"   Avg confidence: {action_analysis['action_quality_metrics']['avg_confidence']:.3f}")
            print(f"   Actions per hour: {action_analysis['action_timing'].get('actions_per_hour', 0):.1f}")
            
            # Show top action types
            print(f"   Top action types:")
            for action_type, count in action_types.most_common(5):
                print(f"     • {action_type}: {count}")
            
        except Exception as e:
            print(f"⚠️  Action Analysis: {e}")
        
        return action_analysis
    
    def _analyze_rewards(self) -> Dict[str, Any]:
        """Analyze reward patterns and distribution"""
        print("\n🏆 REWARD ANALYSIS")
        print("-" * 40)
        
        reward_analysis = {
            "reward_distribution": {},
            "reward_trends": {},
            "high_reward_patterns": {},
            "reward_consistency": {}
        }
        
        try:
            # Get recent actions
            actions = self.dm.get_recent_actions(limit=2000)
            
            if not actions:
                print("ℹ️  No actions found for reward analysis")
                return reward_analysis
            
            rewards = [a.reward for a in actions]
            timestamps = [a.timestamp for a in actions]
            
            # Reward distribution
            reward_analysis["reward_distribution"] = {
                "min_reward": min(rewards),
                "max_reward": max(rewards),
                "mean_reward": statistics.mean(rewards),
                "median_reward": statistics.median(rewards),
                "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0,
                "reward_ranges": {
                    "very_low": sum(1 for r in rewards if r < 0.2),
                    "low": sum(1 for r in rewards if 0.2 <= r < 0.4),
                    "medium": sum(1 for r in rewards if 0.4 <= r < 0.6),
                    "high": sum(1 for r in rewards if 0.6 <= r < 0.8),
                    "very_high": sum(1 for r in rewards if r >= 0.8)
                }
            }
            
            # Reward trends over time
            if len(actions) > 10:
                # Sort by timestamp
                sorted_actions = sorted(actions, key=lambda x: x.timestamp)
                
                # Calculate moving average
                window_size = min(50, len(sorted_actions) // 4)
                moving_averages = []
                
                for i in range(window_size, len(sorted_actions)):
                    window_rewards = [a.reward for a in sorted_actions[i-window_size:i]]
                    moving_averages.append(statistics.mean(window_rewards))
                
                if moving_averages:
                    reward_analysis["reward_trends"] = {
                        "trend_direction": "improving" if moving_averages[-1] > moving_averages[0] else "declining",
                        "trend_magnitude": abs(moving_averages[-1] - moving_averages[0]),
                        "recent_avg": moving_averages[-1],
                        "early_avg": moving_averages[0],
                        "volatility": statistics.stdev(moving_averages) if len(moving_averages) > 1 else 0
                    }
            
            # High reward patterns
            high_reward_actions = [a for a in actions if a.reward >= 0.8]
            
            if high_reward_actions:
                high_reward_types = Counter(a.action_type.value for a in high_reward_actions)
                high_reward_confidences = [a.confidence for a in high_reward_actions]
                
                reward_analysis["high_reward_patterns"] = {
                    "count": len(high_reward_actions),
                    "percentage": len(high_reward_actions) / len(actions) * 100,
                    "type_distribution": dict(high_reward_types),
                    "avg_confidence": statistics.mean(high_reward_confidences),
                    "common_characteristics": {
                        "high_confidence": sum(1 for c in high_reward_confidences if c > 0.8),
                        "fast_execution": sum(1 for a in high_reward_actions if a.execution_time < 1.0)
                    }
                }
            
            # Reward consistency
            reward_analysis["reward_consistency"] = {
                "coefficient_of_variation": statistics.stdev(rewards) / statistics.mean(rewards) if statistics.mean(rewards) > 0 else 0,
                "consistency_rating": "high" if statistics.stdev(rewards) < 0.2 else "medium" if statistics.stdev(rewards) < 0.4 else "low"
            }
            
            print(f"✅ Reward Analysis:")
            print(f"   Mean reward: {reward_analysis['reward_distribution']['mean_reward']:.3f}")
            print(f"   Reward range: {reward_analysis['reward_distribution']['min_reward']:.3f} - {reward_analysis['reward_distribution']['max_reward']:.3f}")
            print(f"   High reward actions: {reward_analysis['high_reward_patterns'].get('count', 0)} ({reward_analysis['high_reward_patterns'].get('percentage', 0):.1f}%)")
            
            if reward_analysis.get("reward_trends"):
                print(f"   Trend: {reward_analysis['reward_trends']['trend_direction']}")
            
        except Exception as e:
            print(f"⚠️  Reward Analysis: {e}")
        
        return reward_analysis
    
    def _analyze_learning_progress(self) -> Dict[str, Any]:
        """Analyze learning progress over time"""
        print("\n📈 LEARNING PROGRESS")
        print("-" * 40)
        
        learning_analysis = {
            "training_sessions": {},
            "performance_trends": {},
            "convergence_analysis": {},
            "learning_efficiency": {}
        }
        
        try:
            # Get training history
            training_history = self.dm.get_training_history(limit=100)
            
            if not training_history:
                print("ℹ️  No training history found")
                return learning_analysis
            
            # Sort by timestamp
            training_history.sort(key=lambda x: x.timestamp)
            
            # Training sessions analysis
            learning_analysis["training_sessions"] = {
                "total_sessions": len(training_history),
                "time_span_hours": (training_history[-1].timestamp - training_history[0].timestamp) / 3600,
                "sessions_per_hour": len(training_history) / ((training_history[-1].timestamp - training_history[0].timestamp) / 3600),
                "model_types": list(set(t.model_type for t in training_history)),
                "environments": list(set(t.environment.value for t in training_history))
            }
            
            # Performance trends
            training_accuracies = [t.training_accuracy for t in training_history]
            validation_accuracies = [t.validation_accuracy for t in training_history]
            losses = [t.loss for t in training_history]
            
            learning_analysis["performance_trends"] = {
                "training_accuracy": {
                    "initial": training_accuracies[0],
                    "final": training_accuracies[-1],
                    "improvement": training_accuracies[-1] - training_accuracies[0],
                    "trend": "improving" if training_accuracies[-1] > training_accuracies[0] else "declining"
                },
                "validation_accuracy": {
                    "initial": validation_accuracies[0],
                    "final": validation_accuracies[-1],
                    "improvement": validation_accuracies[-1] - validation_accuracies[0],
                    "trend": "improving" if validation_accuracies[-1] > validation_accuracies[0] else "declining"
                },
                "loss": {
                    "initial": losses[0],
                    "final": losses[-1],
                    "improvement": losses[0] - losses[-1],  # Loss should decrease
                    "trend": "improving" if losses[-1] < losses[0] else "declining"
                }
            }
            
            # Convergence analysis
            if len(training_history) > 10:
                # Check if performance has plateaued
                recent_sessions = training_history[-10:]
                recent_accuracies = [t.training_accuracy for t in recent_sessions]
                recent_losses = [t.loss for t in recent_sessions]
                
                accuracy_std = statistics.stdev(recent_accuracies) if len(recent_accuracies) > 1 else 0
                loss_std = statistics.stdev(recent_losses) if len(recent_losses) > 1 else 0
                
                learning_analysis["convergence_analysis"] = {
                    "accuracy_converged": accuracy_std < 0.01,
                    "loss_converged": loss_std < 0.01,
                    "recent_accuracy_std": accuracy_std,
                    "recent_loss_std": loss_std,
                    "convergence_status": "converged" if accuracy_std < 0.01 and loss_std < 0.01 else "still_learning"
                }
            
            # Learning efficiency
            if len(training_history) > 1:
                total_improvement = training_accuracies[-1] - training_accuracies[0]
                total_time = training_history[-1].timestamp - training_history[0].timestamp
                
                learning_analysis["learning_efficiency"] = {
                    "improvement_per_hour": total_improvement / (total_time / 3600),
                    "sessions_to_improve": len(training_history),
                    "efficiency_rating": "high" if total_improvement > 0.1 else "medium" if total_improvement > 0.05 else "low"
                }
            
            print(f"✅ Learning Progress:")
            print(f"   Training sessions: {len(training_history)}")
            print(f"   Training accuracy: {training_accuracies[0]:.3f} → {training_accuracies[-1]:.3f}")
            print(f"   Validation accuracy: {validation_accuracies[0]:.3f} → {validation_accuracies[-1]:.3f}")
            print(f"   Loss: {losses[0]:.3f} → {losses[-1]:.3f}")
            
            if learning_analysis.get("convergence_analysis"):
                print(f"   Convergence: {learning_analysis['convergence_analysis']['convergence_status']}")
            
        except Exception as e:
            print(f"⚠️  Learning Progress: {e}")
        
        return learning_analysis
    
    def _analyze_training_effectiveness(self) -> Dict[str, Any]:
        """Analyze training effectiveness and efficiency"""
        print("\n⚡ TRAINING EFFECTIVENESS")
        print("-" * 40)
        
        effectiveness_analysis = {
            "action_quality_improvement": {},
            "reward_optimization": {},
            "exploration_efficiency": {},
            "training_bottlenecks": {}
        }
        
        try:
            # Get actions from different time periods
            all_actions = self.dm.get_recent_actions(limit=2000)
            
            if len(all_actions) < 100:
                print("ℹ️  Insufficient actions for effectiveness analysis")
                return effectiveness_analysis
            
            # Split actions into early and recent periods
            sorted_actions = sorted(all_actions, key=lambda x: x.timestamp)
            mid_point = len(sorted_actions) // 2
            early_actions = sorted_actions[:mid_point]
            recent_actions = sorted_actions[mid_point:]
            
            # Action quality improvement
            early_rewards = [a.reward for a in early_actions]
            recent_rewards = [a.reward for a in recent_actions]
            
            effectiveness_analysis["action_quality_improvement"] = {
                "early_avg_reward": statistics.mean(early_rewards),
                "recent_avg_reward": statistics.mean(recent_rewards),
                "improvement": statistics.mean(recent_rewards) - statistics.mean(early_rewards),
                "improvement_percentage": ((statistics.mean(recent_rewards) - statistics.mean(early_rewards)) / statistics.mean(early_rewards)) * 100 if statistics.mean(early_rewards) > 0 else 0
            }
            
            # Reward optimization
            high_reward_early = sum(1 for r in early_rewards if r > 0.8)
            high_reward_recent = sum(1 for r in recent_rewards if r > 0.8)
            
            effectiveness_analysis["reward_optimization"] = {
                "early_high_reward_rate": high_reward_early / len(early_actions),
                "recent_high_reward_rate": high_reward_recent / len(recent_actions),
                "optimization_improvement": (high_reward_recent / len(recent_actions)) - (high_reward_early / len(early_actions))
            }
            
            # Exploration efficiency
            early_confidences = [a.confidence for a in early_actions]
            recent_confidences = [a.confidence for a in recent_actions]
            
            effectiveness_analysis["exploration_efficiency"] = {
                "early_avg_confidence": statistics.mean(early_confidences),
                "recent_avg_confidence": statistics.mean(recent_confidences),
                "confidence_change": statistics.mean(recent_confidences) - statistics.mean(early_confidences),
                "exploration_balance": "balanced" if 0.3 <= statistics.mean(recent_confidences) <= 0.7 else "overconfident" if statistics.mean(recent_confidences) > 0.7 else "underconfident"
            }
            
            # Training bottlenecks
            execution_times = [a.execution_time for a in all_actions]
            slow_actions = sum(1 for t in execution_times if t > 5.0)
            
            effectiveness_analysis["training_bottlenecks"] = {
                "slow_action_rate": slow_actions / len(all_actions),
                "avg_execution_time": statistics.mean(execution_times),
                "execution_time_std": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                "bottleneck_severity": "high" if slow_actions / len(all_actions) > 0.1 else "medium" if slow_actions / len(all_actions) > 0.05 else "low"
            }
            
            print(f"✅ Training Effectiveness:")
            print(f"   Reward improvement: {effectiveness_analysis['action_quality_improvement']['improvement']:.3f}")
            print(f"   High reward rate: {effectiveness_analysis['reward_optimization']['recent_high_reward_rate']:.1%}")
            print(f"   Exploration balance: {effectiveness_analysis['exploration_efficiency']['exploration_balance']}")
            print(f"   Bottleneck severity: {effectiveness_analysis['training_bottlenecks']['bottleneck_severity']}")
            
        except Exception as e:
            print(f"⚠️  Training Effectiveness: {e}")
        
        return effectiveness_analysis
    
    def _analyze_policy_quality(self) -> Dict[str, Any]:
        """Analyze policy quality and consistency"""
        print("\n🎯 POLICY QUALITY")
        print("-" * 40)
        
        policy_analysis = {
            "policy_consistency": {},
            "action_selection_quality": {},
            "policy_stability": {},
            "decision_making_patterns": {}
        }
        
        try:
            # Get recent actions
            actions = self.dm.get_recent_actions(limit=1000)
            
            if not actions:
                print("ℹ️  No actions found for policy analysis")
                return policy_analysis
            
            # Policy consistency (how consistent are similar actions?)
            action_type_consistency = {}
            for action_type in ActionType:
                type_actions = [a for a in actions if a.action_type == action_type]
                if len(type_actions) > 5:  # Need enough samples
                    type_rewards = [a.reward for a in type_actions]
                    type_confidences = [a.confidence for a in type_actions]
                    
                    action_type_consistency[action_type.value] = {
                        "reward_consistency": 1 - (statistics.stdev(type_rewards) / statistics.mean(type_rewards)) if statistics.mean(type_rewards) > 0 else 0,
                        "confidence_consistency": 1 - (statistics.stdev(type_confidences) / statistics.mean(type_confidences)) if statistics.mean(type_confidences) > 0 else 0,
                        "sample_size": len(type_actions)
                    }
            
            policy_analysis["policy_consistency"] = action_type_consistency
            
            # Action selection quality
            high_confidence_actions = [a for a in actions if a.confidence > 0.8]
            high_confidence_high_reward = [a for a in high_confidence_actions if a.reward > 0.7]
            
            policy_analysis["action_selection_quality"] = {
                "high_confidence_rate": len(high_confidence_actions) / len(actions),
                "high_confidence_high_reward_rate": len(high_confidence_high_reward) / len(high_confidence_actions) if high_confidence_actions else 0,
                "selection_accuracy": len(high_confidence_high_reward) / len(actions)
            }
            
            # Policy stability over time
            if len(actions) > 50:
                # Split into time windows
                sorted_actions = sorted(actions, key=lambda x: x.timestamp)
                window_size = len(sorted_actions) // 5
                
                window_rewards = []
                for i in range(0, len(sorted_actions), window_size):
                    window = sorted_actions[i:i+window_size]
                    if window:
                        window_rewards.append(statistics.mean([a.reward for a in window]))
                
                if len(window_rewards) > 1:
                    policy_analysis["policy_stability"] = {
                        "reward_stability": 1 - (statistics.stdev(window_rewards) / statistics.mean(window_rewards)) if statistics.mean(window_rewards) > 0 else 0,
                        "stability_rating": "stable" if statistics.stdev(window_rewards) < 0.1 else "moderate" if statistics.stdev(window_rewards) < 0.2 else "unstable"
                    }
            
            # Decision making patterns
            action_type_distribution = Counter(a.action_type.value for a in actions)
            total_actions = len(actions)
            
            policy_analysis["decision_making_patterns"] = {
                "action_diversity": len(action_type_distribution),
                "dominant_action_type": action_type_distribution.most_common(1)[0][0] if action_type_distribution else None,
                "dominant_action_percentage": (action_type_distribution.most_common(1)[0][1] / total_actions) * 100 if action_type_distribution else 0,
                "diversity_rating": "high" if len(action_type_distribution) > 5 else "medium" if len(action_type_distribution) > 3 else "low"
            }
            
            print(f"✅ Policy Quality:")
            print(f"   Selection accuracy: {policy_analysis['action_selection_quality']['selection_accuracy']:.1%}")
            print(f"   Action diversity: {policy_analysis['decision_making_patterns']['diversity_rating']} ({policy_analysis['decision_making_patterns']['action_diversity']} types)")
            
            if policy_analysis.get("policy_stability"):
                print(f"   Policy stability: {policy_analysis['policy_stability']['stability_rating']}")
            
        except Exception as e:
            print(f"⚠️  Policy Quality: {e}")
        
        return policy_analysis
    
    def _analyze_exploration_exploitation(self) -> Dict[str, Any]:
        """Analyze exploration vs exploitation balance"""
        print("\n🔍 EXPLORATION VS EXPLOITATION")
        print("-" * 40)
        
        exploration_analysis = {
            "exploration_metrics": {},
            "exploitation_metrics": {},
            "balance_analysis": {},
            "exploration_efficiency": {}
        }
        
        try:
            # Get recent actions
            actions = self.dm.get_recent_actions(limit=1000)
            
            if not actions:
                print("ℹ️  No actions found for exploration analysis")
                return exploration_analysis
            
            # Define exploration vs exploitation based on confidence and reward
            exploration_actions = [a for a in actions if a.confidence < 0.6]  # Low confidence = exploration
            exploitation_actions = [a for a in actions if a.confidence >= 0.6]  # High confidence = exploitation
            
            # Exploration metrics
            if exploration_actions:
                exploration_rewards = [a.reward for a in exploration_actions]
                exploration_analysis["exploration_metrics"] = {
                    "count": len(exploration_actions),
                    "percentage": len(exploration_actions) / len(actions) * 100,
                    "avg_reward": statistics.mean(exploration_rewards),
                    "reward_variance": statistics.variance(exploration_rewards) if len(exploration_rewards) > 1 else 0,
                    "discovery_rate": sum(1 for r in exploration_rewards if r > 0.8) / len(exploration_rewards)
                }
            
            # Exploitation metrics
            if exploitation_actions:
                exploitation_rewards = [a.reward for a in exploitation_actions]
                exploration_analysis["exploitation_metrics"] = {
                    "count": len(exploitation_actions),
                    "percentage": len(exploitation_actions) / len(actions) * 100,
                    "avg_reward": statistics.mean(exploitation_rewards),
                    "reward_variance": statistics.variance(exploitation_rewards) if len(exploitation_rewards) > 1 else 0,
                    "success_rate": sum(1 for r in exploitation_rewards if r > 0.5) / len(exploitation_rewards)
                }
            
            # Balance analysis
            exploration_ratio = len(exploration_actions) / len(actions) if actions else 0
            exploitation_ratio = len(exploitation_actions) / len(actions) if actions else 0
            
            exploration_analysis["balance_analysis"] = {
                "exploration_ratio": exploration_ratio,
                "exploitation_ratio": exploitation_ratio,
                "balance_rating": "balanced" if 0.2 <= exploration_ratio <= 0.4 else "over_exploration" if exploration_ratio > 0.4 else "over_exploitation",
                "optimal_balance": abs(exploration_ratio - 0.3) < 0.1  # 30% exploration is often optimal
            }
            
            # Exploration efficiency
            if exploration_actions and exploitation_actions:
                exploration_avg_reward = statistics.mean([a.reward for a in exploration_actions])
                exploitation_avg_reward = statistics.mean([a.reward for a in exploitation_actions])
                
                exploration_analysis["exploration_efficiency"] = {
                    "exploration_reward_ratio": exploration_avg_reward / exploitation_avg_reward if exploitation_avg_reward > 0 else 0,
                    "efficiency_rating": "high" if exploration_avg_reward > exploitation_avg_reward * 0.8 else "medium" if exploration_avg_reward > exploitation_avg_reward * 0.6 else "low"
                }
            
            print(f"✅ Exploration vs Exploitation:")
            print(f"   Exploration: {exploration_analysis['exploration_metrics'].get('percentage', 0):.1f}%")
            print(f"   Exploitation: {exploration_analysis['exploitation_metrics'].get('percentage', 0):.1f}%")
            print(f"   Balance: {exploration_analysis['balance_analysis']['balance_rating']}")
            
            if exploration_analysis.get("exploration_efficiency"):
                print(f"   Exploration efficiency: {exploration_analysis['exploration_efficiency']['efficiency_rating']}")
            
        except Exception as e:
            print(f"⚠️  Exploration vs Exploitation: {e}")
        
        return exploration_analysis
    
    def _generate_training_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training recommendations based on analysis"""
        print("\n💡 TRAINING RECOMMENDATIONS")
        print("-" * 40)
        
        recommendations = {
            "immediate_actions": [],
            "medium_term_goals": [],
            "long_term_strategies": [],
            "priority_level": "medium"
        }
        
        try:
            # Analyze current state and generate recommendations
            
            # Check reward trends
            reward_analysis = analysis.get("reward_analysis", {})
            if reward_analysis.get("reward_trends", {}).get("trend_direction") == "declining":
                recommendations["immediate_actions"].append("Investigate declining reward trend - check for overfitting or environment changes")
                recommendations["priority_level"] = "high"
            
            # Check exploration balance
            exploration_analysis = analysis.get("exploration_vs_exploitation", {})
            balance_rating = exploration_analysis.get("balance_analysis", {}).get("balance_rating", "unknown")
            if balance_rating == "over_exploitation":
                recommendations["immediate_actions"].append("Increase exploration rate - agent may be stuck in local optima")
            elif balance_rating == "over_exploration":
                recommendations["immediate_actions"].append("Increase exploitation rate - agent may not be learning from good actions")
            
            # Check policy stability
            policy_analysis = analysis.get("policy_quality", {})
            stability_rating = policy_analysis.get("policy_stability", {}).get("stability_rating", "unknown")
            if stability_rating == "unstable":
                recommendations["immediate_actions"].append("Policy is unstable - consider reducing learning rate or increasing batch size")
            
            # Check training effectiveness
            effectiveness_analysis = analysis.get("training_effectiveness", {})
            improvement = effectiveness_analysis.get("action_quality_improvement", {}).get("improvement", 0)
            if improvement < 0.01:
                recommendations["medium_term_goals"].append("Low improvement rate - consider curriculum learning or reward shaping")
            
            # Check convergence
            learning_analysis = analysis.get("learning_progress", {})
            convergence_status = learning_analysis.get("convergence_analysis", {}).get("convergence_status", "unknown")
            if convergence_status == "converged":
                recommendations["long_term_strategies"].append("Model has converged - consider increasing task complexity or adding new environments")
            
            # Check action diversity
            decision_patterns = policy_analysis.get("decision_making_patterns", {})
            diversity_rating = decision_patterns.get("diversity_rating", "unknown")
            if diversity_rating == "low":
                recommendations["medium_term_goals"].append("Low action diversity - encourage exploration of different action types")
            
            # Check bottlenecks
            effectiveness_analysis = analysis.get("training_effectiveness", {})
            bottleneck_severity = effectiveness_analysis.get("training_bottlenecks", {}).get("bottleneck_severity", "unknown")
            if bottleneck_severity == "high":
                recommendations["immediate_actions"].append("High execution time bottleneck - optimize action execution or reduce complexity")
                recommendations["priority_level"] = "high"
            
            # Default recommendations if no specific issues found
            if not recommendations["immediate_actions"]:
                recommendations["immediate_actions"].append("Continue current training - system appears stable")
            
            if not recommendations["medium_term_goals"]:
                recommendations["medium_term_goals"].append("Monitor performance metrics and adjust hyperparameters as needed")
            
            if not recommendations["long_term_strategies"]:
                recommendations["long_term_strategies"].append("Plan for scaling and additional complexity as performance improves")
            
            print(f"✅ Training Recommendations:")
            print(f"   Priority Level: {recommendations['priority_level'].upper()}")
            print(f"   Immediate Actions ({len(recommendations['immediate_actions'])}):")
            for i, action in enumerate(recommendations["immediate_actions"], 1):
                print(f"     {i}. {action}")
            
            print(f"   Medium-term Goals ({len(recommendations['medium_term_goals'])}):")
            for i, goal in enumerate(recommendations["medium_term_goals"], 1):
                print(f"     {i}. {goal}")
            
            print(f"   Long-term Strategies ({len(recommendations['long_term_strategies'])}):")
            for i, strategy in enumerate(recommendations["long_term_strategies"], 1):
                print(f"     {i}. {strategy}")
            
        except Exception as e:
            print(f"⚠️  Training Recommendations: {e}")
        
        return recommendations
    
    def print_rl_summary(self, analysis: Dict[str, Any]):
        """Print a summary of the RL training analysis"""
        print("\n" + "=" * 60)
        print("🤖 RL TRAINING ANALYSIS SUMMARY")
        print("=" * 60)
        
        action_analysis = analysis.get("action_analysis", {})
        reward_analysis = analysis.get("reward_analysis", {})
        learning_analysis = analysis.get("learning_progress", {})
        effectiveness_analysis = analysis.get("training_effectiveness", {})
        recommendations = analysis.get("training_recommendations", {})
        
        # Overall metrics
        total_actions = action_analysis.get("action_quality_metrics", {}).get("total_actions", 0)
        avg_reward = action_analysis.get("action_quality_metrics", {}).get("avg_reward", 0)
        training_sessions = learning_analysis.get("training_sessions", {}).get("total_sessions", 0)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"   • {total_actions} total training actions")
        print(f"   • {avg_reward:.3f} average reward")
        print(f"   • {training_sessions} training sessions")
        
        # Performance summary
        if reward_analysis.get("high_reward_patterns"):
            high_reward_count = reward_analysis["high_reward_patterns"]["count"]
            high_reward_percentage = reward_analysis["high_reward_patterns"]["percentage"]
            print(f"   • {high_reward_count} high-reward actions ({high_reward_percentage:.1f}%)")
        
        # Learning progress
        if learning_analysis.get("performance_trends"):
            trends = learning_analysis["performance_trends"]
            print(f"\n📈 LEARNING PROGRESS:")
            print(f"   • Training accuracy trend: {trends['training_accuracy']['trend']}")
            print(f"   • Validation accuracy trend: {trends['validation_accuracy']['trend']}")
            print(f"   • Loss trend: {trends['loss']['trend']}")
        
        # Training effectiveness
        if effectiveness_analysis.get("action_quality_improvement"):
            improvement = effectiveness_analysis["action_quality_improvement"]["improvement"]
            print(f"\n⚡ TRAINING EFFECTIVENESS:")
            print(f"   • Reward improvement: {improvement:+.3f}")
        
        # Policy quality
        policy_analysis = analysis.get("policy_quality", {})
        if policy_analysis.get("decision_making_patterns"):
            diversity = policy_analysis["decision_making_patterns"]["diversity_rating"]
            print(f"   • Action diversity: {diversity}")
        
        # Exploration balance
        exploration_analysis = analysis.get("exploration_vs_exploitation", {})
        if exploration_analysis.get("balance_analysis"):
            balance = exploration_analysis["balance_analysis"]["balance_rating"]
            print(f"   • Exploration balance: {balance}")
        
        # Recommendations
        if recommendations.get("immediate_actions"):
            print(f"\n💡 IMMEDIATE RECOMMENDATIONS:")
            for i, action in enumerate(recommendations["immediate_actions"][:3], 1):
                print(f"   {i}. {action}")
        
        print(f"\n⏰ Analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(analysis['timestamp']))}")


def main():
    """Main function to run the RL training analysis"""
    try:
        analyzer = RLTrainingAnalyzer()
        analysis = analyzer.analyze_rl_training()
        analyzer.print_rl_summary(analysis)
        
        # Save analysis to file
        import json
        output_file = Path(__file__).parent / "rl_training_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Full analysis saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error running RL training analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
