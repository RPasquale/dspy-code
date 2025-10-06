# DSPy-Code Distributed Training Guide

This guide provides comprehensive instructions for setting up and running distributed training for the DSPy-Code agent using PufferLib's full feature set.

## 🚀 Quick Start

### Basic Training
```bash
# Start distributed training with default settings
dspy-agent train --episodes 1000 --workers 4 --gpus 1 --framework protein

# Run hyperparameter sweep
dspy-agent sweep --trials 100 --framework protein --max-concurrent 4

# Setup judge models
dspy-agent judge --model-type ensemble --models transformer,dspy,openai
```

### Advanced Training
```bash
# Large-scale distributed training
dspy-agent train \
  --episodes 10000 \
  --workers 32 \
  --gpus 8 \
  --framework protein \
  --auto-scaling \
  --global-objectives \
  --tensorboard \
  --wandb \
  --wandb-project "dspy-code-production"

# Hyperparameter sweep with custom configuration
dspy-agent sweep \
  --trials 500 \
  --framework carbs \
  --max-concurrent 16 \
  --timeout 48 \
  --wandb-project "dspy-code-sweeps"
```

## 🏗️ Architecture Overview

### Training Components

1. **Distributed Training System** (`distributed_trainer.py`)
   - PufferLib integration with Protein, Carbs, Ray, and CleanRL frameworks
   - Automatic scaling from single GPU to large clusters
   - Mixed precision training and model compilation
   - Gradient checkpointing and optimization

2. **Hyperparameter Sweeps** (`hyperparameter_sweeps.py`)
   - Distributed hyperparameter optimization
   - Support for all PufferLib frameworks
   - Automatic resource management
   - Weights & Biases integration

3. **Judge Models** (`judge_models.py`)
   - Transformer-based judge models
   - DSPy signature-based evaluation
   - OpenAI API integration
   - Ensemble judge models

4. **Global Objectives** (`global_objective.py`)
   - Query satisfaction optimization
   - Performance, safety, and efficiency objectives
   - Convergence detection
   - Reward shaping

5. **Training Orchestrator** (`training_orchestrator.py`)
   - Unified training pipeline
   - Real-time monitoring
   - Automatic checkpointing
   - Resource optimization

## 🔧 Configuration

### Distributed Training Configuration

```python
from dspy_agent.training.distributed_config import create_training_config

# Single GPU (4090) configuration
config = create_training_config(
    framework="protein",
    num_gpus=1,
    num_workers=8,
    num_envs=32,
    batch_size=32,
    mixed_precision=True,
    compile_model=True
)

# Multi-GPU configuration
config = create_training_config(
    framework="protein",
    num_gpus=4,
    num_workers=32,
    num_envs=128,
    batch_size=64,
    distributed=True
)

# Large cluster configuration
config = create_training_config(
    framework="protein",
    num_gpus=16,
    num_workers=128,
    num_envs=512,
    batch_size=128,
    auto_scaling=True,
    max_workers=100
)
```

### Hyperparameter Sweep Configuration

```python
from dspy_agent.training.hyperparameter_sweeps import create_sweep_config

config = create_sweep_config(
    sweep_name="dspy_code_protein_sweep",
    num_trials=100,
    framework="protein",
    max_concurrent_trials=8,
    timeout_hours=24
)
```

### Global Objective Configuration

```python
from dspy_agent.training.global_objective import GlobalObjectiveConfig, ObjectiveType

config = GlobalObjectiveConfig(
    primary_objective=ObjectiveType.QUERY_SATISFACTION,
    secondary_objectives=[
        ObjectiveType.PERFORMANCE_OPTIMIZATION,
        ObjectiveType.SAFETY_COMPLIANCE,
        ObjectiveType.EFFICIENCY_MAXIMIZATION
    ],
    reward_shaping=True,
    curriculum_learning=True
)
```

## 🎯 Training Frameworks

### Protein Framework
- **Best for**: High-performance distributed training
- **Features**: Advanced optimization, mixed precision, model compilation
- **Scaling**: Excellent for multi-GPU and cluster training
- **Use case**: Production training with maximum performance

```bash
dspy-agent train --framework protein --gpus 4 --workers 16
```

### Carbs Framework
- **Best for**: Research and experimentation
- **Features**: Flexible configuration, easy debugging
- **Scaling**: Good for single to multi-GPU training
- **Use case**: Hyperparameter sweeps and research

```bash
dspy-agent train --framework carbs --gpus 2 --workers 8
```

### Ray Framework
- **Best for**: Large-scale distributed training
- **Features**: Automatic scaling, fault tolerance
- **Scaling**: Excellent for cluster training
- **Use case**: Enterprise-scale training

```bash
dspy-agent train --framework ray --gpus 8 --workers 32
```

### CleanRL Framework
- **Best for**: Simple, clean implementations
- **Features**: Easy to understand and modify
- **Scaling**: Good for single GPU training
- **Use case**: Learning and experimentation

```bash
dspy-agent train --framework cleanrl --gpus 1 --workers 4
```

## 📊 Monitoring and Logging

### TensorBoard Integration
```bash
# Start training with TensorBoard
dspy-agent train --tensorboard

# View TensorBoard
tensorboard --logdir logs/training_orchestrator/tensorboard
```

### Weights & Biases Integration
```bash
# Start training with W&B
dspy-agent train --wandb --wandb-project "dspy-code-training"

# View results in W&B dashboard
# https://wandb.ai/your-username/dspy-code-training
```

### Custom Monitoring
```python
from dspy_agent.training.training_orchestrator import create_training_orchestrator

orchestrator = create_training_orchestrator()

# Get training status
status = orchestrator.get_training_status()
print(f"Current episode: {status['current_episode']}")
print(f"Best performance: {status['best_performance']}")
print(f"Convergence status: {status['convergence_status']}")
```

## 🔍 Hyperparameter Optimization

### Running Sweeps
```bash
# Basic sweep
dspy-agent sweep --trials 100 --framework protein

# Advanced sweep with custom configuration
dspy-agent sweep \
  --trials 500 \
  --framework carbs \
  --max-concurrent 16 \
  --timeout 48 \
  --wandb-project "dspy-code-sweeps"
```

### Custom Search Spaces
```python
from dspy_agent.training.hyperparameter_sweeps import HyperparameterSweepConfig

config = HyperparameterSweepConfig(
    sweep_name="custom_sweep",
    num_trials=200,
    search_space={
        'learning_rate': {
            'type': 'log_uniform',
            'min': 1e-5,
            'max': 1e-2
        },
        'batch_size': {
            'type': 'choice',
            'choices': [16, 32, 64, 128, 256]
        },
        'num_workers': {
            'type': 'choice',
            'choices': [4, 8, 16, 32]
        }
    }
)
```

## ⚖️ Judge Models

### Setup Judge Models
```bash
# Create ensemble judge model
dspy-agent judge --model-type ensemble --models transformer,dspy,openai

# Create single judge model
dspy-agent judge --model-type transformer

# Benchmark judge models
dspy-agent judge --benchmark
```

### Custom Judge Models
```python
from dspy_agent.training.judge_models import create_judge_model, create_ensemble_judge

# Create transformer judge
transformer_judge = create_judge_model("transformer")

# Create ensemble judge
ensemble_judge = create_ensemble_judge(["transformer", "dspy", "openai"])

# Score a query-response pair
score = ensemble_judge.score(
    query="Implement a REST API endpoint",
    response="Here's a Flask endpoint...",
    context={"codebase": "Python web application"}
)

print(f"Overall score: {score.overall_score:.3f}")
print(f"Explanation: {score.explanation}")
```

## 🎯 Global Objectives

### Objective Types
- **Query Satisfaction**: How well the agent satisfies the user's query
- **Performance Optimization**: Code efficiency and performance
- **Safety Compliance**: Security and safety considerations
- **Efficiency Maximization**: Resource usage optimization
- **Quality Assurance**: Code quality and maintainability

### Custom Objectives
```python
from dspy_agent.training.global_objective import GlobalObjectiveSystem, ObjectiveType

system = GlobalObjectiveSystem()

# Evaluate global objective
result = system.evaluate_global_objective(
    query="Implement user authentication",
    response="Here's a secure authentication system...",
    context={"security_requirements": "high"}
)

print(f"Overall score: {result.overall_score:.3f}")
print(f"Weighted score: {result.weighted_score:.3f}")
print(f"Convergence status: {result.convergence_status}")
```

## 🚀 Scaling Strategies

### Single GPU (4090)
```bash
dspy-agent train \
  --gpus 1 \
  --workers 8 \
  --framework protein \
  --mixed-precision \
  --compile-model
```

### Multi-GPU
```bash
dspy-agent train \
  --gpus 4 \
  --workers 32 \
  --framework protein \
  --distributed \
  --auto-scaling
```

### Large Cluster
```bash
dspy-agent train \
  --gpus 16 \
  --workers 128 \
  --framework ray \
  --auto-scaling \
  --max-workers 100
```

## 🔧 Advanced Configuration

### Custom Training Configuration
```python
from dspy_agent.training.training_orchestrator import TrainingOrchestratorConfig

config = TrainingOrchestratorConfig(
    training_type="distributed",
    num_episodes=10000,
    num_workers=32,
    num_gpus=8,
    framework="protein",
    run_sweeps=True,
    use_global_objectives=True,
    auto_scaling=True,
    curriculum_learning=True,
    meta_learning=True,
    transfer_learning=True
)
```

### Environment Variables
```bash
# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export DSPY_CACHE_DIR=/tmp/dspy_cache
```

## 📈 Performance Optimization

### Hardware Optimization
- **GPU Memory**: Use mixed precision training
- **CPU**: Optimize thread count and memory allocation
- **Storage**: Use fast SSD for checkpoint storage
- **Network**: Use high-bandwidth network for distributed training

### Software Optimization
- **Model Compilation**: Enable `torch.compile()` for faster execution
- **Gradient Checkpointing**: Reduce memory usage
- **Data Loading**: Optimize data loader workers and pin memory
- **Mixed Precision**: Use FP16 for faster training

### Monitoring Performance
```python
# Monitor training performance
status = orchestrator.get_training_status()
print(f"Current episode: {status['current_episode']}")
print(f"Best performance: {status['best_performance']}")
print(f"Performance history: {status['performance_history']}")
print(f"Convergence status: {status['convergence_status']}")
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size and enable gradient checkpointing
   dspy-agent train --batch-size 16 --gradient-checkpointing
   ```

2. **Distributed Training Issues**
   ```bash
   # Check network connectivity
   dspy-agent train --framework cleanrl  # Fallback to single-GPU
   ```

3. **Hyperparameter Sweep Failures**
   ```bash
   # Reduce concurrent trials
   dspy-agent sweep --max-concurrent 2
   ```

4. **Judge Model Errors**
   ```bash
   # Test judge models individually
   dspy-agent judge --model-type transformer --benchmark
   ```

### Debug Mode
```bash
# Enable debug logging
export DSPY_DEBUG=1
dspy-agent train --framework protein
```

## 📚 Examples

### Complete Training Pipeline
```bash
# 1. Setup judge models
dspy-agent judge --model-type ensemble

# 2. Run hyperparameter sweep
dspy-agent sweep --trials 100 --framework protein

# 3. Start distributed training
dspy-agent train \
  --episodes 10000 \
  --workers 32 \
  --gpus 8 \
  --framework protein \
  --global-objectives \
  --auto-scaling \
  --tensorboard \
  --wandb
```

### Research Workflow
```bash
# 1. Quick sweep for initial exploration
dspy-agent sweep --trials 50 --framework carbs

# 2. Detailed sweep with best parameters
dspy-agent sweep --trials 200 --framework protein --max-concurrent 8

# 3. Final training with optimal configuration
dspy-agent train --framework protein --gpus 4 --workers 16
```

## 🎉 Conclusion

This distributed training system provides:

- **Full PufferLib Integration**: All frameworks (Protein, Carbs, Ray, CleanRL)
- **Automatic Scaling**: From single 4090 to large clusters
- **Hyperparameter Optimization**: Distributed sweeps with all frameworks
- **Judge Models**: Comprehensive evaluation system
- **Global Objectives**: Query satisfaction optimization
- **Monitoring**: TensorBoard and W&B integration
- **Performance**: Optimized for maximum throughput

The system is designed to scale from a single GPU to massive distributed clusters while maintaining optimal performance and providing comprehensive monitoring and evaluation capabilities.
