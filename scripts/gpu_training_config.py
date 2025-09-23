#!/usr/bin/env python3
"""
GPU-Optimized Training Configuration for NVIDIA 4090

This script configures the training environment for optimal performance
on NVIDIA 4090 GPU with CUDA acceleration.
"""

import os
import torch
import psutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class GPUOptimizedTrainingConfig:
    """Configuration for GPU-optimized training on NVIDIA 4090."""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count()
        self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_available else "No GPU"
        
        # NVIDIA 4090 specifications
        self.target_gpu = "NVIDIA GeForce RTX 4090"
        self.vram_gb = 24  # 4090 has 24GB VRAM
        self.cuda_cores = 16384  # 4090 has 16384 CUDA cores
        
        logger.info(f"GPU Available: {self.gpu_available}")
        logger.info(f"GPU Name: {self.gpu_name}")
        logger.info(f"GPU Count: {self.gpu_count}")
    
    def configure_environment(self):
        """Configure environment variables for optimal GPU training."""
        
        # CUDA configuration
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async execution
        
        # PyTorch configuration
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"  # RTX 4090 architecture
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable CUDA DSA
        
        # Memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Training optimization
        os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count())
        os.environ["MKL_NUM_THREADS"] = str(psutil.cpu_count())
        
        logger.info("Environment configured for GPU training")
    
    def get_optimal_batch_sizes(self) -> dict:
        """Get optimal batch sizes for different model sizes."""
        return {
            "small_model": {
                "batch_size": 64,
                "gradient_accumulation": 1,
                "max_memory_gb": 4
            },
            "medium_model": {
                "batch_size": 32,
                "gradient_accumulation": 2,
                "max_memory_gb": 8
            },
            "large_model": {
                "batch_size": 16,
                "gradient_accumulation": 4,
                "max_memory_gb": 16
            },
            "xlarge_model": {
                "batch_size": 8,
                "gradient_accumulation": 8,
                "max_memory_gb": 24
            }
        }
    
    def get_training_hyperparameters(self) -> dict:
        """Get optimized hyperparameters for RTX 4090."""
        return {
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "warmup_steps": 1000,
            "max_grad_norm": 1.0,
            "adam_epsilon": 1e-8,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "mixed_precision": True,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 8,
            "dataloader_pin_memory": True
        }
    
    def configure_torch_settings(self):
        """Configure PyTorch settings for optimal performance."""
        if not self.gpu_available:
            logger.warning("CUDA not available, using CPU")
            return
        
        # Set device
        self.device = torch.device("cuda:0")
        
        # Configure CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Memory management
        torch.cuda.empty_cache()
        
        # Set memory fraction (use 90% of available VRAM)
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        logger.info(f"PyTorch configured for device: {self.device}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    def get_memory_info(self) -> dict:
        """Get GPU memory information."""
        if not self.gpu_available:
            return {"error": "No GPU available"}
        
        memory_info = {
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "allocated_memory_gb": torch.cuda.memory_allocated() / 1024**3,
            "cached_memory_gb": torch.cuda.memory_reserved() / 1024**3,
            "free_memory_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
        }
        
        return memory_info
    
    def optimize_for_training(self, model_size: str = "medium_model") -> dict:
        """Optimize configuration for specific model size."""
        config = {
            "device": self.device if self.gpu_available else torch.device("cpu"),
            "batch_config": self.get_optimal_batch_sizes()[model_size],
            "hyperparameters": self.get_training_hyperparameters(),
            "memory_info": self.get_memory_info()
        }
        
        # Adjust batch size based on available memory
        if self.gpu_available:
            free_memory = config["memory_info"]["free_memory_gb"]
            max_memory = config["batch_config"]["max_memory_gb"]
            
            if free_memory < max_memory:
                # Reduce batch size if not enough memory
                config["batch_config"]["batch_size"] = max(1, config["batch_config"]["batch_size"] // 2)
                config["batch_config"]["gradient_accumulation"] *= 2
                logger.warning(f"Reduced batch size due to memory constraints")
        
        return config
    
    def setup_mixed_precision(self):
        """Setup mixed precision training for better performance."""
        if not self.gpu_available:
            return None
        
        from torch.cuda.amp import GradScaler, autocast
        
        scaler = GradScaler()
        
        logger.info("Mixed precision training configured")
        return scaler
    
    def monitor_gpu_usage(self):
        """Monitor GPU usage during training."""
        if not self.gpu_available:
            return
        
        import GPUtil
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                logger.info(f"GPU Usage: {gpu.load*100:.1f}%")
                logger.info(f"GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
                logger.info(f"GPU Temperature: {gpu.temperature}°C")
        except ImportError:
            logger.warning("GPUtil not available for GPU monitoring")
        except Exception as e:
            logger.warning(f"Error monitoring GPU: {e}")

def setup_gpu_training():
    """Setup GPU training environment."""
    config = GPUOptimizedTrainingConfig()
    
    # Configure environment
    config.configure_environment()
    config.configure_torch_settings()
    
    # Get optimized configuration
    training_config = config.optimize_for_training()
    
    # Setup mixed precision
    scaler = config.setup_mixed_precision()
    
    logger.info("GPU training environment setup complete")
    
    return {
        "config": config,
        "training_config": training_config,
        "scaler": scaler
    }

if __name__ == "__main__":
    # Test GPU configuration
    logging.basicConfig(level=logging.INFO)
    
    setup = setup_gpu_training()
    
    print("\n" + "="*60)
    print("GPU TRAINING CONFIGURATION")
    print("="*60)
    print(f"GPU Available: {setup['config'].gpu_available}")
    print(f"GPU Name: {setup['config'].gpu_name}")
    print(f"Device: {setup['training_config']['device']}")
    print(f"Batch Config: {setup['training_config']['batch_config']}")
    print(f"Memory Info: {setup['training_config']['memory_info']}")
    print("="*60)
