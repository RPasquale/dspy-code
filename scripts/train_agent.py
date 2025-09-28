#!/usr/bin/env python3
"""
Simple wrapper for unified agent training
Automatically detects environment and uses appropriate backend (local Slurm or cloud GPU)
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_training_orchestrator import UnifiedTrainingOrchestrator, TrainingRequest, TrainingBackend

def main():
    parser = argparse.ArgumentParser(description="Train DSPy Agent - Auto-detects best available backend")
    parser.add_argument('method', choices=['grpo', 'gepa', 'teleprompt', 'rl', 'codegen', 'orchestrator', 'prefs'],
                       help='Training method to use')
    parser.add_argument('--module', default='orchestrator',
                       help='Module to train (orchestrator, context, task, code, etc.)')
    parser.add_argument('--model', default='gpt2',
                       help='Model to use for training')
    parser.add_argument('--dataset', default='/tmp/datasets/training.jsonl',
                       help='Path to training dataset')
    parser.add_argument('--output', default='/tmp/models/agent',
                       help='Output directory for trained model')
    parser.add_argument('--workspace', default='/tmp',
                       help='Workspace directory')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--memory', type=int, default=8,
                       help='Minimum GPU memory in GB')
    parser.add_argument('--price-limit', type=float, default=10.0,
                       help='Maximum price per hour for cloud GPUs')
    parser.add_argument('--backend', default='auto',
                       choices=['auto', 'local', 'prime', 'runpod', 'nebius', 'coreweave'],
                       help='Force specific backend (auto-detects by default)')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor training progress')
    parser.add_argument('--logs', action='store_true',
                       help='Show training logs')
    parser.add_argument('--list', action='store_true',
                       help='List available backends')
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = UnifiedTrainingOrchestrator()
    
    if args.list:
        backends = orchestrator.get_available_backends()
        print("Available training backends:")
        for backend in backends:
            print(f"  ✓ {backend.value}")
        
        if not backends:
            print("  No backends available!")
            print("\nTo enable backends:")
            print("  Local Slurm: Install and configure Slurm")
            print("  Prime Intellect: Set PRIME_INTELLECT_API_KEY environment variable")
            print("  RunPod: Set RUNPOD_API_KEY environment variable")
            print("  Nebius: Set NEBIUS_API_KEY environment variable")
            print("  CoreWeave: Set COREWEAVE_API_KEY environment variable")
        return
    
    # Map backend argument
    backend_map = {
        'auto': TrainingBackend.AUTO_DETECT,
        'local': TrainingBackend.LOCAL_SLURM,
        'prime': TrainingBackend.PRIME_INTELLECT,
        'runpod': TrainingBackend.RUNPOD,
        'nebius': TrainingBackend.NEBIUS,
        'coreweave': TrainingBackend.COREWEAVE
    }
    
    # Create training request
    request = TrainingRequest(
        training_method=args.method,
        module_name=args.module,
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        workspace_dir=args.workspace,
        batch_size=8,
        learning_rate=1e-5,
        max_steps=1000,
        epochs=1,
        gpu_requirements={
            'gpu_count': args.gpus,
            'min_memory_gb': args.memory,
            'min_storage_gb': 100,
            'max_price_per_hour': args.price_limit
        },
        backend=backend_map[args.backend]
    )
    
    try:
        # Start training
        print(f"Starting {args.method} training for {args.module} module...")
        result = orchestrator.start_training(request)
        
        print(f"\n✅ Training started successfully!")
        print(f"   Job ID: {result.job_id}")
        print(f"   Backend: {result.backend.value}")
        print(f"   Status: {result.status}")
        
        if result.instance_id:
            print(f"   Instance ID: {result.instance_id}")
        
        if args.monitor:
            print(f"\n📊 Monitoring training progress...")
            while True:
                status = orchestrator.monitor_training(result.job_id)
                print(f"   Status: {status.get('status', 'unknown')}")
                
                if status.get('progress'):
                    print(f"   Progress: {status['progress']}%")
                
                if status.get('cost'):
                    print(f"   Cost: ${status['cost']:.2f}")
                
                if status.get('status') in ['completed', 'failed', 'stopped']:
                    break
                
                import time
                time.sleep(30)  # Check every 30 seconds
        
        if args.logs:
            print(f"\n📋 Training logs:")
            logs = orchestrator.get_training_logs(result.job_id)
            print(logs)
        
        print(f"\n🎉 Training completed! Check output in: {args.output}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
