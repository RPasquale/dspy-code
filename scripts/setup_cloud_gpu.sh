#!/bin/bash
# Setup script for cloud GPU integration
# This script helps configure the cloud GPU integration system

set -euo pipefail

echo "🚀 Setting up Cloud GPU Integration for DSPy Agent"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if required Python packages are available
echo "📦 Checking Python dependencies..."
python3 -c "import requests, yaml" 2>/dev/null || {
    echo "📦 Installing required Python packages..."
    pip install requests pyyaml
}

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /tmp/datasets
mkdir -p /tmp/models/agent
mkdir -p /tmp/logs
mkdir -p deploy/slurm

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/train_agent.py
chmod +x scripts/unified_training_orchestrator.py
chmod +x deploy/slurm/run_agent_training.sh

# Check for available backends
echo "🔍 Checking available backends..."
python3 scripts/train_agent.py --list

# Setup instructions
echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Set up API keys for cloud providers:"
echo "   export PRIME_INTELLECT_API_KEY='your_api_key_here'"
echo "   export RUNPOD_API_KEY='your_api_key_here'"
echo "   export NEBIUS_API_KEY='your_api_key_here'"
echo "   export COREWEAVE_API_KEY='your_api_key_here'"
echo ""
echo "2. Test the system:"
echo "   python scripts/train_agent.py --list"
echo ""
echo "3. Start training:"
echo "   python scripts/train_agent.py grpo --module orchestrator --monitor"
echo ""
echo "4. For Prime Intellect specifically:"
echo "   - Visit: https://app.primeintellect.ai/dashboard/create-cluster"
echo "   - Get your API key from the dashboard"
echo "   - Set: export PRIME_INTELLECT_API_KEY='your_key'"
echo ""
echo "🎉 Ready to train with cloud GPUs!"
